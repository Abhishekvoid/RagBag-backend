"""Integration tests: isolated Redis, real Lua and independent processes.

Set TEST_REDIS_URL to a disposable Redis instance to run these checks.
"""
import asyncio
import multiprocessing
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase, skipUnless, mock
import redis
from utils.circuit_breaker import CircuitBreaker, ServiceUnavailable
from utils.llm_load_control import SlotManager, SystemOverload
from utils.service_control import async_attempt, sync_attempt

URL = os.getenv("TEST_REDIS_URL")

def occupy(url, name, acquired, release):
    connection = redis.from_url(url, decode_responses=True)
    slots = SlotManager(2, 0, service_name=name, connection=connection)
    try:
        with slots.sync_slot():
            acquired.put("admitted")
            release.wait(10)
    except SystemOverload:
        acquired.put("rejected")

@skipUnless(URL, "Set TEST_REDIS_URL for real Redis integration checks")
class SharedControlTests(TestCase):
    def setUp(self):
        self.redis = redis.from_url(URL, decode_responses=True, socket_timeout=1)
        self.name = "test-" + uuid.uuid4().hex
        self.slots = SlotManager(2, 2, service_name=self.name, connection=self.redis)
        self.cb = CircuitBreaker(self.name, failure_threshold=2, cooldown=0,
                                 connection=self.redis, probe_seconds=10)
        self.addCleanup(self.redis.delete, self.slots.key, self.slots.queue_key,
                        self.cb.key, self.cb.probe_key)

    def open(self):
        for _ in range(2):
            self.cb.finish(self.cb.acquire(), "failure")

    def test_capacity_shared_across_four_processes(self):
        ctx = multiprocessing.get_context("spawn")
        queue, release = ctx.Queue(), ctx.Event()
        processes = [ctx.Process(target=occupy, args=(URL, self.name, queue, release)) for _ in range(4)]
        try:
            for process in processes:
                process.start()
            results = [queue.get(timeout=20) for _ in processes]
            self.assertEqual(results.count("admitted"), 2)
            self.assertEqual(results.count("rejected"), 2)
        finally:
            release.set()
            for process in processes:
                process.join(15)
                if process.is_alive():
                    process.terminate()
                    process.join()
        self.assertEqual(self.redis.zcard(self.slots.key), 0)

    def test_only_one_canary_and_stale_success_cannot_close_it(self):
        stale = self.cb.acquire()
        self.open()
        before = self.redis.hgetall(self.cb.key)
        self.assertTrue(self.cb.is_open())
        self.assertEqual(self.cb.state(), "open")
        self.assertEqual(self.redis.hgetall(self.cb.key), before)
        def admit(_):
            try:
                return self.cb.acquire()
            except ServiceUnavailable:
                return None
        with ThreadPoolExecutor(max_workers=10) as pool:
            permits = [p for p in pool.map(admit, range(10)) if p]
        self.assertEqual(len(permits), 1)
        self.assertTrue(permits[0].probe)
        self.assertFalse(self.cb.finish(stale, "success"))
        self.assertEqual(self.cb.state(), "half-open")
        self.assertTrue(self.cb.finish(permits[0], "success"))
        self.assertEqual(self.cb.state(), "closed")
        self.assertFalse(self.cb.finish(stale, "failure"))

    def test_expired_probe_owner_cannot_close_new_generation(self):
        self.open()
        first = self.cb.acquire()
        self.redis.pexpire(self.cb.probe_key, 1)
        time.sleep(.02)
        second = self.cb.acquire()
        self.assertFalse(self.cb.finish(first, "success"))
        self.assertEqual(self.cb.state(), "half-open")
        self.assertTrue(self.cb.finish(second, "failure"))
        self.assertEqual(self.cb.state(), "open")

    def test_dead_worker_lease_expires_and_live_lease_renews(self):
        slots = SlotManager(1, 0, service_name=self.name, connection=self.redis, lease_seconds=.15)
        self.assertTrue(slots._acquire("dead-owner"))
        time.sleep(.2)
        with slots.sync_slot():
            time.sleep(.35)
            with self.assertRaises(SystemOverload):
                slots._acquire("another-owner")
        self.assertEqual(self.redis.zcard(slots.key), 0)

    def test_sync_probe_is_renewed_during_long_call(self):
        self.cb.probe_seconds = 1
        self.open()
        with sync_attempt(self.slots, self.cb):
            time.sleep(1.3)
            with self.assertRaises(ServiceUnavailable):
                self.cb.acquire()
        self.assertEqual(self.cb.state(), "closed")

    def test_queue_overload_does_not_count_as_downstream_failure(self):
        slots = SlotManager(1, 0, service_name=self.name, connection=self.redis)
        with slots.sync_slot():
            with self.assertRaises(SystemOverload):
                with sync_attempt(slots, self.cb):
                    self.fail("No free slot")
        self.assertEqual(self.redis.hget(self.cb.key, "failures"), None)

    def test_queued_permit_is_rejected_after_circuit_opens(self):
        slots = SlotManager(1, 1, service_name=self.name, connection=self.redis)
        async def run():
            async def waiting():
                async with async_attempt(slots, self.cb):
                    self.fail("Stale queued permit must not call provider")
            async with slots.slot():
                task = asyncio.create_task(waiting())
                for _ in range(100):
                    if self.redis.zcard(slots.queue_key):
                        break
                    await asyncio.sleep(.01)
                self.open()
            with self.assertRaises(ServiceUnavailable):
                await task
        asyncio.run(run())

    def test_redis_failure_prevents_provider_calls(self):
        broken = mock.Mock()
        broken.eval.side_effect = redis.ConnectionError("offline")
        slots = SlotManager(connection=broken)
        breaker = CircuitBreaker(self.name, connection=broken)
        with self.assertRaises(ServiceUnavailable):
            with sync_attempt(slots, breaker):
                self.fail("provider must not be reached")
        with self.assertRaises(SystemOverload):
            with slots.sync_slot():
                self.fail("provider must not be reached")

    def test_cancellation_releases_active_slot_and_probe(self):
        self.open()
        async def run():
            entered = asyncio.Event()
            async def call():
                async with async_attempt(self.slots, self.cb):
                    entered.set()
                    await asyncio.sleep(30)
            task = asyncio.create_task(call())
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(run())
        self.assertEqual(self.redis.zcard(self.slots.key), 0)
        self.assertEqual(self.redis.zcard(self.slots.queue_key), 0)
        self.assertFalse(self.redis.exists(self.cb.probe_key))
        self.assertEqual(self.cb.state(), "open")

    def test_queue_bound_and_cancelled_waiter_removed(self):
        slots = SlotManager(1, 1, service_name=self.name, connection=self.redis)
        async def run():
            async def waiting():
                async with slots.slot():
                    self.fail("should still be waiting")
            async with slots.slot():
                task = asyncio.create_task(waiting())
                for _ in range(100):
                    if self.redis.zcard(slots.queue_key):
                        break
                    await asyncio.sleep(.01)
                with self.assertRaises(SystemOverload):
                    async with slots.slot():
                        pass
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(self.redis.zcard(slots.queue_key), 0)
        asyncio.run(run())

    def test_half_open_embedding_uses_one_attempt_and_records_failure(self):
        from utils.embedding import EmbeddingClient
        from tenacity import wait_none
        import httpx
        self.open()
        client = EmbeddingClient()
        client.cb, client.slot_manager = self.cb, self.slots
        post = mock.AsyncMock(side_effect=httpx.ConnectError("offline"))
        with mock.patch.object(client.client, "post", post), mock.patch.object(client.embed_texts.retry, "wait", wait_none()):
            with self.assertRaises(ServiceUnavailable):
                asyncio.run(client.embed_texts(["test text"]))
        self.assertEqual(post.await_count, 1)
        self.assertEqual(self.cb.state(), "open")

    def test_closed_embedding_failures_open_breaker(self):
        from utils.embedding import EmbeddingClient
        from tenacity import wait_none
        import httpx
        self.cb.cooldown = 60
        client = EmbeddingClient()
        client.cb, client.slot_manager = self.cb, self.slots
        post = mock.AsyncMock(side_effect=httpx.ConnectError("offline"))
        with mock.patch.object(client.client, "post", post), mock.patch.object(client.embed_texts.retry, "wait", wait_none()):
            with self.assertRaises(ServiceUnavailable):
                asyncio.run(client.embed_texts(["test text"]))
        self.assertEqual(post.await_count, 2)
        self.assertEqual(self.cb.state(), "open")
