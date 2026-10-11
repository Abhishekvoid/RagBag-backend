"""Shared Redis admission for HTTP workers and Celery processes."""
import asyncio
import os
import time
import uuid
import logging
import threading
from contextlib import contextmanager, asynccontextmanager
import redis
from .circuit_breaker import redis_client, ServiceUnavailable
from .deadline import check_deadline, timeout_for

logger = logging.getLogger(__name__)

class SystemOverload(ServiceUnavailable):
    pass

_ACQUIRE = """
redis.replicate_commands()
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now)
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[2]) then
    redis.call('ZADD', KEYS[1], now + tonumber(ARGV[4]), ARGV[1])
    redis.call('EXPIRE', KEYS[1], math.ceil(tonumber(ARGV[4])) + 1)
    redis.call('ZREM', KEYS[2], ARGV[1])
    return 1
end
if not redis.call('ZSCORE', KEYS[2], ARGV[1]) and redis.call('ZCARD', KEYS[2]) >= tonumber(ARGV[3]) then return -1 end
redis.call('ZADD', KEYS[2], now + tonumber(ARGV[5]) + 2, ARGV[1])
redis.call('EXPIRE', KEYS[2], math.ceil(tonumber(ARGV[5])) + 3)
return 0
"""
_RENEW = """
redis.replicate_commands()
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local expiry = tonumber(redis.call('ZSCORE', KEYS[1], ARGV[1]) or '0')
if expiry <= now then return 0 end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[1])
redis.call('EXPIRE', KEYS[1], math.ceil(tonumber(ARGV[2])) + 1)
return 1
"""
_RELEASE = "return redis.call('ZREM', KEYS[1], ARGV[1]) + redis.call('ZREM', KEYS[2], ARGV[1])"

async def redis_call(function, *args):
    # Finish an in-flight admission before cancellation releases its token.
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            pass
        raise

@contextmanager
def maintain_lease(renew, interval):
    stopped, lost = threading.Event(), threading.Event()
    def run():
        while not stopped.wait(interval):
            try:
                if renew():
                    continue
            except redis.RedisError:
                pass
            lost.set()
            return
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        yield
        if lost.is_set():
            raise SystemOverload("Service lease was lost")
    finally:
        stopped.set()
        worker.join(timeout=2)

class SlotManager:
    def __init__(self, max_concurrent=20, max_queue=100, *, service_name=None,
                 connection=None, lease_seconds=90, wait_seconds=5):
        if max_concurrent < 1 or max_queue < 0 or lease_seconds <= 0 or wait_seconds <= 0:
            raise ValueError("Invalid service admission limits")
        self.redis = connection if connection is not None else redis_client
        namespace = os.getenv("SERVICE_CONTROL_NAMESPACE", "studywise")
        service_name = service_name or uuid.uuid4().hex
        self.key = f"{namespace}:control:{{{service_name}}}:slots:v2"
        self.queue_key = self.key + ":waiting"
        self.max_concurrent, self.max_queue = max_concurrent, max_queue
        self.lease_seconds, self.wait_seconds = lease_seconds, wait_seconds

    def _acquire(self, token):
        check_deadline()
        try:
            result = self.redis.eval(_ACQUIRE, 2, self.key, self.queue_key, token,
                self.max_concurrent, self.max_queue, self.lease_seconds, self.wait_seconds)
        except redis.RedisError as exc:
            raise SystemOverload("Shared service admission unavailable") from exc
        if result == -1:
            raise SystemOverload("Service queue is full")
        return result == 1

    def _release(self, token):
        try:
            self.redis.eval(_RELEASE, 2, self.key, self.queue_key, token)
        except redis.RedisError:
            logger.warning("Service lease release failed; lease will expire", exc_info=True)

    def _heartbeat(self, token):
        return maintain_lease(lambda: self.redis.eval(_RENEW, 1, self.key, token, self.lease_seconds),
                              self.lease_seconds / 3)

    @contextmanager
    def sync_slot(self):
        token = uuid.uuid4().hex
        end = time.monotonic() + timeout_for(self.wait_seconds)
        try:
            while not self._acquire(token):
                check_deadline()
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise SystemOverload("Service queue wait expired")
                time.sleep(min(0.05, remaining))
            check_deadline()
            with self._heartbeat(token):
                yield
        finally:
            self._release(token)

    @asynccontextmanager
    async def slot(self):
        token = uuid.uuid4().hex
        end = time.monotonic() + timeout_for(self.wait_seconds)
        try:
            while not await redis_call(self._acquire, token):
                check_deadline()
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise SystemOverload("Service queue wait expired")
                await asyncio.sleep(min(0.05, remaining))
            check_deadline()
            with self._heartbeat(token):
                yield
        finally:
            await redis_call(self._release, token)

def _manager(name, env, capacity, queue):
    return SlotManager(int(os.getenv(env + "_MAX_CONCURRENT", capacity)),
        int(os.getenv(env + "_MAX_QUEUE", queue)), service_name=name)

tei_slot_manager = _manager("tei-embedding", "EMBEDDING", 3, 5)
llm_slot_manager = _manager("llm", "LLM", 20, 100)
vision_slot_manager = _manager("vision", "VISION", 1, 5)
rerank_slot_manager = _manager("rerank", "RERANK", 3, 5)
