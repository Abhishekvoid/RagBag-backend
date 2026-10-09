"""One clock for the whole request, including retries and queueing."""
import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, mock
from contextlib import contextmanager

from utils.deadline import (
    DeadlineExceeded, RequestDeadline, current_deadline, deadline_scope,
    stop_at_deadline, timeout_for,
)


class BudgetTests(TestCase):
    def test_attempts_share_a_monotonic_budget(self):
        with mock.patch("utils.deadline.time.monotonic", return_value=100) as clock:
            deadline = RequestDeadline(30)
            with deadline_scope(deadline):
                self.assertEqual(timeout_for(10), 10)
                clock.return_value = 127
                self.assertEqual(timeout_for(45), 3)
                self.assertTrue(stop_at_deadline(SimpleNamespace(upcoming_sleep=4)))
                clock.return_value = 130
                with self.assertRaises(DeadlineExceeded):
                    timeout_for(10)

    def test_ingestion_scope_is_independent_and_restored(self):
        chat = RequestDeadline(30)
        with deadline_scope(chat):
            with deadline_scope(RequestDeadline(900)):
                self.assertGreater(current_deadline().remaining, 899)
            self.assertIs(current_deadline(), chat)
        self.assertIsNone(current_deadline())

    def test_postgres_timeout_shrinks_and_cleanup_survives_expiry(self):
        from utils.deadline import database_deadline
        connection = mock.MagicMock(vendor="postgresql")
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = ["5s"]
        wrappers = []
        @contextmanager
        def register(wrapper):
            wrappers.append(wrapper)
            yield
        connection.execute_wrapper.side_effect = register
        budget = RequestDeadline(.5)
        execute = mock.Mock()
        with mock.patch("django.db.connection", connection), deadline_scope(budget):
            with database_deadline():
                wrappers[0](execute, "SELECT 1", None, False, {})
                command = execute.call_args_list[0].args[0]
                self.assertTrue(command.startswith("SET statement_timeout = "))
                self.assertLessEqual(int(command.rsplit(" ", 1)[1]), 500)
                budget.expires_at -= 1
                with self.assertRaises(DeadlineExceeded):
                    wrappers[0](execute, "INSERT INTO messages VALUES (1)", None, False, {})
                wrappers[0](execute, 'ROLLBACK TO SAVEPOINT "turn"', None, False, {})
            cursor.execute.assert_called_with("SELECT set_config('statement_timeout', %s, false)", ["5s"])
        self.assertFalse(any("INSERT" in call.args[0] for call in execute.call_args_list))

    def test_postgres_statement_cancellation_is_classified_as_deadline(self):
        from utils.deadline import database_deadline
        connection = mock.MagicMock(vendor="postgresql")
        connection.cursor.return_value.__enter__.return_value.fetchone.return_value = ["5s"]
        wrappers = []
        @contextmanager
        def register(wrapper):
            wrappers.append(wrapper)
            yield
        connection.execute_wrapper.side_effect = register
        error = RuntimeError("statement cancelled")
        error.pgcode = "57014"
        with mock.patch("django.db.connection", connection), deadline_scope(RequestDeadline(30)):
            with database_deadline():
                with self.assertRaises(DeadlineExceeded):
                    wrappers[0](mock.Mock(side_effect=[None, error]), "SELECT 1", None, False, {})


class RetryBudgetTests(IsolatedAsyncioTestCase):
    async def test_backoff_cannot_outlive_request(self):
        from utils.llm_wrapper import _call_llm_with_retry
        import httpx
        import openai
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=mock.AsyncMock(side_effect=openai.APITimeoutError(
                request=httpx.Request("POST", "https://example.invalid"))),
        )))
        with deadline_scope(RequestDeadline(.05)):
            with self.assertRaises((DeadlineExceeded, openai.APITimeoutError)):
                await _call_llm_with_retry(client, [], model="test", timeout=45)
        self.assertEqual(client.chat.completions.create.await_count, 1)
        self.assertLessEqual(client.chat.completions.create.call_args.kwargs["timeout"], .05)

    async def test_pipeline_deadline_cancels_stage_and_releases_queue_slot(self):
        from accounts.rag_pipeline import RagPipeline, PipelineOutcome
        from utils.llm_load_control import SlotManager
        pipeline = object.__new__(RagPipeline)
        slots = SlotManager(1, 2)
        cancelled = asyncio.Event()

        async def slow(*args):
            async with slots.slot():
                try:
                    await asyncio.sleep(30)
                finally:
                    cancelled.set()

        with mock.patch.object(pipeline, "contextualize_and_route", side_effect=slow):
            with deadline_scope(RequestDeadline(.03)):
                result = await pipeline.run("Explain gravity", [], "chapter", 1)
        self.assertEqual(result.outcome, PipelineOutcome.DEADLINE_EXCEEDED)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(slots._waiting, 0)
        async with asyncio.timeout(.1):
            async with slots.slot():
                pass

    async def test_concurrent_requests_do_not_share_budget(self):
        async def read_budget(seconds):
            with deadline_scope(RequestDeadline(seconds)):
                await asyncio.sleep(0)
                return timeout_for(1000)
        short, long = await asyncio.gather(read_budget(1), read_budget(900))
        self.assertLessEqual(short, 1)
        self.assertGreater(long, 899)

    async def test_retry_recalculates_timeout_after_attempt(self):
        from utils.llm_wrapper import _call_llm_with_retry
        from tenacity import wait_none
        from accounts.tests.test_pipeline_outcomes import completion
        import httpx
        import openai
        budget = RequestDeadline(10)
        timeouts = []
        async def create(**kwargs):
            timeouts.append(kwargs["timeout"])
            if len(timeouts) == 1:
                budget.expires_at -= 4
                raise openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
            return completion("ok")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with deadline_scope(budget), mock.patch.object(_call_llm_with_retry.retry, "wait", wait_none()):
            await _call_llm_with_retry(client, [], model="test", timeout=45)
        self.assertEqual(len(timeouts), 2)
        self.assertGreater(timeouts[0] - timeouts[1], 3.9)

    async def test_waiting_for_llm_slot_counts_toward_deadline(self):
        from utils.llm_load_control import SlotManager
        from utils.llm_gateway import ask_llm
        slots = SlotManager(1, 2)
        with mock.patch("utils.llm_gateway.llm_slot_manager", slots), \
             mock.patch("utils.llm_gateway.llm_circuit_breaker") as breaker, \
             mock.patch("utils.llm_gateway._call_llm_with_retry") as provider:
            breaker.is_open.return_value = False
            async with slots.slot():
                with deadline_scope(RequestDeadline(.03)):
                    with self.assertRaises(DeadlineExceeded):
                        await ask_llm(None, [], model="test")
            self.assertEqual(slots._waiting, 0)
            provider.assert_not_called()

    async def test_native_pinecone_read_is_cancelled_and_clients_are_closed(self):
        from accounts import ai_clients, rag_service
        from utils.deadline import within_deadline
        client, index = mock.MagicMock(), mock.MagicMock()
        client.__aenter__.return_value = client
        index.__aenter__.return_value = index
        client.index = mock.AsyncMock(return_value=index)
        cancelled = asyncio.Event()
        async def blocked(**kwargs):
            self.assertLessEqual(kwargs["timeout"], .03)
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()
        index.query = mock.AsyncMock(side_effect=blocked)
        with mock.patch.object(ai_clients, "async_pinecone_client", return_value=client):
            with deadline_scope(RequestDeadline(.03)):
                with self.assertRaises(DeadlineExceeded):
                    await within_deadline(rag_service._query_dense)([.1], {}, 5)
        self.assertTrue(cancelled.is_set())
        index.__aexit__.assert_awaited_once()
        client.__aexit__.assert_awaited_once()
