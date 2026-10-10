"""Dependency failures must not become persisted, successful chat answers."""

from types import SimpleNamespace
from unittest import mock

import httpx
import openai
from tenacity import Future, RetryError
from rest_framework.test import APITestCase

from accounts.models import Chapter, ChatMessage, CustomUserModel, Document
from accounts.rag_pipeline import PipelineOutcome, PipelineResult, RagPipeline
from accounts.serializers import ChatMessageSerializer
from utils.llm_gateway import LLMUnavailable


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class ChatPipelineOutcomeTests(APITestCase):
    def setUp(self):
        self.real_followups = RagPipeline._generate_followups
        self.user = CustomUserModel.objects.create_user(
            email="outcomes@example.com", password="x", name="Student"
        )
        self.client.force_authenticate(self.user)
        self.chapter = Chapter.objects.create(user=self.user, name="Chapter")
        self.document = Document.objects.create(
            chapter=self.chapter, user=self.user, title="Notes", file="notes.txt",
            status=Document.STATUS_COMPLETED,
        )
        self.hits = [SimpleNamespace(
            id="chunk", score=1.0,
            payload={"text": "Useful evidence. " * 20, "document_id": str(self.document.id)},
        )]
        patches = {
            "accounts.rag_pipeline.RagPipeline.contextualize_and_route": mock.AsyncMock(
                return_value=("Explain gravity", "question")
            ),
            "accounts.rag_pipeline.ask_llm": mock.AsyncMock(return_value=completion("An answer")),
            "accounts.rag_pipeline.embed_texts": mock.AsyncMock(return_value=[[0.1]]),
            "accounts.rag_pipeline.hybrid_search": mock.AsyncMock(return_value=self.hits),
            "accounts.rag_pipeline.RagPipeline._generate_followups": mock.AsyncMock(return_value=[]),
            "accounts.rag_pipeline.retrieval_evaluator.evaluate": mock.Mock(),
        }
        self.dependencies = {}
        for target, replacement in patches.items():
            patcher = mock.patch(target, replacement)
            self.dependencies[target.rsplit(".", 1)[-1]] = patcher.start()
            self.addCleanup(patcher.stop)

    def post_chat(self, **options):
        return self.client.post("/auth/rag-chat/", {
            "chapter": str(self.chapter.id), "text": "Explain gravity",
            **options,
        }, format="json")

    def test_empty_chapter_does_not_search_other_notes_by_default(self):
        self.dependencies["hybrid_search"].side_effect = [[], self.hits]
        response = self.post_chat()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["is_unanswered"])
        self.assertEqual(response.data["sources"], [])
        self.dependencies["hybrid_search"].assert_awaited_once()
        self.dependencies["_generate_followups"].assert_not_awaited()
        # Only expansion may call the LLM; answer synthesis must not run.
        self.assertEqual(self.dependencies["ask_llm"].await_count, 1)

    def test_explicit_false_does_not_widen_scope(self):
        self.dependencies["hybrid_search"].side_effect = [[], self.hits]
        self.assertEqual(self.post_chat(allow_library_fallback=False).status_code, 200)
        self.dependencies["hybrid_search"].assert_awaited_once()

    def test_opt_in_keeps_successful_chapter_search_in_scope(self):
        response = self.post_chat(allow_library_fallback=True)
        self.assertEqual(response.data["text"], "An answer")
        self.assertFalse(response.data["sources"][0]["is_fallback_scope"])
        self.dependencies["hybrid_search"].assert_awaited_once()

    def test_empty_other_notes_remain_unanswered(self):
        chapter = Chapter.objects.create(user=self.user, name="Other")
        Document.objects.create(user=self.user, chapter=chapter, title="Notes", active_version=1)
        self.dependencies["hybrid_search"].return_value = []
        response = self.post_chat(allow_library_fallback=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["sources"], [])
        self.assertTrue(response.data["is_unanswered"])
        self.assertEqual(self.dependencies["hybrid_search"].await_count, 2)

    def test_provider_failure_never_triggers_fallback(self):
        self.dependencies["hybrid_search"].side_effect = TimeoutError("down")
        self.assertEqual(self.post_chat(allow_library_fallback=True).status_code, 504)
        self.dependencies["hybrid_search"].assert_awaited_once()

    def test_legacy_page_guess_is_only_a_document_citation(self):
        self.hits[0].payload["page_number"] = 2
        source = self.post_chat().data["sources"][0]
        self.assertIsNone(source["page_number"])
        self.assertIsNone(source["version"])

    def test_empty_chunk_text_counts_as_no_evidence(self):
        for text in ("   ", None):
            self.hits[0].payload["text"] = text
            self.dependencies["hybrid_search"].reset_mock()
            self.assertTrue(self.post_chat().data["is_unanswered"])
            self.dependencies["hybrid_search"].assert_awaited_once()

    def test_fallback_requires_an_actual_boolean(self):
        for value in ("true", 1, [], None):
            self.assertEqual(self.post_chat(allow_library_fallback=value).status_code, 400)
        self.dependencies["hybrid_search"].assert_not_awaited()

    def test_page_citations_survive_response_and_history(self):
        self.hits[0].payload.update(page_number=2, version=3)
        second = SimpleNamespace(id="page3", score=0.9, payload={
            **self.hits[0].payload, "page_number": 3, "text": "Another page. " * 20,
        })
        self.hits.append(second)
        response = self.post_chat()
        self.assertEqual([s["page_number"] for s in response.data["sources"]], [2, 3])
        self.assertTrue(all(s["version"] == 3 for s in response.data["sources"]))
        saved = ChatMessageSerializer(ChatMessage.objects.get(sender="ai")).data
        self.assertEqual(saved["citations"], response.data["sources"])

    def test_dependency_failures_are_not_saved_as_answers(self):
        request = httpx.Request("POST", "https://dependency.invalid")
        failures = [
            (LLMUnavailable("circuit open"), 503),
            (httpx.ReadTimeout("provider timeout"), 504),
            (httpx.HTTPStatusError("quota", request=request,
                                  response=httpx.Response(429, request=request)), 429),
        ]
        for stage in ("embed_texts", "hybrid_search", "ask_llm"):
            for error, expected in failures:
                with self.subTest(stage=stage, error=type(error).__name__):
                    dependency = self.dependencies[stage]
                    dependency.side_effect = error
                    try:
                        response = self.post_chat()
                        self.assertEqual(response.status_code, expected)
                        self.assertTrue(response.data["retryable"])
                        self.assertIn("error", response.data)
                        self.assertFalse(ChatMessage.objects.exists())
                    finally:
                        dependency.side_effect = None

    def test_success_persists_answer_and_source_metadata(self):
        response = self.post_chat()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["text"], "An answer")
        self.assertEqual(response.data["sources"][0]["title"], "Notes")
        self.assertEqual(ChatMessage.objects.count(), 2)
        self.assertFalse(ChatMessage.objects.get(sender="ai").is_unanswered)

    def test_total_deadline_returns_retryable_504_without_saving_messages(self):
        import asyncio
        async def blocked(*args, **kwargs):
            await asyncio.sleep(30)
        self.dependencies["embed_texts"].side_effect = blocked
        with mock.patch("utils.deadline.CHAT_BUDGET", .03):
            response = self.post_chat()
        self.assertEqual(response.status_code, 504)
        self.assertTrue(response.data["retryable"])
        self.assertFalse(ChatMessage.objects.exists())

    def test_no_evidence_is_saved_as_unanswered(self):
        self.dependencies["hybrid_search"].return_value = []
        response = self.post_chat()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["is_unanswered"])
        self.assertTrue(ChatMessage.objects.get(sender="ai").is_unanswered)
        self.assertTrue(ChatMessageSerializer(ChatMessage.objects.get(sender="ai")).data["is_unanswered"])

    def test_short_context_is_insufficient_evidence(self):
        self.hits[0].payload["text"] = "Too little evidence"
        self.assertEqual(self.post_chat().status_code, 200)
        self.assertTrue(ChatMessage.objects.get(sender="ai").is_unanswered)

    def test_summary_timeout_does_not_persist(self):
        self.dependencies["contextualize_and_route"].return_value = ("Summarize", "summary")
        self.dependencies["ask_llm"].side_effect = TimeoutError("timeout")
        with mock.patch("accounts.rag_pipeline.RagPipeline._chapter_pages",
                        new=mock.AsyncMock(return_value=["Chapter text"])):
            response = self.post_chat()
        self.assertEqual(response.status_code, 504)
        self.assertFalse(ChatMessage.objects.exists())

    def test_empty_summary_is_unanswered(self):
        self.dependencies["contextualize_and_route"].return_value = ("Summarize", "summary")
        with mock.patch("accounts.rag_pipeline.RagPipeline._chapter_pages",
                        new=mock.AsyncMock(return_value=[])):
            response = self.post_chat()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(ChatMessage.objects.get(sender="ai").is_unanswered)

    def test_summary_success_is_persisted(self):
        self.dependencies["contextualize_and_route"].return_value = ("Summarize", "summary")
        with mock.patch("accounts.rag_pipeline.RagPipeline._chapter_pages",
                        new=mock.AsyncMock(return_value=["Chapter text"])):
            response = self.post_chat()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(ChatMessage.objects.get(sender="ai").text, "An answer")

    def test_wrapped_provider_failures_preserve_timeout_and_rate_limit(self):
        request = httpx.Request("POST", "https://dependency.invalid")
        for error, expected in [
            (openai.APITimeoutError(request=request), 504),
            (openai.RateLimitError("private-provider-data",
                                  response=httpx.Response(429, request=request), body=None), 429),
        ]:
            with self.subTest(error=type(error).__name__):
                future = Future(1)
                future.set_exception(error)
                self.dependencies["embed_texts"].side_effect = RetryError(future)
                response = self.post_chat()
                self.assertEqual(response.status_code, expected)
                self.assertNotIn("private-provider-data", str(response.data))
                self.assertFalse(ChatMessage.objects.exists())

    def test_retry_saves_one_turn_and_only_completed_turns_reach_history(self):
        self.dependencies["embed_texts"].side_effect = LLMUnavailable("down")
        self.assertEqual(self.post_chat().status_code, 503)
        self.dependencies["embed_texts"].side_effect = None
        self.assertEqual(self.post_chat().status_code, 201)
        self.assertEqual(ChatMessage.objects.count(), 2)
        self.assertEqual(self.dependencies["contextualize_and_route"].call_args.args[1], [])
        self.assertEqual(self.post_chat().status_code, 201)
        history = self.dependencies["contextualize_and_route"].call_args.args[1]
        self.assertEqual([(m.sender, m.text) for m in history], [
            ("user", "Explain gravity"), ("ai", "An answer"),
        ])

    def test_failure_logs_actual_outcome(self):
        self.dependencies["embed_texts"].side_effect = TimeoutError("timeout")
        with self.assertLogs("accounts", level="INFO") as logs:
            self.assertEqual(self.post_chat().status_code, 504)
        completed = next(r for r in logs.records if r.msg == "rag_request_completed")
        self.assertEqual(completed.status, "deadline_exceeded")
        failed = next(r for r in logs.records if r.msg == "rag_chat_dependency_failed")
        self.assertEqual(failed.outcome, "deadline_exceeded")
        self.assertEqual(failed.http_status, 504)

    def test_success_returns_a_typed_result(self):
        from asgiref.sync import async_to_sync
        from accounts.views import rag_pipeline

        result = async_to_sync(rag_pipeline.run)("Hello", [], self.chapter.id, self.user.id)
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.outcome, PipelineOutcome.SUCCESS)

    def test_optional_followup_failure_keeps_real_answer(self):
        # Restore the real helper, while keeping its provider offline.
        self.dependencies["_generate_followups"].side_effect = None
        from accounts.rag_pipeline import RagPipeline

        with mock.patch("accounts.rag_pipeline.ask_llm", new=mock.AsyncMock(side_effect=[
            completion('{"queries": []}'), completion("Real answer"), LLMUnavailable("down"),
        ])):
            # setUp patches the helper, so stop that patch just for this request.
            with mock.patch.object(RagPipeline, "_generate_followups", self.real_followups):
                response = self.post_chat()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["text"], "Real answer")
        self.assertEqual(response.data["followups"], [])

    def test_primary_and_fallback_searches_use_active_version_filters(self):
        self.document.active_version = 4
        self.document.pending_version = 5
        self.document.save(update_fields=["active_version", "pending_version"])
        other_chapter = Chapter.objects.create(user=self.user, name="Other notes")
        other = Document.objects.create(user=self.user, chapter=other_chapter,
                                        title="Physics", status=Document.STATUS_COMPLETED,
                                        active_version=2)
        foreign_user = CustomUserModel.objects.create_user(email="other@test.com", password="x")
        Document.objects.create(user=foreign_user, title="Private", active_version=1)
        self.hits[0].payload.update(document_id=str(other.id), chapter_id=str(other_chapter.id),
                                    page_number=1, version=2)
        self.dependencies["hybrid_search"].side_effect = [[], self.hits]
        response = self.post_chat(allow_library_fallback=True)
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.data["text"].startswith(
            "I couldn't find this in the current chapter, but here is what I found in your other notes..."))
        self.assertTrue(response.data["sources"][0]["is_fallback_scope"])
        self.assertEqual(response.data["sources"][0]["chapter_id"], str(other_chapter.id))
        calls = self.dependencies["hybrid_search"].call_args_list
        self.assertEqual(len(calls), 2)
        for call, document, version in zip(calls, [self.document, other], [4, 2]):
            scope = call.kwargs["filter"]["$and"]
            self.assertEqual(scope[0], {"user_id": {"$eq": str(self.user.id)}})
            self.assertEqual(scope[1]["$or"], [{"$and": [
                {"document_id": {"$eq": str(document.id)}}, {"version": {"$eq": version}},
            ]}])
