import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

from asgiref.sync import async_to_sync
from django.test import TestCase

from accounts.models import Chapter, CustomUserModel, Document, DocumentPage
from accounts.rag_pipeline import RagPipeline, PipelineOutcome
from accounts.summary import MAP_PROMPT, REDUCE_PROMPT, MAX_BATCH_CHARS
from utils.deadline import RequestDeadline, deadline_scope


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class SummarySourceTests(TestCase):
    def test_only_owned_active_pages_and_legacy_fallback_are_read(self):
        user = CustomUserModel.objects.create_user(email="summary@test.com", password="x", name="S")
        chapter = Chapter.objects.create(user=user, name="Chapter")
        doc = Document.objects.create(user=user, chapter=chapter, title="Book", active_version=2,
                                      extracted_text="stale extracted text")
        for version, content in [(1, "retired"), (2, "Canonical [?formula]"), (3, "pending")]:
            DocumentPage.objects.create(document=doc, version=version, page_number=1, reconstructed_md=content)
        legacy = Document.objects.create(user=user, chapter=chapter, title="Legacy", status=Document.STATUS_COMPLETED,
                                         extracted_text="Unpaginated material")
        Document.objects.create(user=user, chapter=chapter, title="Unready", extracted_text="Not ready")
        other_chapter = Chapter.objects.create(user=user, name="Other")
        Document.objects.create(user=user, chapter=other_chapter, title="Wrong chapter", active_version=1,
                                extracted_text="Other chapter material")
        pages = async_to_sync(RagPipeline._chapter_pages)(chapter.id, user.id)
        self.assertEqual(pages, ["Book, page 1:\nCanonical formula", "Legacy (unpaginated):\nUnpaginated material"])
        self.assertEqual(async_to_sync(RagPipeline._chapter_pages)(chapter.id, "00000000-0000-0000-0000-000000000000"), [])
        DocumentPage.objects.create(document=legacy, version=0, page_number=1, reconstructed_md="")
        self.assertEqual(len(async_to_sync(RagPipeline._chapter_pages)(chapter.id, user.id)), 1)


class SummaryGenerationTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.pipeline = object.__new__(RagPipeline)
        self.pipeline.llm_client = mock.Mock()

    async def test_map_groups_five_pages_and_reduce_receives_every_summary(self):
        pages = [f"page-{i}" for i in range(12)]
        mapped = []
        async def llm(*args, messages, **kwargs):
            if messages[0]["content"] == MAP_PROMPT:
                mapped.append(messages[1]["content"])
                return completion("summary: " + messages[1]["content"])
            self.assertEqual(messages[0]["content"], REDUCE_PROMPT)
            for page in pages:
                self.assertIn(page, messages[1]["content"])
            return completion("Revision guide")
        with mock.patch.object(self.pipeline, "_chapter_pages", return_value=pages), \
             mock.patch("accounts.rag_pipeline.ask_llm", side_effect=llm):
            result = await self.pipeline.handle_summary("chapter", "user")
        self.assertEqual(result.outcome, PipelineOutcome.SUMMARY_GENERATED)
        self.assertEqual([len(batch.split("\n\n")) for batch in mapped], [5, 5, 2])

    async def test_large_chapter_is_not_truncated_and_reduction_is_hierarchical(self):
        pages = [f"page{i}: " + "x" * 14000 for i in range(20)]
        map_inputs, reduce_inputs = [], []
        running = peak = 0
        async def llm(*args, messages, **kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0)
            running -= 1
            material = messages[1]["content"]
            self.assertLessEqual(len(material), MAX_BATCH_CHARS)
            (map_inputs if messages[0]["content"] == MAP_PROMPT else reduce_inputs).append(material)
            return completion("a bounded summary")
        with mock.patch.object(self.pipeline, "_chapter_pages", return_value=pages), \
             mock.patch("accounts.rag_pipeline.ask_llm", side_effect=llm):
            result = await self.pipeline.handle_summary("chapter", "user")
        self.assertEqual(result.outcome, PipelineOutcome.SUMMARY_GENERATED)
        self.assertEqual("".join(map_inputs).replace("\n\n", ""), "".join(pages))
        self.assertGreater(len(reduce_inputs), 1)
        self.assertLessEqual(peak, 3)

    async def test_deadline_cancels_all_map_work(self):
        active = 0
        async def slow(*args, **kwargs):
            nonlocal active
            active += 1
            try:
                await asyncio.sleep(30)
            finally:
                active -= 1
        with mock.patch.object(self.pipeline, "_chapter_pages", return_value=["page"] * 20), \
             mock.patch.object(self.pipeline, "contextualize_and_route", return_value=("Summary", "summary")), \
             mock.patch("accounts.rag_pipeline.ask_llm", side_effect=slow):
            with deadline_scope(RequestDeadline(.03)):
                result = await self.pipeline.run("Summarize chapter", [], "chapter", "user")
        self.assertEqual(result.outcome, PipelineOutcome.DEADLINE_EXCEEDED)
        self.assertEqual(active, 0)

    async def test_missing_material_and_failed_map_do_not_claim_success(self):
        with mock.patch.object(self.pipeline, "_chapter_pages", return_value=[]), \
             mock.patch("accounts.rag_pipeline.ask_llm") as llm:
            result = await self.pipeline.handle_summary("chapter", "user")
            self.assertEqual(result.outcome, PipelineOutcome.INSUFFICIENT_EVIDENCE)
            llm.assert_not_called()
        with mock.patch.object(self.pipeline, "_chapter_pages", return_value=["page"]), \
             mock.patch("accounts.rag_pipeline.ask_llm", return_value=completion("")):
            result = await self.pipeline.handle_summary("chapter", "user")
            self.assertEqual(result.outcome, PipelineOutcome.DEPENDENCY_UNAVAILABLE)
