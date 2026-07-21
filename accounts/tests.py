from django.test import TestCase
from rest_framework.test import APIClient
from django.contrib.auth import get_user_model

User = get_user_model()


class JWTRotationBlacklistTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.password = "supersecret123"
        self.user = User.objects.create_user(
            email="a@b.com", password=self.password, name="Tester"
        )

    def _login(self):
        res = self.client.post(
            "/auth/jwt/create/",
            {"email": "a@b.com", "password": self.password},
            format="json",
        )
        self.assertEqual(res.status_code, 200, res.content)
        return res.data["access"], res.data["refresh"]

    def test_refresh_rotates_and_blacklists_old(self):
        _, refresh = self._login()
        res = self.client.post("/auth/jwt/refresh/", {"refresh": refresh}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        self.assertIn("access", res.data)
        self.assertIn("refresh", res.data)  # rotation returns a new refresh
        self.assertNotEqual(res.data["refresh"], refresh)
        # old refresh is now blacklisted -> reuse fails
        reuse = self.client.post("/auth/jwt/refresh/", {"refresh": refresh}, format="json")
        self.assertEqual(reuse.status_code, 401, reuse.content)

    def test_blacklist_endpoint_revokes_refresh(self):
        _, refresh = self._login()
        res = self.client.post("/auth/jwt/blacklist/", {"refresh": refresh}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        after = self.client.post("/auth/jwt/refresh/", {"refresh": refresh}, format="json")
        self.assertEqual(after.status_code, 401, after.content)


class IngestionStatusBuilderTests(TestCase):
    def test_minimal_payload(self):
        from accounts.realtime import build_ingestion_status
        p = build_ingestion_status("doc1", "reading")
        self.assertEqual(p, {"type": "ingestion_status",
                             "document_id": "doc1", "phase": "reading"})

    def test_optional_fields_included_only_when_set(self):
        from accounts.realtime import build_ingestion_status
        p = build_ingestion_status("doc2", "embedding", batch=2, total_batches=5)
        self.assertEqual(p["batch"], 2)
        self.assertEqual(p["total_batches"], 5)
        self.assertNotIn("chapter_id", p)
        self.assertNotIn("error", p)

    def test_naming_carries_chapter_and_title(self):
        from accounts.realtime import build_ingestion_status
        p = build_ingestion_status(
            "doc3", "naming", chapter_id="ch9", title="Data Structures")
        self.assertEqual(p["chapter_id"], "ch9")
        self.assertEqual(p["title"], "Data Structures")

    def test_document_id_coerced_to_str(self):
        import uuid
        from accounts.realtime import build_ingestion_status
        u = uuid.uuid4()
        p = build_ingestion_status(u, "ready")
        self.assertEqual(p["document_id"], str(u))


class NotificationConsumerForwardingTests(TestCase):
    def test_forwards_data_payload_verbatim(self):
        import asyncio
        from accounts.consumers import NotificationConsumer

        sent = {}

        async def scenario():
            consumer = NotificationConsumer()

            async def fake_send(text_data=None):
                sent["text"] = text_data

            consumer.send = fake_send
            await consumer.send_notification(
                {"type": "send_notification",
                 "data": {"type": "ingestion_status",
                          "document_id": "abc", "phase": "embedding"}})

        asyncio.new_event_loop().run_until_complete(scenario())
        self.assertIn('"ingestion_status"', sent["text"])
        self.assertIn('"document_id": "abc"', sent["text"])

    def test_legacy_message_still_works(self):
        import asyncio
        from accounts.consumers import NotificationConsumer

        sent = {}

        async def scenario():
            consumer = NotificationConsumer()

            async def fake_send(text_data=None):
                sent["text"] = text_data

            consumer.send = fake_send
            await consumer.send_notification(
                {"type": "send_notification", "message": "notebook_updated"})

        asyncio.new_event_loop().run_until_complete(scenario())
        self.assertIn('"notebook_updated"', sent["text"])


class DocumentRetryTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="r@b.com", password="pw12345", name="R")
        self.client.force_authenticate(self.user)

    def _make_failed_doc(self, chapter=None):
        from accounts.models import Document
        return Document.objects.create(
            user=self.user, chapter=chapter, title="t",
            file="uploads/x.txt", file_type="txt",
            status=Document.STATUS_FAILED, error_message="boom")

    def test_retry_no_chapter_requeues_create(self):
        from unittest.mock import patch
        from accounts.models import Document
        doc = self._make_failed_doc()
        with patch("accounts.views.create_chapter_from_document") as mock_task:
            res = self.client.post(f"/auth/documents/{doc.id}/retry/")
            self.assertEqual(res.status_code, 202, res.content)
            mock_task.delay.assert_called_once_with(str(doc.id))
        doc.refresh_from_db()
        self.assertEqual(doc.status, Document.STATUS_PENDING)
        self.assertIsNone(doc.error_message)

    def test_retry_foreign_document_404(self):
        from accounts.models import Document
        other = User.objects.create_user(
            email="o@b.com", password="pw12345", name="O")
        doc = Document.objects.create(
            user=other, title="t", file="uploads/x.txt",
            file_type="txt", status=Document.STATUS_FAILED)
        res = self.client.post(f"/auth/documents/{doc.id}/retry/")
        self.assertEqual(res.status_code, 404)


class BuildAnswerMessagesTests(TestCase):
    def test_returns_system_then_user_with_context_and_query(self):
        from accounts.rag_pipeline import build_answer_messages
        msgs = build_answer_messages("CTX-TEXT", "What is X?")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("CTX-TEXT", msgs[1]["content"])
        self.assertIn("What is X?", msgs[1]["content"])
        self.assertIn("tutor", msgs[0]["content"].lower())
        self.assertIn("Beyond your notes", msgs[0]["content"])


class ParseFollowupsTests(TestCase):
    def test_valid_json(self):
        from accounts.rag_pipeline import parse_followups
        raw = '{"followups": ["What is chaining?", "When resize?", "Load factor?"]}'
        self.assertEqual(
            parse_followups(raw),
            ["What is chaining?", "When resize?", "Load factor?"],
        )

    def test_caps_at_three_and_strips(self):
        from accounts.rag_pipeline import parse_followups
        raw = '{"followups": ["  a ", "b", "c", "d"]}'
        self.assertEqual(parse_followups(raw), ["a", "b", "c"])

    def test_drops_empty_entries(self):
        from accounts.rag_pipeline import parse_followups
        raw = '{"followups": ["", "  ", "real one"]}'
        self.assertEqual(parse_followups(raw), ["real one"])

    def test_garbage_returns_empty(self):
        from accounts.rag_pipeline import parse_followups
        self.assertEqual(parse_followups("not json"), [])
        self.assertEqual(parse_followups('{"nope": 1}'), [])


class BuildSourcesTests(TestCase):
    def test_dedup_and_snippet(self):
        from types import SimpleNamespace
        from accounts.rag_pipeline import build_sources
        results = [
            SimpleNamespace(payload={"document_id": "d1", "text": "A" * 200}),
            SimpleNamespace(payload={"document_id": "d1", "text": "dup"}),
            SimpleNamespace(payload={"document_id": "d2", "text": "second"}),
        ]
        srcs = build_sources(results)
        self.assertEqual(len(srcs), 2)
        self.assertEqual(srcs[0]["document_id"], "d1")
        self.assertEqual(len(srcs[0]["snippet"]), 140)
        self.assertEqual(srcs[1]["document_id"], "d2")

    def test_caps_at_three(self):
        from types import SimpleNamespace
        from accounts.rag_pipeline import build_sources
        results = [
            SimpleNamespace(payload={"document_id": f"d{i}", "text": "t"})
            for i in range(6)
        ]
        self.assertEqual(len(build_sources(results)), 3)


class CascadeDeleteTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="del@b.com", password="pw12345", name="D")
        self.client.force_authenticate(self.user)

    def _doc(self, chapter, name="uploads/x.txt"):
        from accounts.models import Document
        return Document.objects.create(
            user=self.user, chapter=chapter, title="t",
            file=name, file_type="txt",
            status=Document.STATUS_COMPLETED)

    def test_delete_chapter_purges_documents_and_enqueues_cleanup(self):
        from unittest.mock import patch
        from accounts.models import Chapter, Document, GenerateQuestion
        chapter = Chapter.objects.create(user=self.user, name="Ch")
        doc = self._doc(chapter, "uploads/a.txt")
        GenerateQuestion.objects.create(
            chapter=chapter, question_text="q", answer_text="a")

        with patch("accounts.views.cleanup_document_data") as mock_task:
            res = self.client.delete(f"/auth/chapters/{chapter.id}/")
            self.assertEqual(res.status_code, 204, res.content)
            mock_task.delay.assert_called_once_with(
                [str(doc.id)], ["uploads/a.txt"])

        self.assertFalse(Chapter.objects.filter(id=chapter.id).exists())
        self.assertFalse(Document.objects.filter(id=doc.id).exists())
        self.assertEqual(GenerateQuestion.objects.count(), 0)

    def test_delete_subject_purges_all_chapter_documents(self):
        from unittest.mock import patch
        from accounts.models import Subject, Chapter, Document
        subject = Subject.objects.create(user=self.user, name="Subj")
        ch1 = Chapter.objects.create(user=self.user, subject=subject, name="C1")
        ch2 = Chapter.objects.create(user=self.user, subject=subject, name="C2")
        d1 = self._doc(ch1, "uploads/1.txt")
        d2 = self._doc(ch2, "uploads/2.txt")

        with patch("accounts.views.cleanup_document_data") as mock_task:
            res = self.client.delete(f"/auth/subjects/{subject.id}/")
            self.assertEqual(res.status_code, 204, res.content)
            self.assertTrue(mock_task.delay.called)
            doc_ids, files = mock_task.delay.call_args[0]
            self.assertCountEqual(doc_ids, [str(d1.id), str(d2.id)])
            self.assertCountEqual(files, ["uploads/1.txt", "uploads/2.txt"])

        self.assertFalse(Subject.objects.filter(id=subject.id).exists())
        self.assertEqual(Chapter.objects.filter(id__in=[ch1.id, ch2.id]).count(), 0)
        self.assertEqual(Document.objects.count(), 0)

    def test_delete_chapter_keeps_chat_sessions(self):
        from unittest.mock import patch
        from accounts.models import Chapter, ChatSession
        chapter = Chapter.objects.create(user=self.user, name="Ch")
        session = ChatSession.objects.create(user=self.user, chapter=chapter)

        with patch("accounts.views.cleanup_document_data"):
            res = self.client.delete(f"/auth/chapters/{chapter.id}/")
            self.assertEqual(res.status_code, 204, res.content)

        session.refresh_from_db()
        self.assertIsNone(session.chapter_id)

    def test_cannot_delete_foreign_chapter(self):
        from accounts.models import Chapter
        other = User.objects.create_user(
            email="x@b.com", password="pw12345", name="X")
        chapter = Chapter.objects.create(user=other, name="Ch")
        with patch_cleanup():
            res = self.client.delete(f"/auth/chapters/{chapter.id}/")
        self.assertEqual(res.status_code, 404)
        self.assertTrue(Chapter.objects.filter(id=chapter.id).exists())


class CleanupTaskTests(TestCase):
    def test_deletes_vectors_by_prefix_and_files(self):
        from unittest.mock import patch, MagicMock
        from accounts.tasks import cleanup_document_data

        index = MagicMock()
        index.list.return_value = iter([["d1#a", "d1#b"]])

        with patch("accounts.tasks.get_pinecone_index", return_value=index), \
             patch("accounts.tasks.default_storage") as storage:
            storage.exists.return_value = True
            cleanup_document_data(["d1"], ["uploads/a.txt"])

        index.list.assert_called_once_with(prefix="d1#")
        index.delete.assert_called_once_with(ids=["d1#a", "d1#b"])
        storage.delete.assert_called_once_with("uploads/a.txt")

    def test_survives_pinecone_errors(self):
        from unittest.mock import patch, MagicMock
        from accounts.tasks import cleanup_document_data

        index = MagicMock()
        index.list.side_effect = RuntimeError("boom")
        with patch("accounts.tasks.get_pinecone_index", return_value=index), \
             patch("accounts.tasks.default_storage") as storage:
            storage.exists.return_value = False
            # Should not raise
            cleanup_document_data(["d1"], [])


def patch_cleanup():
    from unittest.mock import patch
    return patch("accounts.views.cleanup_document_data")


class ChapterQuestionListTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="q@b.com", password="pw12345", name="Q")
        self.client.force_authenticate(self.user)

    def test_lists_saved_questions_for_chapter(self):
        from accounts.models import Chapter, GenerateQuestion
        chapter = Chapter.objects.create(user=self.user, name="Ch")
        GenerateQuestion.objects.create(
            chapter=chapter, question_text="Q1?", answer_text="A1")
        GenerateQuestion.objects.create(
            chapter=chapter, question_text="Q2?", answer_text="A2")

        res = self.client.get(f"/auth/chapters/{chapter.id}/questions/")
        self.assertEqual(res.status_code, 200, res.content)
        self.assertEqual(len(res.data), 2)
        self.assertEqual(
            {q["question_text"] for q in res.data}, {"Q1?", "Q2?"})

    def test_cannot_list_foreign_chapter_questions(self):
        from accounts.models import Chapter, GenerateQuestion
        other = User.objects.create_user(
            email="o2@b.com", password="pw12345", name="O")
        chapter = Chapter.objects.create(user=other, name="Ch")
        GenerateQuestion.objects.create(
            chapter=chapter, question_text="secret?", answer_text="a")

        res = self.client.get(f"/auth/chapters/{chapter.id}/questions/")
        self.assertEqual(res.status_code, 200, res.content)
        self.assertEqual(len(res.data), 0)


class RagChatResponseShapeTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="c@b.com", password="pw12345", name="C")
        self.client.force_authenticate(self.user)

    def test_response_includes_sources_and_followups(self):
        from unittest.mock import patch
        from accounts.models import Chapter, Document
        chapter = Chapter.objects.create(user=self.user, name="Ch")
        doc = Document.objects.create(
            user=self.user, chapter=chapter, title="My Doc",
            file="uploads/x.txt", file_type="txt",
            status=Document.STATUS_COMPLETED)
        fake = {
            "answer": "The answer.",
            "sources": [{"document_id": str(doc.id), "snippet": "snip"}],
            "followups": ["Next q?"],
        }
        with patch("accounts.views.rag_pipeline.run", return_value=fake):
            res = self.client.post(
                "/auth/rag-chat/",
                {"chapter": str(chapter.id), "text": "hi"},
                format="json")
        self.assertEqual(res.status_code, 201, res.content)
        self.assertEqual(res.data["text"], "The answer.")
        self.assertEqual(res.data["followups"], ["Next q?"])
        self.assertEqual(res.data["sources"][0]["title"], "My Doc")
