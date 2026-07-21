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
