"""Ingestion must not report a document ready when its vectors are incomplete.

A FAILED document is visible: the user sees the error and can retry. A document
marked COMPLETED with a batch missing is not — it opens for chat and answers
questions from a partial index, confidently, with citations, and nothing
anywhere says so. That failure mode became reachable the moment embedding gained
a hard input limit, so it is closed here.

All external services are mocked; nothing in this module touches Pinecone, TEI,
Cloudflare, or the network.
"""

from unittest import mock

from django.test import TestCase

from accounts import tasks
from accounts.models import CustomUserModel, Document


def _index():
    index = mock.MagicMock()
    index.list.return_value = iter([])
    return index


class IngestionFailureVisibilityTests(TestCase):
    def setUp(self):
        self.user = CustomUserModel.objects.create_user(
            email="ingest@test.com", password="x", name="I"
        )
        self.doc = Document.objects.create(
            user=self.user,
            title="t",
            file="u/x.txt",
            file_type="txt",
            extracted_text="The mitochondria is the powerhouse of the cell. " * 300,
        )

    def _run(self, embed_side_effect):
        import tiktoken

        with mock.patch.object(
            tasks, "_get_clients",
            return_value=(_index(), tiktoken.get_encoding("cl100k_base"), None),
        ), mock.patch.object(tasks, "push_ingestion_status"), \
             mock.patch.object(tasks, "get_channel_layer"), \
             mock.patch.object(tasks, "async_to_sync") as ats:

            def dispatch(fn):
                if getattr(fn, "__name__", "") == "embed_texts":
                    return embed_side_effect
                return mock.MagicMock()

            ats.side_effect = dispatch
            try:
                tasks.process_document_ingestion(str(self.doc.id))
            except Exception as exc:
                return exc
        return None

    def test_a_failed_batch_does_not_mark_the_document_completed(self):
        def always_fails(_batch):
            raise RuntimeError("provider exploded")

        self._run(always_fails)
        self.doc.refresh_from_db()

        self.assertNotEqual(
            self.doc.status, Document.STATUS_COMPLETED,
            "a document missing vectors was reported as ready for chat",
        )
        self.assertEqual(self.doc.status, Document.STATUS_FAILED)

    def test_the_failure_is_recorded_where_the_user_can_see_it(self):
        def always_fails(_batch):
            raise RuntimeError("provider exploded")

        self._run(always_fails)
        self.doc.refresh_from_db()

        self.assertTrue(self.doc.error_message)
        self.assertIn("batch", self.doc.error_message.lower())

    def test_a_clean_run_still_completes(self):
        """The guard must not break the happy path."""
        def succeeds(batch):
            return [[0.1] * 384 for _ in batch]

        self._run(succeeds)
        self.doc.refresh_from_db()

        self.assertEqual(self.doc.status, Document.STATUS_COMPLETED)
        self.assertIsNone(self.doc.error_message)


class IngestionIsIdempotentTests(TestCase):
    """Retries re-embed the whole document; ids carry a fresh uuid4 each pass.

    Without a purge, every retry would layer another copy of the document into
    the index, and retrieval would start returning the same passage several
    times over.
    """

    def setUp(self):
        self.user = CustomUserModel.objects.create_user(
            email="idem@test.com", password="x", name="J"
        )
        self.doc = Document.objects.create(
            user=self.user, title="t", file="u/x.txt", file_type="txt",
            extracted_text="Photosynthesis converts light into chemical energy. " * 50,
        )

    def test_existing_vectors_are_cleared_before_reindexing(self):
        import tiktoken

        index = mock.MagicMock()
        index.list.return_value = iter([["old-1", "old-2"]])

        with mock.patch.object(
            tasks, "_get_clients",
            return_value=(index, tiktoken.get_encoding("cl100k_base"), None),
        ), mock.patch.object(tasks, "push_ingestion_status"), \
             mock.patch.object(tasks, "get_channel_layer"), \
             mock.patch.object(tasks, "async_to_sync") as ats:
            ats.side_effect = lambda fn: (
                (lambda batch: [[0.1] * 384 for _ in batch])
                if getattr(fn, "__name__", "") == "embed_texts"
                else mock.MagicMock()
            )
            tasks.process_document_ingestion(str(self.doc.id))

        index.delete.assert_any_call(ids=["old-1", "old-2"])

    def test_a_purge_failure_does_not_lose_the_document(self):
        """Best-effort: worst case is duplicates, which beats a failed upload."""
        index = mock.MagicMock()
        index.list.side_effect = RuntimeError("pinecone down")

        removed = tasks._delete_document_vectors(index, str(self.doc.id), "test")
        self.assertEqual(removed, 0)
