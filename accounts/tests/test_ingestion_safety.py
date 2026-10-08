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
from accounts.models import CustomUserModel, Document, Chapter
from contextlib import contextmanager


class OfflineIngestionTest(TestCase):
    def setUp(self):
        super().setUp()
        @contextmanager
        def lease(document_id):
            yield mock.Mock(token="test-owner")
        for target, kwargs in [
            ("document_lease", {"side_effect": lease}),
            ("verify_vectors", {}),
            ("_queue_version_cleanup", {}),
            ("_sparse_index_for_cleanup", {"return_value": None}),
        ]:
            patcher = mock.patch.object(tasks, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)


def _index():
    index = mock.MagicMock()
    index.list.return_value = iter([])
    return index


class IngestionFailureVisibilityTests(OfflineIngestionTest):
    def setUp(self):
        super().setUp()
        self.user = CustomUserModel.objects.create_user(
            email="ingest@test.com", password="x", name="I"
        )
        self.doc = Document.objects.create(
            user=self.user,
            chapter=Chapter.objects.create(user=self.user, name="Test"),
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
             mock.patch.object(tasks, "_sparse_index_or_none", return_value=None), \
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


class IngestionIsIdempotentTests(OfflineIngestionTest):
    """A rebuild writes a shadow version and never purges live vectors."""

    def setUp(self):
        super().setUp()
        self.user = CustomUserModel.objects.create_user(
            email="idem@test.com", password="x", name="J"
        )
        self.doc = Document.objects.create(
            user=self.user, title="t", file="u/x.txt", file_type="txt",
            chapter=Chapter.objects.create(user=self.user, name="Test"),
            extracted_text="Photosynthesis converts light into chemical energy. " * 50,
        )

    def test_existing_vectors_are_kept_while_reindexing(self):
        import tiktoken

        index = mock.MagicMock()
        index.list.return_value = iter([["old-1", "old-2"]])

        with mock.patch.object(
            tasks, "_get_clients",
            return_value=(index, tiktoken.get_encoding("cl100k_base"), None),
        ), mock.patch.object(tasks, "push_ingestion_status"), \
             mock.patch.object(tasks, "get_channel_layer"), \
             mock.patch.object(tasks, "_sparse_index_or_none", return_value=None), \
             mock.patch.object(tasks, "async_to_sync") as ats:
            ats.side_effect = lambda fn: (
                (lambda batch: [[0.1] * 384 for _ in batch])
                if getattr(fn, "__name__", "") == "embed_texts"
                else mock.MagicMock()
            )
            tasks.process_document_ingestion(str(self.doc.id))

        index.delete.assert_not_called()

    def test_a_purge_failure_does_not_lose_the_document(self):
        """Best-effort: worst case is duplicates, which beats a failed upload."""
        index = mock.MagicMock()
        index.list.side_effect = RuntimeError("pinecone down")

        removed = tasks._delete_document_vectors(index, str(self.doc.id), "test")
        self.assertEqual(removed, 0)


class HybridIngestionTests(OfflineIngestionTest):
    """Deterministic ids and the sparse write path.

    The id is load-bearing now: it is the key RRF fuses on, so the dense and
    sparse halves of a chunk MUST agree on it. A random or batch-derived id
    would leave fusion unable to recognise the two halves as one chunk.
    """

    def setUp(self):
        super().setUp()
        self.user = CustomUserModel.objects.create_user(
            email="hybrid@test.com", password="x", name="H"
        )
        self.doc = Document.objects.create(
            user=self.user, title="t", file="u/x.txt", file_type="txt",
            chapter=Chapter.objects.create(user=self.user, name="Test"),
            extracted_text="Photosynthesis converts light into chemical energy. " * 200,
        )

    def _ingest(self, sparse_index=None, sparse_vectors=None):
        import tiktoken

        dense_index = _index()

        def dispatch(fn):
            name = getattr(fn, "__name__", "")
            if name == "embed_texts":
                return lambda batch: [[0.1] * 384 for _ in batch]
            if name == "embed_sparse":
                return lambda batch, input_type=None: (
                    sparse_vectors(batch) if sparse_vectors
                    else [{"indices": [1], "values": [1.0]} for _ in batch]
                )
            return mock.MagicMock()

        with mock.patch.object(
            tasks, "_get_clients",
            return_value=(dense_index, tiktoken.get_encoding("cl100k_base"), None),
        ), mock.patch.object(tasks, "push_ingestion_status"), \
             mock.patch.object(tasks, "get_channel_layer"), \
             mock.patch.object(tasks, "_sparse_index_or_none", return_value=sparse_index), \
             mock.patch.object(tasks, "async_to_sync") as ats:
            ats.side_effect = dispatch
            tasks.process_document_ingestion(str(self.doc.id))

        return dense_index

    @staticmethod
    def _upserted_ids(index):
        ids = []
        for call in index.upsert.call_args_list:
            for point in call.kwargs.get("vectors", []):
                ids.append(point["id"])
        return ids

    def test_ids_are_deterministic_and_sequential(self):
        ids = self._upserted_ids(self._ingest())
        prefix = f"doc_{self.doc.id}_v1_p0_c"
        self.assertTrue(all(i.startswith(prefix) for i in ids))
        self.assertEqual(
            [int(i.removeprefix(prefix)) for i in ids],
            list(range(len(ids))),
        )

    def test_new_attempts_use_new_versions_with_the_same_chunk_positions(self):
        """A new lease must not share a namespace with an expired worker."""
        first = self._upserted_ids(self._ingest())
        self.doc.refresh_from_db()
        second = self._upserted_ids(self._ingest())
        self.assertEqual([key.replace('_v1_', '_v2_') for key in first], second)

    def test_dense_and_sparse_receive_identical_ids(self):
        """If these ever diverge, RRF sees two unrelated sets of chunks."""
        sparse_index = mock.MagicMock()
        sparse_index.list.return_value = iter([])
        dense_index = self._ingest(sparse_index=sparse_index)

        self.assertEqual(
            self._upserted_ids(dense_index),
            self._upserted_ids(sparse_index),
        )

    def test_chunks_with_no_lexical_content_are_skipped_not_failed(self):
        """An empty sparse vector is rejected by Pinecone with a 400 and would
        take the whole batch down. Those chunks are correctly dense-only."""
        sparse_index = mock.MagicMock()
        sparse_index.list.return_value = iter([])

        def half_empty(batch):
            return [
                {"indices": [], "values": []} if i % 2 else {"indices": [1], "values": [1.0]}
                for i, _ in enumerate(batch)
            ]

        dense_index = self._ingest(sparse_index=sparse_index, sparse_vectors=half_empty)

        dense_ids = self._upserted_ids(dense_index)
        sparse_ids = self._upserted_ids(sparse_index)

        self.assertTrue(sparse_ids, "non-empty vectors should still be upserted")
        self.assertLess(len(sparse_ids), len(dense_ids))
        self.assertTrue(set(sparse_ids).issubset(set(dense_ids)))

        self.doc.refresh_from_db()
        self.assertEqual(self.doc.status, Document.STATUS_COMPLETED)

    def test_sparse_failure_never_fails_the_document(self):
        sparse_index = mock.MagicMock()
        sparse_index.list.return_value = iter([])
        sparse_index.upsert.side_effect = RuntimeError("pinecone 500")

        self._ingest(sparse_index=sparse_index)

        self.doc.refresh_from_db()
        self.assertEqual(self.doc.status, Document.STATUS_COMPLETED)

    def test_purge_clears_both_indexes(self):
        dense = mock.MagicMock()
        dense.list.return_value = iter([["a"]])
        sparse = mock.MagicMock()
        sparse.list.return_value = iter([["a"]])

        with mock.patch.object(tasks, "_sparse_index_for_cleanup", return_value=sparse):
            removed = tasks._purge_document_vectors(dense, str(self.doc.id), "test")

        dense.delete.assert_any_call(ids=["a"])
        sparse.delete.assert_any_call(ids=["a"])
        self.assertEqual(removed, 2)
