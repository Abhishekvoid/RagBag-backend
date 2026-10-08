"""Publication and concurrency invariants with offline providers and real DB rows."""
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

from django.test import TestCase, SimpleTestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from accounts import tasks, ingestion_versions as versions
from accounts.models import Chapter, CustomUserModel, Document, DocumentPage, DocumentIndexVersion
from accounts.page_pipeline import DocumentOversizedError


class VersionedIngestionTests(TestCase):
    def setUp(self):
        self.user = CustomUserModel.objects.create_user(email="versions@test.com", password="x", name="V")
        self.chapter = Chapter.objects.create(user=self.user, name="Chapter")
        self.doc = Document.objects.create(
            user=self.user, chapter=self.chapter, title="Doc", file="doc.txt", file_type="txt",
            extracted_text="The complete old material. " * 25, status=Document.STATUS_COMPLETED,
        )
        self.lease = mock.Mock(token="owner-one")
        @contextmanager
        def lease(_):
            yield self.lease
        self.records = {}
        self.index = mock.Mock()
        self.index.upsert.side_effect = lambda *, vectors: self.records.update({p["id"]: p for p in vectors})
        self.index.fetch.side_effect = lambda *, ids: {"vectors": {i: self.records[i] for i in ids if i in self.records}}
        self.index.query.side_effect = lambda **kw: {"matches": [{"id": i} for i in self.records]}
        self.index.list.side_effect = lambda *, prefix: [[i for i in self.records if i.startswith(prefix)]]
        tokenizer = SimpleNamespace(encode=list, decode="".join)
        self.embed = mock.Mock(side_effect=lambda batch: [[0.1] * 384 for _ in batch])
        def dispatch(fn):
            return self.embed if getattr(fn, "__name__", "") == "embed_texts" else mock.Mock()
        patches = [
            ("document_lease", {"side_effect": lease}),
            ("_get_clients", {"return_value": (self.index, tokenizer, mock.Mock())}),
            ("_sparse_index_or_none", {"return_value": None}),
            ("_sparse_index_for_cleanup", {"return_value": None}),
            ("async_to_sync", {"side_effect": dispatch}),
            ("push_ingestion_status", {}), ("get_channel_layer", {}),
            ("_queue_version_cleanup", {}),
        ]
        for name, kwargs in patches:
            patcher = mock.patch.object(tasks, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def ingest(self, **kwargs):
        return tasks.process_document_ingestion.run(str(self.doc.id), **kwargs)

    def test_live_version_is_untouched_until_all_records_are_searchable(self):
        def query(**kwargs):
            self.doc.refresh_from_db()
            self.assertEqual(self.doc.active_version, 0)
            self.assertEqual(self.doc.status, Document.STATUS_COMPLETED)
            self.assertIsNotNone(self.doc.pending_version)
            return {"matches": [{"id": i} for i in self.records]}
        self.index.query.side_effect = query
        self.assertEqual(self.ingest()["status"], "completed")
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.active_version, 1)
        self.assertIsNone(self.doc.pending_version)
        self.index.delete.assert_not_called()

    def test_failed_rebuild_keeps_old_text_pages_and_readiness(self):
        DocumentPage.objects.create(document=self.doc, page_number=1, reconstructed_md="Old page")
        old_text = self.doc.extracted_text
        def extract(doc, *, version, **kwargs):
            DocumentPage.objects.create(document=doc, version=version, page_number=1,
                                        reconstructed_md="New page text " * 10)
            return "Replacement text " * 20
        self.embed.side_effect = RuntimeError("provider down")
        with mock.patch.object(tasks, "extract_document_text", side_effect=extract):
            with self.assertRaises(RuntimeError):
                self.ingest(rescan=True)
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.active_version, 0)
        self.assertEqual(self.doc.extracted_text, old_text)
        self.assertEqual(self.doc.status, Document.STATUS_COMPLETED)
        client = APIClient()
        client.force_authenticate(self.user)
        response = client.get(f"/auth/documents/{self.doc.id}/pages/")
        self.assertEqual(response.data[0]["reconstructed_md"], "Old page")
        self.assertEqual(len(response.data), 1)
        self.index.delete.assert_not_called()

    def test_incomplete_embedding_batch_cannot_publish(self):
        self.embed.side_effect = lambda batch: [[0.1] * 384]
        with self.assertRaises(RuntimeError):
            self.ingest()
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.active_version, 0)
        self.index.upsert.assert_not_called()

    def test_successful_rescan_publishes_text_and_pages_together(self):
        DocumentPage.objects.create(document=self.doc, page_number=1, reconstructed_md="Old page")
        def extract(doc, *, version, **kwargs):
            DocumentPage.objects.create(document=doc, version=version, page_number=1,
                                        reconstructed_md="New page text " * 20)
            return "New page text " * 20
        with mock.patch.object(tasks, "extract_document_text", side_effect=extract):
            self.ingest(rescan=True)
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.active_version, 1)
        self.assertEqual(self.doc.extracted_text, "New page text " * 20)
        self.assertTrue(self.doc.pages.filter(version=0).exists())
        client = APIClient()
        client.force_authenticate(self.user)
        response = client.get(f"/auth/documents/{self.doc.id}/pages/")
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["reconstructed_md"], self.doc.extracted_text)

    def test_lost_lease_during_provider_call_does_not_write_vectors(self):
        def embed(batch):
            self.lease.check.side_effect = versions.LeaseLost("expired")
            return [[0.1] * 384 for _ in batch]
        self.embed.side_effect = embed
        with mock.patch.object(tasks.process_document_ingestion, "retry") as retry:
            with self.assertRaises(versions.LeaseLost):
                self.ingest()
        retry.assert_not_called()
        self.index.upsert.assert_not_called()

    def test_chunk_limit_rejects_instead_of_publishing_partial_material(self):
        with mock.patch.object(tasks, "MAX_CHUNKS_PER_DOCUMENT", 1):
            with self.assertRaises(DocumentOversizedError):
                self.ingest()
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.active_version, 0)
        self.index.upsert.assert_not_called()

    @override_settings(INGEST_VERIFY_TIMEOUT=0)
    def test_fetchable_but_not_searchable_vectors_do_not_publish(self):
        self.index.query.side_effect = lambda **kw: {"matches": []}
        with self.assertRaises(TimeoutError):
            self.ingest()
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.active_version, 0)

    @override_settings(INGEST_VERIFY_TIMEOUT=0)
    def test_missing_vector_blocks_activation(self):
        self.index.fetch.side_effect = lambda **kw: {"vectors": {}}
        with self.assertRaises(TimeoutError):
            self.ingest()
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.active_version, 0)

    def test_permanent_oversize_is_not_retried(self):
        with mock.patch.object(tasks, "extract_document_text", side_effect=DocumentOversizedError("201 pages")), \
             mock.patch.object(tasks.process_document_ingestion, "retry") as retry:
            with self.assertRaises(DocumentOversizedError):
                self.ingest(rescan=True)
        retry.assert_not_called()

    def test_duplicate_task_makes_no_database_or_provider_changes(self):
        @contextmanager
        def busy(_):
            yield None
        with mock.patch.object(tasks, "document_lease", side_effect=busy):
            self.assertEqual(self.ingest()["status"], "skipped")
        self.assertFalse(DocumentIndexVersion.objects.exists())
        self.index.upsert.assert_not_called()

    def test_lost_owner_cannot_activate_or_fail_its_successor(self):
        _, first = versions.reserve_version(self.doc.id, self.lease)
        second_lease = mock.Mock(token="owner-two")
        _, second = versions.reserve_version(self.doc.id, second_lease)
        self.assertGreater(second.version, first.version)
        with self.assertRaises(versions.LeaseLost):
            versions.activate_version(self.doc.id, first, self.lease)
        versions.fail_version(self.doc.id, first, self.lease, RuntimeError("stale error"))
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.pending_version, second.version)
        self.assertIsNone(self.doc.error_message)
        versions.activate_version(self.doc.id, second, second_lease)
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.active_version, second.version)

    def test_pdf_chunk_ids_use_exact_page_and_local_chunk_position(self):
        _, revision = versions.reserve_version(self.doc.id, self.lease)
        for page in (1, 2):
            DocumentPage.objects.create(document=self.doc, version=revision.version,
                                        page_number=page, reconstructed_md="Repeated content. " * 25)
        tokenizer = SimpleNamespace(encode=list, decode="".join)
        first = tasks._version_chunks(self.doc, "", tokenizer, revision.version)
        second = tasks._version_chunks(self.doc, "", tokenizer, revision.version)
        self.assertEqual(first, second)
        self.assertTrue(any(key.endswith("p1_c0") for key, _, _ in first))
        self.assertTrue(any(key.endswith("p2_c0") for key, _, _ in first))

    @override_settings(INDEX_VERSION_GRACE_SECONDS=3600)
    def test_cleanup_protects_active_pending_and_recent_versions(self):
        old = timezone.now() - timedelta(hours=2)
        self.doc.active_version = 2
        self.doc.pending_version = 3
        self.doc.save()
        for version, retired in [(1, old), (2, old), (3, old), (4, timezone.now())]:
            DocumentIndexVersion.objects.create(document=self.doc, version=version, retired_at=retired)
            DocumentPage.objects.create(document=self.doc, version=version, page_number=1)
        self.index.list.side_effect = lambda *, prefix: [[prefix + "p1_c0"]]
        with mock.patch.object(tasks, "get_pinecone_index", return_value=self.index), \
             mock.patch.object(tasks, "HYBRID_SEARCH_ENABLED", False):
            tasks.prune_document_versions.run(str(self.doc.id))
        self.index.list.assert_called_once_with(prefix=versions.version_prefix(self.doc.id, 1))
        self.assertEqual(set(self.doc.index_versions.values_list("version", flat=True)), {2, 3, 4})
        self.assertEqual(set(self.doc.pages.values_list("version", flat=True)), {2, 3, 4})

    @override_settings(INDEX_VERSION_GRACE_SECONDS=0)
    def test_cleanup_failure_retains_record_for_retry(self):
        DocumentIndexVersion.objects.create(document=self.doc, version=1, retired_at=timezone.now())
        self.index.list.side_effect = RuntimeError("offline")
        with mock.patch.object(tasks, "get_pinecone_index", return_value=self.index), \
             mock.patch.object(tasks, "HYBRID_SEARCH_ENABLED", False):
            with self.assertRaises(RuntimeError):
                tasks.prune_document_versions.run(str(self.doc.id))
        self.assertTrue(self.doc.index_versions.filter(version=1).exists())

    def test_active_filter_contains_document_version_pairs_and_legacy_guard(self):
        fresh = Document.objects.create(user=self.user, chapter=self.chapter, title="New", active_version=7)
        Document.objects.create(user=self.user, chapter=self.chapter, title="Pending only", pending_version=2)
        result = versions.active_document_filter(self.user.id, self.chapter.id)
        clauses = result["$and"][1]["$or"]
        self.assertEqual(len(clauses), 2)
        pairs = {c["$and"][0]["document_id"]["$eq"]: c["$and"][1] for c in clauses}
        self.assertEqual(pairs[str(self.doc.id)], {"version": {"$exists": False}})
        self.assertEqual(pairs[str(fresh.id)], {"version": {"$eq": 7}})
        self.assertIsNone(versions.active_document_filter("00000000-0000-0000-0000-000000000000"))


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.renewed = 0

    def set(self, key, value, *, nx, ex):
        if key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key):
        return self.values.get(key)

    def eval(self, script, count, key, token, *args):
        if self.values.get(key) != token:
            return 0
        if "expire" in script:
            self.renewed += 1
        else:
            del self.values[key]
        return 1


class DocumentLeaseTests(SimpleTestCase):
    def test_cleanup_discovers_sparse_index_even_when_hybrid_is_disabled(self):
        with mock.patch.object(tasks, "pinecone_client") as client, \
             mock.patch.object(tasks, "HYBRID_SEARCH_ENABLED", False):
            client.has_index.return_value = True
            self.assertIs(tasks._sparse_index_for_cleanup(), client.Index.return_value)
            client.Index.assert_called_once_with(tasks.PINECONE_SPARSE_INDEX)
            client.create_index.assert_not_called()

    def test_cleanup_does_not_create_a_missing_sparse_index(self):
        with mock.patch.object(tasks, "pinecone_client") as client:
            client.has_index.return_value = False
            self.assertIsNone(tasks._sparse_index_for_cleanup())
            client.Index.assert_not_called()
            client.create_index.assert_not_called()

    def test_duplicate_is_rejected_and_expired_owner_cannot_unlock_successor(self):
        redis = FakeRedis()
        with mock.patch.object(versions, "redis_client", redis):
            with versions.document_lease("doc") as first:
                with versions.document_lease("doc") as duplicate:
                    self.assertIsNone(duplicate)
                redis.values[first.key] = "replacement-owner"
                with self.assertRaises(versions.LeaseLost):
                    first.check()
            self.assertEqual(redis.values[first.key], "replacement-owner")

    def test_heartbeat_renews_only_owned_lease(self):
        redis = FakeRedis()
        lease = versions.IngestionLease("doc")
        redis.values[lease.key] = lease.token
        with mock.patch.object(versions, "redis_client", redis), \
             mock.patch.object(lease.stopped, "wait", side_effect=[False, True]):
            lease.heartbeat()
        self.assertEqual(redis.renewed, 1)
        redis.values[lease.key] = "other-owner"
        with mock.patch.object(versions, "redis_client", redis), \
             mock.patch.object(lease.stopped, "wait", return_value=False):
            lease.heartbeat()
        self.assertTrue(lease.lost.is_set())

    def test_redis_failure_does_not_allow_unlocked_ingestion(self):
        with mock.patch.object(versions.redis_client, "set", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                with versions.document_lease("doc"):
                    self.fail("must fail closed")
