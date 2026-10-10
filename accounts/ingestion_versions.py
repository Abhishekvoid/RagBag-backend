"""Document leases, publication fencing, and version-scoped vector operations."""
import logging
import threading
import time
import uuid
from contextlib import contextmanager

from django.conf import settings
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from utils.circuit_breaker import redis_client
from utils.deadline import check_deadline, timeout_for
from .models import Document, DocumentIndexVersion

logger = logging.getLogger(__name__)
LEASE_SECONDS = 15 * 60


class LeaseLost(RuntimeError):
    pass


class IngestionLease:
    def __init__(self, document_id):
        self.key = f"ingest_lock:doc:{document_id}"
        self.token = uuid.uuid4().hex
        self.stopped = threading.Event()
        self.lost = threading.Event()

    def check(self):
        check_deadline()
        if self.lost.is_set() or redis_client.get(self.key) != self.token:
            raise LeaseLost("Document ingestion lease was lost")

    def heartbeat(self):
        while not self.stopped.wait(LEASE_SECONDS / 3):
            try:
                renewed = redis_client.eval(
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end",
                    1, self.key, self.token, LEASE_SECONDS,
                )
                if not renewed:
                    self.lost.set()
                    return
            except Exception:
                self.lost.set()
                return


@contextmanager
def document_lease(document_id):
    lease = IngestionLease(document_id)
    if not redis_client.set(lease.key, lease.token, nx=True, ex=LEASE_SECONDS):
        yield None
        return
    thread = threading.Thread(target=lease.heartbeat, daemon=True)
    thread.start()
    try:
        yield lease
    finally:
        lease.stopped.set()
        # Owner comparison prevents a delayed worker from unlocking its successor.
        try:
            redis_client.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end",
                1, lease.key, lease.token,
            )
        except Exception:
            logger.warning("Could not release ingestion lease; it will expire")


def reserve_version(document_id, lease):
    with transaction.atomic():
        doc = Document.objects.select_for_update().get(id=document_id)
        lease.check()
        # Never reuse an abandoned generation: an expired worker may still be
        # finishing a provider call into that generation's vectors or pages.
        if doc.pending_version is not None:
            doc.index_versions.filter(version=doc.pending_version).update(retired_at=timezone.now())
        doc.version_counter += 1
        doc.pending_version = doc.version_counter
        doc.ingestion_token = lease.token
        doc.error_message = None
        if doc.status != Document.STATUS_COMPLETED and doc.active_version == 0:
            doc.status = Document.STATUS_PROCESSING
        doc.save(update_fields=["version_counter", "pending_version", "ingestion_token", "error_message", "status"])
        revision = DocumentIndexVersion.objects.create(document=doc, version=doc.pending_version)
    return doc, revision


def activate_version(document_id, revision, lease):
    with transaction.atomic():
        doc = Document.objects.select_for_update().get(id=document_id)
        lease.check()
        if doc.ingestion_token != lease.token or doc.pending_version != revision.version:
            raise LeaseLost("A newer ingestion owns this document")
        # Include legacy version zero in delayed cleanup after its replacement.
        old, _ = DocumentIndexVersion.objects.get_or_create(document=doc, version=doc.active_version)
        old.retired_at = timezone.now()
        old.save(update_fields=["retired_at"])
        doc.active_version = revision.version
        doc.pending_version = None
        doc.ingestion_token = ""
        doc.extracted_text = revision.extracted_text
        doc.status = Document.STATUS_COMPLETED
        doc.error_message = None
        doc.save(update_fields=["active_version", "pending_version", "ingestion_token", "extracted_text", "status", "error_message"])


def fail_version(document_id, revision, lease, error):
    with transaction.atomic():
        doc = Document.objects.select_for_update().get(id=document_id)
        if doc.ingestion_token != lease.token or doc.pending_version != revision.version:
            return
        revision.retired_at = timezone.now()
        revision.save(update_fields=["retired_at"])
        doc.pending_version = None
        doc.ingestion_token = ""
        doc.error_message = str(error)
        if doc.active_version == 0 and doc.status != Document.STATUS_COMPLETED:
            doc.status = Document.STATUS_FAILED
        doc.save(update_fields=["pending_version", "ingestion_token", "error_message", "status"])


def active_document_filter(user_id, chapter_id=None, *, exclude_chapter_id=None):
    """Snapshot allowed document/version pairs, including the legacy baseline.

    Return None for no readable documents; callers must not send an unfiltered
    search. Legacy vectors lack version metadata, unlike every new write.
    """
    docs = Document.objects.filter(user_id=user_id).filter(
        Q(active_version__gt=0) | Q(status=Document.STATUS_COMPLETED)
    )
    if chapter_id is not None:
        docs = docs.filter(chapter_id=chapter_id)
    if exclude_chapter_id is not None:
        docs = docs.exclude(chapter_id=exclude_chapter_id)
    clauses = []
    for doc_id, version in docs.values_list("id", "active_version"):
        clauses.append({"$and": [
            {"document_id": {"$eq": str(doc_id)}},
            {"version": {"$eq": version}} if version else {"version": {"$exists": False}},
        ]})
    if not clauses:
        return None
    scope = [{"user_id": {"$eq": str(user_id)}}, {"$or": clauses}]
    if chapter_id is not None:
        scope.append({"chapter_id": {"$eq": str(chapter_id)}})
    return {"$and": scope}


def verify_vectors(index, ids, document_id, version, lease, *, probe_vector=None):
    """Wait for every expected record to be visible; never accept a count alone."""
    deadline = time.monotonic() + settings.INGEST_VERIFY_TIMEOUT
    remaining = set(ids)
    while True:
        lease.check()
        for start in range(0, len(ids), 100):
            batch = [key for key in ids[start:start + 100] if key in remaining]
            if not batch:
                continue
            response = index.fetch(ids=batch, timeout=timeout_for(10))
            vectors = response.get("vectors", {}) if isinstance(response, dict) else response.vectors
            for key, vector in vectors.items():
                metadata = vector.get("metadata", {}) if isinstance(vector, dict) else vector.metadata
                if str(metadata.get("document_id")) == str(document_id) and metadata.get("version") == version:
                    remaining.discard(key)
        if not remaining:
            # Fetch visibility alone is not enough: also wait for the search
            # path to see the full manifest before publishing the DB pointer.
            response = index.query(vector=probe_vector, top_k=len(ids),
                                   filter={"$and": [
                                       {"document_id": {"$eq": str(document_id)}},
                                       {"version": {"$eq": version}},
                                   ]}, include_metadata=False, include_values=False,
                                   timeout=timeout_for(10))
            matches = response.get("matches", []) if isinstance(response, dict) else response.matches
            visible = {m.get("id") if isinstance(m, dict) else m.id for m in matches}
            if set(ids).issubset(visible):
                return
        if time.monotonic() >= deadline:
            raise TimeoutError("Index verification timed out; the complete version is not searchable")
        time.sleep(timeout_for(1))


def version_prefix(document_id, version):
    return f"doc_{document_id}_v{version}_"
