# Safe ingestion and bounded query expansion

## Rollout

1. Drain old ingestion workers before deploying this change. Old worker code
   still performs delete-before-write and must not run alongside versioned workers.
2. Apply migrations with `python manage.py migrate`, then deploy the web and worker
   processes together. No production migrations or backfill are run by tests.
3. Existing documents and pages start at version `0`. Completed legacy documents
   remain readable using their unversioned metadata; new writes always carry a
   version. Do not manually delete existing vectors before backfilling.
4. Preview with `python manage.py reindex_hybrid --dry-run`. Queue the backfill
   with `python manage.py reindex_hybrid`, or use `--document UUID --sync` for a
   single document. The active version switches only after verification succeeds.
5. Celery queues delayed cleanup after publication/failure. If broker delivery
   or cleanup fails, run `python manage.py prune_index_versions --dry-run`, then
   `python manage.py prune_index_versions`. Cleanup preserves active/pending
   versions and enforces the grace period. Legacy cleanup uses document-scoped
   metadata deletion, including old random IDs.

## Version lifecycle

The renewable Redis lease is `ingest_lock:doc:{id}` with a 900-second TTL.
Redis errors stop ingestion; a duplicate exits without modifying state. Lease
renewal and release compare the owner token. Each acquired attempt reserves a
monotonically increasing version under a DB row lock. Expired/abandoned version
numbers are never reused, so an old provider request cannot overwrite a newer
attempt. A DB owner-token check also fences publication and failure reporting.

Chunk IDs are `doc_{document_id}_v{version}_p{page}_c{chunk_index}`. Page zero
represents non-page source text. PDF chunking uses actual page boundaries, with
positions assigned after embedding-safe splitting. Dense and sparse writes use
the same IDs. Repeated writes within a version overwrite the same IDs; a new
attempt uses a fresh version and retires abandoned work for delayed cleanup.

Pages and extracted text are staged. The document's active text, pages and vectors
remain available through failed rescans. All expected dense records must be
fetchable with correct metadata and visible through the query path before the
database pointer and extracted text switch in one transaction. This explicit
visibility check accounts for Pinecone's asynchronous writes; see the
[Pinecone SDK data operations documentation](https://sdk.pinecone.io/python/reference/sync-index.html).
Sparse retrieval remains optional, as before, but every search is constrained to
the same active document/version pairs. User-wide fallback also applies these
constraints. Old versions remain for a grace period so in-flight queries can finish.

`Document.error_message` records rebuild failures while an already-readable
document stays `COMPLETED`. Initial ingestion failures still become `FAILED`.
Oversized input and lost ownership are not automatically retried. After a worker
crash, a new ingestion can acquire the expired lease and reserve a fresh version.

## Limits

- `MAX_PAGES_PER_DOCUMENT`: 200 by default; checked before any page rendering.
- Rendering: at most 150 DPI and a 2048 × 2048 bounding box, RGB without alpha.
- PDF input is copied in blocks to a temporary file. Each page is rendered,
  uploaded, extracted/OCRed and saved before processing the next. File, generator
  and image resources close on failure. This bounds accumulated render buffers;
  it is not a fixed total-process-memory guarantee for arbitrary PDF internals.
- Documents exceeding the existing 1,000-chunk budget are rejected instead of
  silently publishing only their first chunks.
- `INGEST_VERIFY_TIMEOUT`: 60 seconds for visibility verification.
- `INDEX_VERSION_GRACE_SECONDS`: 3600 seconds before pruning retired versions.
- Expansion accepts 1–3 strict strings, 3–150 characters each. The submitted
  question is always first. Contextualization shares the three-alternative budget.
  Invalid output falls back to just the original question; duplicate alternatives
  and unsafe embedding inputs are removed. Maximum batch size is four queries.

## Regression checks

`python manage.py test accounts.tests.test_query_expansion accounts.tests.test_page_pipeline accounts.tests.test_ingestion_versions accounts.tests.test_ingestion_safety accounts.tests.test_pipeline_outcomes accounts.tests.test_hybrid_retrieval accounts.tests.test_views accounts.tests.test_models`

These use an isolated SQLite database and offline dependency doubles. They do not
replace a deployment smoke test against Redis, PostgreSQL and both Pinecone indexes.
