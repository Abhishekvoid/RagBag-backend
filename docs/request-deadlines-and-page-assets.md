# Chat deadlines, page assets, and summaries

Chat processing has one 30-second monotonic deadline, starting in the chat
endpoint before document/history queries. Context variables carry it through
async tasks and Django's sync bridge. Each HTTP attempt uses the lesser of its
stage timeout and remaining time. Queueing and retry sleeps consume the same
budget; retries stop when their next backoff cannot fit. OpenAI and Pinecone SDK
retries are disabled so they cannot multiply the application's retry loop.

Deadline exhaustion returns a retryable HTTP 504 and does not save either chat
message. SQL waits are bounded on PostgreSQL; cancellation uses native async
Pinecone reads instead of leaving synchronous retrieval threads running.
Network cancellation and connection cleanup can add a small amount of overhead
to the deadline; this is an application budget, not an exact response-time SLA.

Each ingestion task execution starts a separate 900-second budget. Page and
batch boundaries check it, network calls have finite timeouts, and publishing
checks the budget again. An exhausted execution fails its pending version while
preserving the existing active version. Celery's soft limit is 900 seconds and
the emergency hard limit is 930 seconds, allowing cleanup. Production should use
the prefork worker pool: Celery process time limits are not enforced by the
Windows solo pool. A new manual ingestion attempt gets a new budget; ordinary
transient failures still use the existing bounded Celery retry policy.

## Migration and rollout

1. Stop old ingestion workers before upgrading: old code writes image URLs
   instead of object keys.
2. Run `python manage.py migrate` against the intended deployment database.
   Migration `0013_documentpage_s3_object_key` adds the durable key and backfills
   recognized native S3, configured S3-compatible, and local-media URLs. It does
   not download images or contact storage. Existing URL values remain available
   for recovery. Review any warning reporting unrecognized URLs; those rows
   need their correct key supplied or their source document rescanned.
3. Start the updated web and ingestion workers. New page images store only their
   object key. Reindexing copies keys into the next version.

The page API checks ownership and serves only active page versions, then signs
S3 image URLs for 900 seconds on every read. Local filesystem storage still
returns local media URLs. Unknown legacy URLs produce an empty `image_url`
rather than an expired link. Existing clients should refetch page data when a
signed URL expires; URLs are temporary and must not be saved as permanent assets.

## Summaries

The summary route reads active canonical `DocumentPage.reconstructed_md` records
for the user's chapter, without vector retrieval. Documents with no page records
retain the extracted-text fallback. Blank canonical pages never fall back to
possibly stale extracted text.

Maps group up to five pages, splitting oversized inputs into at most 12,000
characters without truncating the chapter. At most three map/reduce calls run
concurrently. Intermediate summaries are bounded to 4,000 characters and reduced
hierarchically into an overview, definitions, supported formulas, and takeaways.
Success is `SUMMARY_GENERATED`, persisted as a normal answered chat turn.

The same 30-second chat deadline applies to the entire summary. A long chapter
or slow provider can return 504; no partial guide is presented as complete.
Background summary generation and caching are outside this change.

## Offline validation

Run `python manage.py test accounts.tests`. Providers are mocked. The regression
tests cover shared/recomputed budgets, retry backoff, queue cancellation,
concurrent request isolation, native retrieval cancellation, ingestion expiry,
HTTP 504 persistence behavior, storage signing/backfill/ownership, actual schema
upgrade, summary coverage, active versions, reduction, and cancellation.
