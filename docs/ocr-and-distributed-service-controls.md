# OCR coverage and distributed service controls (issues 10 and 11)

## Page extraction and coverage

A PDF page with an inadequate text layer uses the configured vision service when enabled and within its page cap. If an image page still lacks usable text, Tesseract reads the already-rendered PNG. This also works when vision is disabled, unavailable, returns empty text, or the page exceeds the vision cap. The existing Docker image already installs Tesseract and requirements include pytesseract; no additional Python OCR package is introduced.

OCR defaults to English and 20 seconds per page (`OCR_LANGUAGES`, `OCR_TIMEOUT`). The timeout shrinks with the ingestion deadline and pytesseract terminates an overdue OCR subprocess. Install language data for any additional configured languages. Rendering remains bounded to 150 dpi / 2048 pixels as before.

Image detection uses the images actually displayed on the PDF page, including inline images absent from its resource list. After extraction, fewer than 20 alphanumeric characters marks `DocumentPage.needs_review` and records a fixed public `review_reason`. Whitespace and punctuation do not count. A genuinely blank page with no image is not flagged. This is a minimum coverage check, not a guarantee that OCR recognized every equation correctly.

Unreadable pages retain their original scan and partial transcription for review but are excluded from canonical text, vector chunks and chapter summaries. No usable chunks means ingestion fails. A partial usable document can publish with a coverage warning.

`INCOMPLETE_COVERAGE` is logged and broadcast, and its version/page/reason list is persisted on `DocumentIndexVersion.coverage_audit`. `Document.coverage_audit` describes the latest extracted attempt; `active_coverage` describes the published version. A failed rebuild preserves the previous text, pages, vectors and active coverage. This distinction prevents a failed new upload from mislabelling old evidence. Audit fields are read-only in the API and existing document ownership checks apply.

The reader displays review notices and scans. Document warnings survive refresh, including latest failed attempts; failed new uploads with coverage alerts are restored from the documents API. New indexed chunks carry `incomplete_coverage`, and answers based on those chunks prepend an advisory. Summaries apply the same exclusion/disclosure to their active page source set.

## Shared admission and recovery

HTTP and Celery workers use the same Redis key namespace and service names. Every provider attempt goes through the common controls: asynchronous chat, synchronous AI views, ingestion title generation, embeddings, vision and enabled reranking. Retry backoff occurs outside the slot lease. Redis failure blocks admission; vision and reranking retain their existing optional fallbacks.

| Service | Default active calls | Default queued calls |
| --- | ---: | ---: |
| Embedding | 3 | 5 |
| LLM | 20 | 100 |
| Vision | 1 | 5 |
| Reranking | 3 | 5 |

The corresponding `*_MAX_CONCURRENT` and `*_MAX_QUEUE` variables are listed in `.env.example`. Use identical configuration for workers calling the same service. Limits count provider calls, not documents or chat turns. Celery rate limits alone would not protect HTTP callers, so they are not used as a substitute.

Atomic Lua scripts use Redis time, prune expired tokens, and enforce active/queue capacity across processes. Admission waits at most five seconds or the remaining request deadline. Active leases expire after 90 seconds and renew every 30 seconds while held; release removes only the owner’s token. A worker crash eventually frees its slot. Cancellation waits for any in-flight admission write before releasing, avoiding a leaked late acquisition. Queue overload does not count as a downstream failure.

Breakers open after three consecutive recorded failed attempts and cool down for 60 seconds. Recovery requires an atomic `SET NX EX` canary lease. Other callers fail immediately while the canary runs. It receives one provider attempt; normal retry decorators cannot launch a canary retry burst. The probe lease is 90 seconds; asynchronous attempts have a hard bound of 80 seconds including admission, and synchronous probes renew their lease. Provider SDK timeouts also shrink with the request deadline. Before calling the provider, queued permits are revalidated against current state/generation.

Only the current generation and probe owner can close/reopen a recovering circuit. Late results from older generations cannot reset it. Metrics inspect state without transitioning or claiming a probe. Lease renewal failure rejects the result; expired workers cannot renew an already-expired token. As with any expiring lease, a suspended client or an already-submitted remote operation cannot be physically stopped by Redis: downstream request timeouts remain necessary.

## Rollout

1. Back up and apply migration `0014_page_coverage` using the normal deployment migration step. Existing pages default to no review flag; old documents are not rescanned by the migration.
2. Confirm the OCR worker image contains Tesseract plus the configured language data, and that every worker uses the same reachable Redis and service-control namespace.
3. Drain old HTTP/Celery workers and restart all workers with the new controls. A mixed deployment retains old workers that bypass distributed limits. New control keys use `:v2`; old `cb:*` keys are intentionally not reused.
4. Deploy the frontend. Rescan affected existing PDFs through the existing rescan action to populate OCR text, review flags, coverage audits and new vector metadata.

No production migrations, document rescans or deployment are performed by these code changes.

## Validation

Run backend tests against a disposable Redis instance, not production:

```powershell
$env:REDIS_URL='redis://127.0.0.1:16389/0'
$env:TEST_REDIS_URL=$env:REDIS_URL
$env:SERVICE_CONTROL_NAMESPACE='test-run-unique-id'
.venv/Scripts/python.exe manage.py test accounts.tests --noinput
```

Distributed integration/deadline tests require `TEST_REDIS_URL`; other provider tests mock admission and do not contact Redis. Integration tests cover four processes sharing capacity, ten contenders sharing one canary, stale results, lease expiry/renewal, unavailable Redis, bounded queues, cancellation and embedding failure accounting. OCR tests cover disabled/failed/empty/capped vision, unavailable OCR, short text, deadline bounds, source exclusion, publication and durable versioned warnings. Actual Tesseract recognition should also be smoke-tested in the deployment image; the Windows development host lacks the executable.

Frontend: `npm run build`, then run `scripts/check-coverage.cjs` and `scripts/check-citations.cjs` against the local production server. These browser checks use offline API fixtures at desktop and mobile widths and cover refresh persistence, failed rebuilds, page scans and citation navigation.
