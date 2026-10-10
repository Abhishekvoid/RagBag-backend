# Chapter scope and page citations

Chat searches the current chapter by default. An empty search returns
`INSUFFICIENT_EVIDENCE`, `is_unanswered: true`, and no sources. It does not synthesize
an answer. `POST /auth/rag-chat/` accepts the optional JSON boolean
`allow_library_fallback` (default `false`; strings and numbers are rejected).
Only explicit `true` permits a second search after the chapter has no usable
chunks. That search excludes the current chapter and retains the authenticated
user and active document/version filters. Provider errors never broaden scope.
This applies to retrieval questions; summaries remain chapter-scoped.

Fallback chunks carry request-local `is_fallback_scope: true`. Reranking and
source serialization preserve it. Application code prefixes successful fallback
answers with:

> I couldn't find this in the current chapter, but here is what I found in your other notes...

Both chat views offer **Search my other notes** for the latest unanswered question.
This sends one explicitly permitted request; ordinary messages continue to use
chapter-only retrieval. The implementation does not introduce a relevance-score
threshold: nonempty retrieval is not a guarantee of semantic relevance.

## Citation contract

Page chunking is performed separately for each canonical `DocumentPage`.
Additional token-budget splitting and overlap remain inside that page. Dense
and sparse records share `document_id`, `page_number`, and `version`. Non-paginated
content has no `page_number` metadata; internal `p0` chunk IDs are not page claims.
Legacy vectors without a version are treated as document-level citations, even
if they contain an old guessed page number.

Each answer source includes `document_id`, `chapter_id`, `title`, `snippet`,
`page_number` (nullable), `version` (nullable), and `is_fallback_scope`. Different
pages within the same document remain distinct; there are at most eight selected
evidence chunks. This shape is saved in the existing JSON citations field and
retained on history reload. No database migration is required.

The UI opens the owning chapter/document and focuses the cited page, exposing
its available scan. Page requests carry `?version=N`. The page API checks
ownership first and returns 409 if that version is no longer active. It never
substitutes a newer version. Missing pages and network errors show an explicit
message. Retired versions need not remain stored indefinitely for old links;
students can ask again for current citations.

## Targeted legacy rollout (operator action, not run by this change)

1. Deploy the updated backend, frontend, and ingestion workers together. Stop old
   ingestion workers before allowing new jobs.
2. Identify affected document IDs from the intended deployment's records: legacy
   active version zero, PDFs with no active page records, or known old page guesses.
3. Inspect a single target with
   `python manage.py reindex_hybrid --document DOCUMENT_UUID --dry-run`.
4. After approving provider usage, run
   `python manage.py reindex_hybrid --document DOCUMENT_UUID` to queue it (or add
   `--sync` for a synchronous operator run). Do not omit `--document` for a targeted
   repair. This uses the existing versioned publication and cleanup workflow.
5. PDFs with no page records are automatically extracted again. PDFs with existing
   canonical pages reuse those pages. Reindexing preserves the old active version
   until replacement vectors are verified; a failure keeps the old version usable.
6. Check a new answer against the source scan. Previously saved document-level
   citations are not retroactively assigned guessed pages. Previously versioned
   citations become stale after replacement and prompt the student to ask again.

Reindexing consumes embedding quota and can consume OCR quota. No production
reindexing or deployment is performed by the regression tests.

## Verification

`python manage.py test accounts.tests --noinput` exercises retrieval scope,
explicit opt-in, provider failures, source persistence, ownership/version guards,
repeated page headers, both vector indexes, and unpaginated/legacy ingestion.
The frontend browser check in `scripts/check-citations.cjs` uses a local running
frontend with mocked API responses, plus Playwright supplied by the development
environment. It does not require real user data or live providers.
