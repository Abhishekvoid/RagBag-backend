# Delete for Chapters, Subjects & Loose Chapters — Design

**Date:** 2026-07-21
**Status:** Approved (pending spec review)

## Goal

Let a user delete a chapter, a subject, or a loose (uncategorized) chapter from the
notebook sidebar. Deletion is a **full cascade**: the chapter/subject and all of its
documents, document files, generated questions, flashcards, and Pinecone vectors are
removed. The action is irreversible and gated behind a confirmation dialog.

## Current State (what already exists)

- **Backend endpoints already accept DELETE.** `SubjectDetailView` and `ChapterDetailView`
  are `RetrieveUpdateDestroyAPIView`s at `/auth/subjects/<id>/` and `/auth/chapters/<id>/`.
  The default `destroy` only removes the DB row — it does **not** clean up documents,
  files, or vectors.
- **Model cascade today (insufficient for full cascade):**
  - `Document.chapter` → `SET_NULL` (documents survive as loose docs when a chapter is deleted).
  - `Subject` delete → `Chapter` rows `CASCADE`; their documents then `SET_NULL`.
  - `GenerateQuestion.chapter`, `GenerateFlashCards.chapter` → `CASCADE` (auto-removed).
  - `ChatSession.chapter` / `ChatSession.subject` → `SET_NULL` (sessions detach, survive).
- **Vectors:** Pinecone **serverless** index (`ServerlessSpec`). Vectors carry metadata
  `document_id`, `user_id`, `chapter_id` (when in a chapter), `text`, `file_type`, and use
  **random UUID IDs** with no namespace. Serverless indexes do **not** support
  delete-by-metadata-filter, and random IDs can't be reconstructed — so existing vectors
  are not targetable for deletion.
- **Frontend:** `notebookApi.deleteSubject` / `deleteChapter` / `deleteDocument` exist.
  The store has `deleteSubject` but **not** `deleteChapter`. There are **no delete buttons
  in the UI**. Subject chapters render via an inline `ChapterItem` inside `SubjectItem.tsx`;
  loose chapters render via the standalone `ChapterItem.tsx`.

## Key Decisions

1. **Delete depth:** Full cascade — documents + files + vectors are removed.
2. **Vector cleanup:** Prefix vector IDs with `document_id`, then list-and-delete by prefix.
   Cleans up all **new** documents. Old (pre-change) vectors stay orphaned but are
   **harmless**: every query filters by `chapter_id`/`user_id`, and chapter IDs are UUIDs
   that are never reused, so a deleted chapter's vectors can never surface again.
3. **Async cleanup:** DB rows are deleted in the request (fast 204). Vector + file cleanup
   is offloaded to a Celery task, since a subject delete can cascade across many
   documents.
4. **Chat sessions:** Left intact (`SET_NULL` detach) to preserve chat history. Not deleted.
5. **UX:** Hover-revealed trash icon per row + confirmation dialog before deleting.

## Backend Design

### 1. Prefixed vector IDs (ingestion)

In `accounts/tasks.py`, change the upsert point ID from `str(uuid.uuid4())` to
`f"{document_id}#{uuid.uuid4()}"`. Everything else (metadata, search, filters) is
unchanged. Search never references vector IDs, so this is transparent to retrieval.

### 2. Cleanup Celery task

New task in `accounts/tasks.py`:

```
@shared_task
def cleanup_document_data(document_ids: list[str], file_names: list[str]) -> None
```

- For each `document_id`: page `index.list(prefix=f"{document_id}#")` and call
  `index.delete(ids=<page>)` per page. Guard for empty results.
- For each `file_name`: delete from Django storage
  (`default_storage.delete(name)` / `Document.file.storage`), best-effort.
- Wrap each unit in try/except with logging; a failure in one document must not abort the
  rest. This task is best-effort hygiene, not user-blocking.

### 3. Destroy overrides

Override `perform_destroy` on both detail views.

**`ChapterDetailView.perform_destroy(instance)`**
1. Collect `document_ids` and `file_names` from `instance.documents.all()`.
2. `cleanup_document_data.delay(document_ids, file_names)`.
3. `instance.documents.all().delete()` (explicit — overrides the `SET_NULL` default so docs
   are truly removed).
4. `instance.delete()` (removes chapter; `GenerateQuestion`/`GenerateFlashCards` cascade).

**`SubjectDetailView.perform_destroy(instance)`**
1. Across `instance.chapters.all()`, collect all `document_ids` + `file_names`.
2. `cleanup_document_data.delay(...)`.
3. Delete all documents belonging to those chapters.
4. `instance.delete()` (chapters cascade).

Ownership is already enforced by each view's `get_queryset` (`user=self.request.user`),
so a user can only delete their own subjects/chapters.

Loose chapters need **no** new backend code — they are chapters with `subject=null`,
already handled by `ChapterDetailView`.

## Frontend Design

### 4. Store: `deleteChapter`

Add to `useNotebook.ts`, mirroring `deleteSubject`:
- Call `notebookApi.deleteChapter(id)`.
- Optimistically remove the chapter from `subjects[*].chapters`, matching whichever
  subject holds it — including the `"uncategorized-chapters"` pseudo-subject that carries
  loose chapters.
- On error, set the store error (same pattern as `deleteSubject`).

Add `deleteChapter` to the store's TypeScript interface.

### 5. Reusable `ConfirmDialog`

New component built on the existing `dialog.tsx` primitive (no `alert-dialog` exists):
props `open`, `onOpenChange`, `title`, `description`, `confirmLabel`, `onConfirm`, and a
destructive-styled confirm button. Used for both subject and chapter deletion.

### 6. Trash affordances + wiring

- `NotebookSidebar` owns a single `ConfirmDialog` and pending-target state
  `{ kind: 'subject' | 'chapter', id, name } | null`.
- It passes `onDeleteSubject(id, name)` and `onDeleteChapter(id, name)` down.
- `SubjectItem`: add a hover-revealed trash button in the header (next to the existing `+`),
  calling `onDeleteSubject`. Its inline `ChapterItem` gains a hover trash button →
  `onDeleteChapter`.
- Standalone `ChapterItem.tsx` (loose chapters): add an optional `onDelete?` prop rendering
  a hover trash button.
- Confirm copy: e.g. *"Delete "<name>" and all its documents? This can't be undone."*
  (subject copy notes it also removes all chapters).

### Data Flow

```
trash click → NotebookSidebar sets pending target + opens ConfirmDialog
            → user confirms
            → store.deleteChapter(id) | store.deleteSubject(id)
            → optimistic removal from `subjects` + DELETE request
            → backend perform_destroy: enqueue cleanup task, delete DB rows → 204
            → Celery cleanup_document_data: delete vectors by prefix + files
```

## Testing

- **Backend:** unit tests for `perform_destroy` on chapter and subject — assert documents +
  questions + flashcards are gone, `cleanup_document_data` was enqueued with the right
  document IDs/file names, chat sessions survive (detached), and a user cannot delete
  another user's chapter/subject (404). Test the cleanup task's prefix-list-and-delete with
  a mocked Pinecone index.
- **Frontend:** store test for `deleteChapter` removing from both a real subject and the
  uncategorized pseudo-subject; confirm-dialog gating (no DELETE fires until confirm).

## Out of Scope

- Bulk / multi-select delete.
- Undo / soft-delete / trash bin.
- Deleting orphaned pre-change vectors (harmless; not worth a migration).
- Rename or other row actions (the kebab-menu option was declined in favor of a trash icon).
