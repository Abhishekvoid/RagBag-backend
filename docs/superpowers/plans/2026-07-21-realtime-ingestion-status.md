# Real-Time Ingestion Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a live, phase-by-phase loading screen (Uploading → Reading → Naming → Chunking → Embedding → Storing → Ready) the instant a user uploads a document, and auto-open the finished chapter — no blank gap, no reload.

**Architecture:** The backend already has a per-user WebSocket group (`user_{id}`) via Django Channels. We fix the consumer (which currently drops all payload except `message`), add a single structured `ingestion_status` event pushed from each Celery pipeline step, and add a retry endpoint. The frontend keeps an `ingestions` map keyed by `document_id`, seeded optimistically from the upload POST and on mount from in-flight documents, reduced live by WebSocket events, and rendered as a center-panel checklist plus a sidebar row that swaps into the real chapter.

**Tech Stack:** Backend — Django 5, Django Channels, Celery, Redis channel layer. Frontend — Next.js (App Router), Zustand, axios, TypeScript, framer-motion.

## Global Constraints

- Backend tests use Django `TestCase` / `APIClient`; run with `python manage.py test accounts`. Use the project `venv`: `./venv/Scripts/python.exe manage.py test accounts`.
- Frontend has **no test runner**. Verify with `npx tsc --noEmit` and `npm run lint` from `frontend/RAG_tutor_frontend`, plus manual browser run. Keep pure logic (reducer) in its own module so it is trivially reasoned about; do NOT add a test framework.
- The WebSocket join key between optimistic UI and live events is always `document_id` (string UUID).
- Reuse the existing `Document.status` field (PENDING/PROCESSING/COMPLETED/FAILED) — **no new DB fields, no migrations**.
- Backend paths are relative to `C:\Users\Abhishek\RAG tutor\backend`. Frontend paths are relative to `C:\Users\Abhishek\RAG tutor\frontend\RAG_tutor_frontend`.
- Phase string constants (exact): `reading`, `naming`, `chunking`, `embedding`, `storing`, `ready`, `failed`. The frontend adds a client-only `uploading` phase.
- Commit after every task.

---

## File Structure

**Backend**
- Create `accounts/realtime.py` — pure payload builder `build_ingestion_status()` + side-effecting `push_ingestion_status()`. Single home for the WS event contract.
- Modify `accounts/consumers.py` — forward full payload, keep back-compat with old `{"message": ...}` events.
- Modify `accounts/tasks.py` — call `push_ingestion_status()` at each phase in `create_chapter_from_document` and `process_document_ingestion`.
- Modify `accounts/views.py` + `accounts/urls.py` — add `DocumentRetryView` and its route.
- Modify `accounts/tests.py` — tests for builder, consumer, retry endpoint.

**Frontend**
- Create `src/features/notebook/ingestion.ts` — types (`IngestionPhase`, `Ingestion`, `IngestionStatusEvent`) + pure reducer `applyIngestionEvent()`.
- Modify `src/features/notebook/notebook.api.ts` — `uploadDocument` gains an `onUploadProgress` param; add `retryDocument`.
- Modify `src/lib/store/useNotebook.ts` — `ingestions` map + actions + WS `onmessage` wiring + mount seed.
- Create `src/components/dashboard/ContentPanel/IngestionProgress.tsx` — center-panel phase checklist + failure card.
- Modify `src/components/dashboard/ContentPanel/AddSourceView.tsx` — optimistic entry + upload progress.
- Modify `src/components/dashboard/SideBar/NotebookSidebar.tsx` — processing rows for in-flight ingestions.
- Modify `src/app/dashboard/page.tsx` (or the ContentPanel host) — render `IngestionProgress`, auto-open on ready.

---

## Task 1: Fix the WebSocket consumer to forward full payloads

This is the root cause of `document_id` never reaching the browser.

**Files:**
- Modify: `accounts/consumers.py`
- Test: `accounts/tests.py`

**Interfaces:**
- Produces: A consumer whose `send_notification(event)` sends `event["data"]` (a dict) verbatim when present, else falls back to `{"message": event["message"]}` for legacy events.

- [ ] **Step 1: Write the failing test**

Add to `accounts/tests.py`:

```python
from channels.testing import WebsocketCommunicator
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from core.asgi import application  # ASGI app with auth + routing


class IngestionWebSocketTests(TestCase):
    def test_consumer_forwards_full_data_payload(self):
        async def scenario():
            user = await self._make_user()
            communicator = WebsocketCommunicator(
                application, "/ws/notifications/"
            )
            communicator.scope["user"] = user  # bypass token middleware in test
            connected, _ = await communicator.connect()
            assert connected
            layer = get_channel_layer()
            await layer.group_send(
                f"user_{user.id}",
                {"type": "send_notification",
                 "data": {"type": "ingestion_status",
                          "document_id": "abc", "phase": "embedding"}},
            )
            msg = await communicator.receive_json_from()
            assert msg == {"type": "ingestion_status",
                           "document_id": "abc", "phase": "embedding"}
            await communicator.disconnect()

        import asyncio
        from channels.db import database_sync_to_async
        User = get_user_model()

        @database_sync_to_async
        def make_user():
            return User.objects.create_user(
                email="ws@b.com", password="x", name="WS")

        self._make_user = make_user
        asyncio.get_event_loop().run_until_complete(scenario())
```

> Note: if `WebsocketCommunicator` scope injection is awkward with the existing token middleware, instead assert directly against the consumer method by constructing `NotificationConsumer`, setting a fake `self.send`, and calling `send_notification`. Use whichever the codebase's middleware allows; the behavioral assertion (data forwarded verbatim) is the requirement.

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe manage.py test accounts.IngestionWebSocketTests -v 2`
Expected: FAIL — consumer currently sends `{"message": ...}` and drops `data`.

- [ ] **Step 3: Implement the consumer change**

Replace the `send_notification` method in `accounts/consumers.py`:

```python
    async def send_notification(self, event):
        # New structured events carry a full "data" dict; forward verbatim.
        if "data" in event:
            await self.send(text_data=json.dumps(event["data"]))
            return
        # Back-compat: legacy events only carry a "message" string.
        await self.send(text_data=json.dumps({"message": event["message"]}))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe manage.py test accounts.IngestionWebSocketTests -v 2`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add accounts/consumers.py accounts/tests.py
git commit -m "fix(ws): forward full ingestion payload from notification consumer"
```

---

## Task 2: Ingestion status contract (`accounts/realtime.py`)

**Files:**
- Create: `accounts/realtime.py`
- Test: `accounts/tests.py`

**Interfaces:**
- Produces:
  - `PHASE_READING="reading"`, `PHASE_NAMING="naming"`, `PHASE_CHUNKING="chunking"`, `PHASE_EMBEDDING="embedding"`, `PHASE_STORING="storing"`, `PHASE_READY="ready"`, `PHASE_FAILED="failed"`.
  - `build_ingestion_status(document_id, phase, *, chapter_id=None, title=None, batch=None, total_batches=None, error=None) -> dict` — pure; includes only provided optional keys; always includes `type="ingestion_status"`, `document_id` (str), `phase`.
  - `push_ingestion_status(user_id, document_id, phase, **extra) -> dict` — builds payload and `group_send`s to `user_{user_id}` with `{"type": "send_notification", "data": payload}`; returns the payload.

- [ ] **Step 1: Write the failing test**

Add to `accounts/tests.py`:

```python
from accounts.realtime import build_ingestion_status


class IngestionStatusBuilderTests(TestCase):
    def test_minimal_payload(self):
        p = build_ingestion_status("doc1", "reading")
        self.assertEqual(p, {"type": "ingestion_status",
                             "document_id": "doc1", "phase": "reading"})

    def test_optional_fields_included_only_when_set(self):
        p = build_ingestion_status(
            "doc2", "embedding", batch=2, total_batches=5)
        self.assertEqual(p["batch"], 2)
        self.assertEqual(p["total_batches"], 5)
        self.assertNotIn("chapter_id", p)
        self.assertNotIn("error", p)

    def test_naming_carries_chapter_and_title(self):
        p = build_ingestion_status(
            "doc3", "naming", chapter_id="ch9", title="Data Structures")
        self.assertEqual(p["chapter_id"], "ch9")
        self.assertEqual(p["title"], "Data Structures")

    def test_document_id_coerced_to_str(self):
        import uuid
        u = uuid.uuid4()
        p = build_ingestion_status(u, "ready")
        self.assertEqual(p["document_id"], str(u))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe manage.py test accounts.IngestionStatusBuilderTests -v 2`
Expected: FAIL — `ModuleNotFoundError: No module named 'accounts.realtime'`.

- [ ] **Step 3: Create `accounts/realtime.py`**

```python
"""Real-time ingestion status contract.

One structured WebSocket event type, `ingestion_status`, is pushed to the
per-user Channels group `user_{id}` at each pipeline phase. The frontend
reduces these into an ingestions map keyed by document_id.
"""
import logging
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)

PHASE_READING = "reading"
PHASE_NAMING = "naming"
PHASE_CHUNKING = "chunking"
PHASE_EMBEDDING = "embedding"
PHASE_STORING = "storing"
PHASE_READY = "ready"
PHASE_FAILED = "failed"


def build_ingestion_status(document_id, phase, *, chapter_id=None, title=None,
                           batch=None, total_batches=None, error=None) -> dict:
    """Pure builder for an ingestion_status payload. Optional keys are
    omitted when None so the frontend reducer can treat missing == unchanged."""
    payload = {
        "type": "ingestion_status",
        "document_id": str(document_id),
        "phase": phase,
    }
    if chapter_id is not None:
        payload["chapter_id"] = str(chapter_id)
    if title is not None:
        payload["title"] = title
    if batch is not None:
        payload["batch"] = batch
    if total_batches is not None:
        payload["total_batches"] = total_batches
    if error is not None:
        payload["error"] = error
    return payload


def push_ingestion_status(user_id, document_id, phase, **extra) -> dict:
    """Build and broadcast an ingestion_status event to the user's group.
    Never raises — a telemetry failure must not break ingestion."""
    payload = build_ingestion_status(document_id, phase, **extra)
    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"user_{user_id}",
            {"type": "send_notification", "data": payload},
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("push_ingestion_status failed (%s): %s", phase, e)
    return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe manage.py test accounts.IngestionStatusBuilderTests -v 2`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add accounts/realtime.py accounts/tests.py
git commit -m "feat(realtime): add ingestion_status payload builder and push helper"
```

---

## Task 3: Emit phases from `create_chapter_from_document`

**Files:**
- Modify: `accounts/tasks.py` (function `create_chapter_from_document`, ~lines 131-209)

**Interfaces:**
- Consumes: `push_ingestion_status`, `PHASE_READING`, `PHASE_NAMING`, `PHASE_FAILED` from `accounts.realtime`.
- Produces: `reading` event at extraction start; `naming` event (with `chapter_id`, `title`) after chapter creation; `failed` event (with `error`) in the except block.

- [ ] **Step 1: Add the import**

At the top of `accounts/tasks.py`, alongside the existing `from .ai_clients import get_pinecone_index`:

```python
from .realtime import (
    push_ingestion_status,
    PHASE_READING, PHASE_NAMING, PHASE_CHUNKING,
    PHASE_EMBEDDING, PHASE_STORING, PHASE_READY, PHASE_FAILED,
)
```

- [ ] **Step 2: Emit `reading` after status is set to PROCESSING**

In `create_chapter_from_document`, immediately after `doc.save(update_fields=['status'])` (the PROCESSING save near line 141), add:

```python
        push_ingestion_status(doc.user.id, doc.id, PHASE_READING)
```

- [ ] **Step 3: Emit `naming` after the chapter is created**

Right after `logger.info(f"[{document_id}] Document updated with new chapter and title.")` (near line 179), add:

```python
        push_ingestion_status(
            doc.user.id, doc.id, PHASE_NAMING,
            chapter_id=new_chapter.id, title=ai_generated_title,
        )
```

- [ ] **Step 4: Emit `failed` in the except block**

In the `except Exception as e:` block of `create_chapter_from_document` (near line 196), after the `doc.save(update_fields=['status', 'error_message'])` inside the inner try, add a push. Replace the inner try/except so it reads:

```python
        try:
            doc = Document.objects.get(id=document_id)
            doc.status = Document.STATUS_FAILED
            doc.error_message = str(e)
            doc.save(update_fields=['status', 'error_message'])
            push_ingestion_status(
                doc.user.id, doc.id, PHASE_FAILED, error=str(e))
        except Exception:
            pass  # If doc doesn't exist, we can't update it
```

- [ ] **Step 5: Manual verification (worker + browser console)**

Start Redis, TEI, the Celery worker (`./venv/Scripts/celery -A core worker -l info -P solo`), and the frontend. Open the dashboard, open browser DevTools console, upload a PDF. Expected console `WS EVENT:` logs include `{type:"ingestion_status", phase:"reading", ...}` then `{phase:"naming", chapter_id:..., title:...}`.

- [ ] **Step 6: Commit**

```bash
git add accounts/tasks.py
git commit -m "feat(ingest): emit reading/naming/failed phases from create_chapter_from_document"
```

---

## Task 4: Emit phases from `process_document_ingestion`

**Files:**
- Modify: `accounts/tasks.py` (function `process_document_ingestion`, ~lines 245-358)

**Interfaces:**
- Consumes: the same `realtime` symbols imported in Task 3.
- Produces: `chunking` after chunks computed; `embedding` per batch with `batch`/`total_batches`; `storing` before the final chapter save; `ready` (with `chapter_id`, `title`) on success; `failed` (with `error`) on exception.

- [ ] **Step 1: Emit `chunking` after chunks are computed**

After `text_chunks = text_chunks[:MAX_CHUNKS_PER_DOCUMENT]` and its log line (near line 266-268), add:

```python
        total_batches = max(1, (len(text_chunks) + BATCH - 1) // BATCH)
        push_ingestion_status(doc.user.id, doc.id, PHASE_CHUNKING,
                              total_batches=total_batches)
```

Note: `BATCH = 16` is defined a few lines below at ~line 273. **Move the `BATCH = 16` assignment to just above this new block** so `total_batches` can use it. The moved line is:

```python
        BATCH = 16
```

- [ ] **Step 2: Emit `embedding` inside the batch loop**

Inside `for i in range(0, len(text_chunks), BATCH):`, immediately after the `logger.info(f"[{correlation_id}] Embedding batch {i//BATCH + 1}/...` line (near line 295), add:

```python
            push_ingestion_status(
                doc.user.id, doc.id, PHASE_EMBEDDING,
                batch=i // BATCH + 1, total_batches=total_batches,
            )
```

- [ ] **Step 3: Emit `storing` before the success save**

After the batch `for` loop ends and before `doc.status = Document.STATUS_COMPLETED` (near line 331), add:

```python
        push_ingestion_status(doc.user.id, doc.id, PHASE_STORING)
```

- [ ] **Step 4: Emit `ready` on success**

After `doc.save(update_fields=['status', 'error_message'])` for COMPLETED (near line 334) and its success log, add:

```python
        push_ingestion_status(
            doc.user.id, doc.id, PHASE_READY,
            chapter_id=(doc.chapter.id if doc.chapter else None),
            title=(doc.chapter.name if doc.chapter else doc.title),
        )
```

- [ ] **Step 5: Emit `failed` in the except block**

In the `except Exception as e:` of `process_document_ingestion` (near line 345), after `doc.save(update_fields=['status', 'error_message'])`, add:

```python
        push_ingestion_status(doc.user.id, doc.id, PHASE_FAILED, error=str(e))
```

- [ ] **Step 6: Manual verification**

Upload a PDF with the worker + frontend running. Expected browser console sequence: `reading → naming → chunking → embedding (batch 1..N) → storing → ready`. The `ready` event carries `chapter_id` and `title`.

- [ ] **Step 7: Commit**

```bash
git add accounts/tasks.py
git commit -m "feat(ingest): emit chunking/embedding/storing/ready/failed phases from ingestion task"
```

---

## Task 5: Retry endpoint

**Files:**
- Modify: `accounts/views.py` (add `DocumentRetryView` after `DocumentDetailView`, ~line 227)
- Modify: `accounts/urls.py` (add route near line 42)
- Test: `accounts/tests.py`

**Interfaces:**
- Produces: `POST /auth/documents/<uuid:id>/retry/` → 202 `{"status": "requeued", "document_id": "..."}`; resets `status=PENDING`, clears `error_message`, re-enqueues `process_document_ingestion` if the doc has a chapter, else `create_chapter_from_document`. 404 for a doc the user does not own.

- [ ] **Step 1: Write the failing test**

Add to `accounts/tests.py`:

```python
from unittest.mock import patch
from accounts.models import Document


class DocumentRetryTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        User = get_user_model()
        self.user = User.objects.create_user(
            email="r@b.com", password="pw12345", name="R")
        self.client.force_authenticate(self.user)

    def _make_failed_doc(self, chapter=None):
        return Document.objects.create(
            user=self.user, chapter=chapter, title="t",
            file="uploads/x.txt", file_type="txt",
            status=Document.STATUS_FAILED, error_message="boom")

    @patch("accounts.views.create_chapter_from_document")
    def test_retry_no_chapter_requeues_create(self, mock_task):
        doc = self._make_failed_doc()
        res = self.client.post(f"/auth/documents/{doc.id}/retry/")
        self.assertEqual(res.status_code, 202, res.content)
        doc.refresh_from_db()
        self.assertEqual(doc.status, Document.STATUS_PENDING)
        self.assertIsNone(doc.error_message)
        mock_task.delay.assert_called_once_with(str(doc.id))

    def test_retry_foreign_document_404(self):
        User = get_user_model()
        other = User.objects.create_user(
            email="o@b.com", password="pw12345", name="O")
        doc = Document.objects.create(
            user=other, title="t", file="uploads/x.txt",
            file_type="txt", status=Document.STATUS_FAILED)
        res = self.client.post(f"/auth/documents/{doc.id}/retry/")
        self.assertEqual(res.status_code, 404)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe manage.py test accounts.DocumentRetryTests -v 2`
Expected: FAIL — route/view does not exist (404 for the first test too, or NoReverseMatch).

- [ ] **Step 3: Add the view**

In `accounts/views.py`, after `DocumentDetailView` (near line 227). Confirm these imports exist at the top of the file (add any missing): `from rest_framework.views import APIView`, `from rest_framework.response import Response`, `from rest_framework import status`, `from rest_framework.permissions import IsAuthenticated`, `from django.shortcuts import get_object_or_404`, and the task imports `create_chapter_from_document`, `process_document_ingestion`.

```python
class DocumentRetryView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, id):
        doc = get_object_or_404(Document, id=id, user=request.user)
        doc.status = Document.STATUS_PENDING
        doc.error_message = None
        doc.save(update_fields=["status", "error_message"])
        if doc.chapter:
            process_document_ingestion.delay(str(doc.id))
        else:
            create_chapter_from_document.delay(str(doc.id))
        return Response(
            {"status": "requeued", "document_id": str(doc.id)},
            status=status.HTTP_202_ACCEPTED,
        )
```

- [ ] **Step 4: Add the route**

In `accounts/urls.py`, import `DocumentRetryView` in the existing import block, and add after the `document-detail` line (line 42):

```python
    path('documents/<uuid:id>/retry/', DocumentRetryView.as_view(), name='document-retry'),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `./venv/Scripts/python.exe manage.py test accounts.DocumentRetryTests -v 2`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add accounts/views.py accounts/urls.py accounts/tests.py
git commit -m "feat(api): add document retry endpoint that re-enqueues ingestion"
```

---

## Task 6: Frontend ingestion types + pure reducer

**Files:**
- Create: `src/features/notebook/ingestion.ts`

**Interfaces:**
- Produces:
  - `type IngestionPhase = 'uploading'|'reading'|'naming'|'chunking'|'embedding'|'storing'|'ready'|'failed'`
  - `interface IngestionStatusEvent { type: 'ingestion_status'; document_id: string; phase: Exclude<IngestionPhase,'uploading'>; chapter_id?: string; title?: string; batch?: number; total_batches?: number; error?: string; }`
  - `interface Ingestion { documentId: string; filename: string; phase: IngestionPhase; uploadPercent: number; chapterId: string|null; title: string|null; batch: number|null; totalBatches: number|null; error: string|null; }`
  - `const PHASE_ORDER: IngestionPhase[]`
  - `function applyIngestionEvent(prev: Ingestion|undefined, evt: IngestionStatusEvent): Ingestion`
  - `function makeOptimisticIngestion(documentId: string, filename: string): Ingestion`

- [ ] **Step 1: Create the module**

```typescript
// Ingestion phase state, reduced live from WebSocket ingestion_status events.

export type IngestionPhase =
  | 'uploading' | 'reading' | 'naming' | 'chunking'
  | 'embedding' | 'storing' | 'ready' | 'failed';

// Ordered for rendering the checklist and preventing out-of-order regressions.
export const PHASE_ORDER: IngestionPhase[] = [
  'uploading', 'reading', 'naming', 'chunking',
  'embedding', 'storing', 'ready',
];

export const PHASE_LABELS: Record<IngestionPhase, string> = {
  uploading: 'Uploading file',
  reading: 'Reading document',
  naming: 'Naming chapter',
  chunking: 'Chunking text',
  embedding: 'Embedding',
  storing: 'Storing in Pinecone',
  ready: 'Ready',
  failed: 'Failed',
};

export interface IngestionStatusEvent {
  type: 'ingestion_status';
  document_id: string;
  phase: Exclude<IngestionPhase, 'uploading'>;
  chapter_id?: string;
  title?: string;
  batch?: number;
  total_batches?: number;
  error?: string;
}

export interface Ingestion {
  documentId: string;
  filename: string;
  phase: IngestionPhase;
  uploadPercent: number;   // 0-100, meaningful during 'uploading'
  chapterId: string | null;
  title: string | null;
  batch: number | null;
  totalBatches: number | null;
  error: string | null;
}

export function makeOptimisticIngestion(
  documentId: string,
  filename: string,
): Ingestion {
  return {
    documentId,
    filename,
    phase: 'uploading',
    uploadPercent: 0,
    chapterId: null,
    title: null,
    batch: null,
    totalBatches: null,
    error: null,
  };
}

function phaseRank(p: IngestionPhase): number {
  const i = PHASE_ORDER.indexOf(p);
  return i === -1 ? 99 : i; // 'failed' sorts last / always wins
}

export function applyIngestionEvent(
  prev: Ingestion | undefined,
  evt: IngestionStatusEvent,
): Ingestion {
  const base: Ingestion =
    prev ?? makeOptimisticIngestion(evt.document_id, 'Document');

  // Never regress to an earlier phase from a late/duplicate event,
  // except 'failed' which always takes over.
  const nextPhase: IngestionPhase =
    evt.phase === 'failed' || phaseRank(evt.phase) >= phaseRank(base.phase)
      ? evt.phase
      : base.phase;

  return {
    ...base,
    phase: nextPhase,
    uploadPercent: nextPhase === 'uploading' ? base.uploadPercent : 100,
    chapterId: evt.chapter_id ?? base.chapterId,
    title: evt.title ?? base.title,
    batch: evt.batch ?? (nextPhase === 'embedding' ? base.batch : null),
    totalBatches: evt.total_batches ?? base.totalBatches,
    error: evt.error ?? (nextPhase === 'failed' ? base.error : null),
  };
}
```

- [ ] **Step 2: Typecheck**

Run from `frontend/RAG_tutor_frontend`: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Sanity-check the reducer (throwaway, not committed)**

Run: `npx tsx -e "import {applyIngestionEvent,makeOptimisticIngestion} from './src/features/notebook/ingestion'; let s=makeOptimisticIngestion('d','f.pdf'); s=applyIngestionEvent(s,{type:'ingestion_status',document_id:'d',phase:'embedding',batch:2,total_batches:5}); console.log(s.phase===' embedding'.trim(), s.batch===2, s.uploadPercent===100); s=applyIngestionEvent(s,{type:'ingestion_status',document_id:'d',phase:'reading'}); console.log('no regress:', s.phase==='embedding');"`
Expected: prints `true true true` then `no regress: true`. (If `tsx` is unavailable, skip — the typecheck plus manual browser run in later tasks covers it.)

- [ ] **Step 4: Commit**

```bash
git add src/features/notebook/ingestion.ts
git commit -m "feat(notebook): add ingestion types and pure phase reducer"
```

---

## Task 7: API client — upload progress + retry

**Files:**
- Modify: `src/features/notebook/notebook.api.ts` (near line 71-74)

**Interfaces:**
- Consumes: existing `api` axios instance.
- Produces:
  - `uploadDocument(formData: FormData, onUploadProgress?: (percent: number) => void)` — reports 0-100 integer percent.
  - `retryDocument(id: string)` — `POST /auth/documents/${id}/retry/`.

- [ ] **Step 1: Replace `uploadDocument` and add `retryDocument`**

```typescript
  uploadDocument: (
    formData: FormData,
    onUploadProgress?: (percent: number) => void,
  ) =>
    api.post<DocumentDTO>("/auth/documents/", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (e) => {
        if (onUploadProgress && e.total) {
          onUploadProgress(Math.round((e.loaded / e.total) * 100));
        }
      },
    }),

  retryDocument: (id: string) =>
    api.post(`/auth/documents/${id}/retry/`),
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors. (If existing callers of `uploadDocument` break, they don't — the new param is optional.)

- [ ] **Step 3: Commit**

```bash
git add src/features/notebook/notebook.api.ts
git commit -m "feat(notebook-api): upload progress callback and retryDocument"
```

---

## Task 8: Store — ingestions map, actions, WS wiring, mount seed

**Files:**
- Modify: `src/lib/store/useNotebook.ts` (state near line 90-100; `initWebSocket` 100-134; add actions; interface 70-77)

**Interfaces:**
- Consumes: `Ingestion`, `IngestionStatusEvent`, `applyIngestionEvent`, `makeOptimisticIngestion` from `@/features/notebook/ingestion`; `notebookApi.fetchDocuments`, `notebookApi.retryDocument`.
- Produces on the store:
  - state `ingestions: Record<string, Ingestion>`
  - `startIngestion(documentId: string, filename: string): void`
  - `setUploadPercent(documentId: string, percent: number): void`
  - `dismissIngestion(documentId: string): void`
  - `retryIngestion(documentId: string): Promise<void>`
  - `seedInFlightIngestions(): Promise<void>` — GET documents, add PROCESSING/PENDING ones as coarse cards.
  - `onIngestionReady?: (chapterId: string) => void` — settable callback the UI uses to auto-open.

- [ ] **Step 1: Add imports** at the top of `useNotebook.ts`:

```typescript
import {
  Ingestion, IngestionStatusEvent,
  applyIngestionEvent, makeOptimisticIngestion,
} from "@/features/notebook/ingestion";
```

- [ ] **Step 2: Add state fields** in the store initializer (near line 90, next to `error: null`):

```typescript
      ingestions: {},
      onIngestionReady: undefined,
```

And in the store's TypeScript interface (near line 70), add:

```typescript
  ingestions: Record<string, Ingestion>;
  onIngestionReady?: (chapterId: string) => void;
  startIngestion: (documentId: string, filename: string) => void;
  setUploadPercent: (documentId: string, percent: number) => void;
  dismissIngestion: (documentId: string) => void;
  retryIngestion: (documentId: string) => Promise<void>;
  seedInFlightIngestions: () => Promise<void>;
```

- [ ] **Step 3: Replace the `onmessage` handler** in `initWebSocket` (lines 112-127) so it reduces ingestion events and still refreshes the notebook:

```typescript
        ws.onmessage = async (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log("WS EVENT:", data);

            if (data.type === "ingestion_status") {
              const evt = data as IngestionStatusEvent;
              set((state) => ({
                ingestions: {
                  ...state.ingestions,
                  [evt.document_id]: applyIngestionEvent(
                    state.ingestions[evt.document_id], evt),
                },
              }));

              if (evt.phase === "ready") {
                await get().fetchSubjects();
                const chapterId = evt.chapter_id ?? null;
                if (chapterId) {
                  get().setActiveChapter(chapterId);
                  get().onIngestionReady?.(chapterId);
                }
                // Remove the finished card shortly after the swap.
                setTimeout(() => get().dismissIngestion(evt.document_id), 1200);
              }
              return;
            }

            // Legacy coarse events still trigger a refresh.
            if (data.message === "notebook_updated" ||
                data.message === "document_ready") {
              await get().fetchSubjects();
            }
          } catch (e) {
            console.error("WS parse error", e);
          }
        };
```

- [ ] **Step 4: Add the actions** (place near the other CRUD actions, e.g. after `fetchSubjects`):

```typescript
      startIngestion: (documentId, filename) => {
        set((state) => ({
          ingestions: {
            ...state.ingestions,
            [documentId]: makeOptimisticIngestion(documentId, filename),
          },
        }));
      },

      setUploadPercent: (documentId, percent) => {
        set((state) => {
          const cur = state.ingestions[documentId];
          if (!cur) return {};
          return {
            ingestions: {
              ...state.ingestions,
              [documentId]: { ...cur, uploadPercent: percent },
            },
          };
        });
      },

      dismissIngestion: (documentId) => {
        set((state) => {
          const next = { ...state.ingestions };
          delete next[documentId];
          return { ingestions: next };
        });
      },

      retryIngestion: async (documentId) => {
        try {
          await notebookApi.retryDocument(documentId);
          set((state) => {
            const cur = state.ingestions[documentId];
            if (!cur) return {};
            return {
              ingestions: {
                ...state.ingestions,
                [documentId]: {
                  ...cur, phase: "reading", error: null, uploadPercent: 100,
                },
              },
            };
          });
        } catch (err) {
          console.error("retryIngestion failed:", err);
        }
      },

      seedInFlightIngestions: async () => {
        try {
          const docs = await notebookApi.fetchDocuments();
          set((state) => {
            const next = { ...state.ingestions };
            for (const d of docs) {
              if ((d.status === "PENDING" || d.status === "PROCESSING")
                  && !next[d.id]) {
                const seed = makeOptimisticIngestion(
                  d.id, d.title || "Document");
                // Coarse resume: show a generic processing phase until a
                // live WS event refines it.
                next[d.id] = { ...seed, phase: "reading", uploadPercent: 100 };
              }
            }
            return { ingestions: next };
          });
        } catch (err) {
          console.error("seedInFlightIngestions failed:", err);
        }
      },
```

> `notebookApi.fetchDocuments()` returns `DocumentDTO[]` with `id`, `status`, `title`. If the store's `notebookApi` import is not already present, add `import { notebookApi } from "@/features/notebook/notebook.api";` (match the existing import style used by `fetchSubjects`).

- [ ] **Step 5: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/lib/store/useNotebook.ts
git commit -m "feat(store): ingestions map, WS reduce, mount seed, retry/dismiss actions"
```

---

## Task 9: Center-panel phase checklist component

**Files:**
- Create: `src/components/dashboard/ContentPanel/IngestionProgress.tsx`

**Interfaces:**
- Consumes: `Ingestion`, `IngestionPhase`, `PHASE_ORDER`, `PHASE_LABELS` from `@/features/notebook/ingestion`; store actions `retryIngestion`, `dismissIngestion`.
- Produces: `export function IngestionProgress({ ingestion }: { ingestion: Ingestion })` — renders the checklist for a non-failed ingestion, or the error card (with Retry/Dismiss) when `ingestion.phase === 'failed'`.

- [ ] **Step 1: Create the component**

```tsx
"use client";

import { motion } from "framer-motion";
import { useNotebookStore } from "@/lib/store/useNotebook";
import {
  Ingestion, IngestionPhase, PHASE_ORDER, PHASE_LABELS,
} from "@/features/notebook/ingestion";

function rank(p: IngestionPhase) {
  const i = PHASE_ORDER.indexOf(p);
  return i === -1 ? 99 : i;
}

export function IngestionProgress({ ingestion }: { ingestion: Ingestion }) {
  const retryIngestion = useNotebookStore((s) => s.retryIngestion);
  const dismissIngestion = useNotebookStore((s) => s.dismissIngestion);

  if (ingestion.phase === "failed") {
    return (
      <div className="mx-auto max-w-md rounded-xl border border-destructive/40 bg-destructive/5 p-6">
        <p className="font-medium text-destructive">Ingestion failed</p>
        <p className="mt-1 text-sm text-muted-foreground break-words">
          {ingestion.error || "Something went wrong while processing this document."}
        </p>
        <div className="mt-4 flex gap-2">
          <button
            onClick={() => retryIngestion(ingestion.documentId)}
            className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground"
          >
            Retry
          </button>
          <button
            onClick={() => dismissIngestion(ingestion.documentId)}
            className="rounded-md border px-3 py-1.5 text-sm"
          >
            Dismiss
          </button>
        </div>
      </div>
    );
  }

  const current = rank(ingestion.phase);

  return (
    <div className="mx-auto max-w-md rounded-xl border bg-card p-6">
      <p className="mb-4 font-medium">Processing “{ingestion.filename}”</p>
      <ul className="space-y-2">
        {PHASE_ORDER.filter((p) => p !== "uploading" || true).map((phase) => {
          const done = rank(phase) < current || ingestion.phase === "ready";
          const active = phase === ingestion.phase;
          const label = PHASE_LABELS[phase];
          return (
            <li key={phase} className="flex items-center gap-3 text-sm">
              <span className="inline-flex h-5 w-5 items-center justify-center">
                {done ? (
                  <span className="text-primary">✓</span>
                ) : active ? (
                  <motion.span
                    className="text-primary"
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                  >
                    ⟳
                  </motion.span>
                ) : (
                  <span className="text-muted-foreground">∘</span>
                )}
              </span>
              <span className={active ? "text-foreground" : done ? "text-muted-foreground" : "text-muted-foreground/60"}>
                {label}
                {phase === "uploading" && active
                  ? ` ${ingestion.uploadPercent}%`
                  : ""}
                {phase === "embedding" && active && ingestion.batch
                  ? `  ·  batch ${ingestion.batch} of ${ingestion.totalBatches ?? "?"}`
                  : ""}
              </span>
            </li>
          );
        })}
      </ul>
      {ingestion.phase === "uploading" && (
        <div className="mt-4 h-1.5 w-full overflow-hidden rounded bg-muted">
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${ingestion.uploadPercent}%` }}
          />
        </div>
      )}
    </div>
  );
}
```

> Tailwind color tokens (`bg-card`, `text-primary`, `border-destructive`, etc.) already exist in `global.css` (Task context confirmed the design system). If any token is missing, substitute the nearest existing one rather than inventing new CSS.

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/dashboard/ContentPanel/IngestionProgress.tsx
git commit -m "feat(ui): ingestion phase checklist and failure card component"
```

---

## Task 10: Wire upload → optimistic entry + progress

**Files:**
- Modify: `src/components/dashboard/ContentPanel/AddSourceView.tsx` (`handleFileSelect`, lines 26-45)

**Interfaces:**
- Consumes: store `startIngestion`, `setUploadPercent`; `notebookApi.uploadDocument(formData, onProgress)` returning `{ data: DocumentDTO }`.
- Produces: on file select, an optimistic ingestion appears immediately keyed by a temporary id, rebound to the real `document.id` once the POST resolves.

- [ ] **Step 1: Update `handleFileSelect`**

Replace the body of `handleFileSelect` (keep the existing validation lines) so the try block reads:

```tsx
  const startIngestion = useNotebookStore((s) => s.startIngestion);
  const setUploadPercent = useNotebookStore((s) => s.setUploadPercent);
  const dismissIngestion = useNotebookStore((s) => s.dismissIngestion);
```

(Add those three selector hooks near the existing `fetchSubjects` selector at line 24.)

Then in `handleFileSelect`, after building `formData` and before the API call:

```tsx
    // Optimistic entry under a temp id so the UI never blanks out.
    const tempId = `temp-${Date.now()}`;
    startIngestion(tempId, file.name);

    try {
      const res = await notebookApi.uploadDocument(formData, (percent) => {
        setUploadPercent(tempId, percent);
      });
      const realId = res.data.id;
      // Rebind temp -> real document id so WS events (keyed by real id) match.
      const store = useNotebookStore.getState();
      const temp = store.ingestions[tempId];
      if (temp) {
        store.startIngestion(realId, file.name);
        store.setUploadPercent(realId, 100);
        store.dismissIngestion(tempId);
      }
    } catch (err) {
      console.error("Upload failed:", err);
      dismissIngestion(tempId);
      setError("Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
```

> Rebinding is necessary because the optimistic card is created before the server assigns the real `document_id`. WS `ingestion_status` events use the real id, so we migrate the temp entry to it as soon as the POST returns. There is a brief window where both could exist; `dismissIngestion(tempId)` closes it.

- [ ] **Step 2: Typecheck + lint**

Run: `npx tsc --noEmit && npm run lint`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/dashboard/ContentPanel/AddSourceView.tsx
git commit -m "feat(upload): optimistic ingestion card with live upload percentage"
```

---

## Task 11: Render progress in the content panel + auto-open on ready

**Files:**
- Modify: the ContentPanel host that shows `AddSourceView` (find via grep, likely `src/components/dashboard/ContentPanel/*.tsx` or `src/app/dashboard/page.tsx`).

**Interfaces:**
- Consumes: store `ingestions`, `seedInFlightIngestions`, `setActiveChapter`; `IngestionProgress`.
- Produces: when any ingestion is in-flight, the content panel renders `IngestionProgress` for the most recent one instead of (or above) the empty dropzone; auto-opens the chapter on ready (handled in Task 8's onmessage via `setActiveChapter`).

- [ ] **Step 1: Locate the host**

Run: `grep -rn "AddSourceView" src/`
Open the component that renders `<AddSourceView`. That is the content-panel host.

- [ ] **Step 2: Seed in-flight ingestions on mount**

In `src/app/dashboard/page.tsx`, in the existing `useEffect` that calls `initWebSocket()` (line 32-34), add a seed call:

```tsx
  useEffect(() => {
    useNotebookStore.getState().initWebSocket();
    useNotebookStore.getState().seedInFlightIngestions();
  }, []);
```

- [ ] **Step 3: Render `IngestionProgress` in the host**

In the content-panel host, read ingestions and render the newest in-flight one:

```tsx
  const ingestions = useNotebookStore((s) => s.ingestions);
  const inFlight = Object.values(ingestions);
  const active = inFlight[inFlight.length - 1];
```

Then, where the empty-state dropzone renders, show the progress card when `active` exists:

```tsx
  {active ? (
    <div className="flex h-full items-center justify-center p-6">
      <IngestionProgress ingestion={active} />
    </div>
  ) : (
    /* existing dropzone / AddSourceView JSX unchanged */
  )}
```

Add the import: `import { IngestionProgress } from "@/components/dashboard/ContentPanel/IngestionProgress";`

- [ ] **Step 4: Typecheck + lint**

Run: `npx tsc --noEmit && npm run lint`
Expected: no errors.

- [ ] **Step 5: Manual end-to-end verification**

With Redis, TEI, Celery worker, backend, and frontend all running: upload a PDF from the empty dashboard. Expected:
1. Immediately: center panel shows "Uploading X%" with a growing bar.
2. Then the checklist advances: Reading → Naming (row relabels toward the chapter) → Chunking → Embedding (batch n/N) → Storing → Ready.
3. On Ready: the app auto-opens the new chapter (chat/questions/flashcards view).
4. Reload mid-ingest: a coarse "Processing" card reappears and refines via WS.
5. Kill TEI mid-ingest to force a failure: error card with Retry/Dismiss appears; Retry re-runs.

- [ ] **Step 6: Commit**

```bash
git add src/app/dashboard/page.tsx src/components/dashboard/ContentPanel/*.tsx
git commit -m "feat(dashboard): render live ingestion progress and auto-open chapter on ready"
```

---

## Task 12: Sidebar processing rows

**Files:**
- Modify: `src/components/dashboard/SideBar/NotebookSidebar.tsx` (Alone Chapters section, ~lines 166-190)

**Interfaces:**
- Consumes: store `ingestions`.
- Produces: for each in-flight ingestion NOT yet resolved to a listed chapter, a compact non-clickable row ("⟳ Processing · <phase>") in the Alone Chapters section; it disappears when the real chapter appears after `ready`.

- [ ] **Step 1: Read ingestions in the sidebar**

Near the existing store destructure (line 26), add:

```tsx
  const ingestions = useNotebookStore((s) => s.ingestions);
  const processingRows = Object.values(ingestions).filter(
    (i) => i.phase !== "ready",
  );
```

- [ ] **Step 2: Render processing rows above the Alone Chapters list**

Inside the Alone Chapters section (after the header near line 171), add:

```tsx
  {processingRows.map((ing) => (
    <div
      key={ing.documentId}
      className="flex items-center gap-2 px-2 py-1 text-sm text-muted-foreground"
    >
      <span className="animate-spin">⟳</span>
      <span className="truncate">
        {ing.title || ing.filename}
        {ing.phase === "failed" ? " · failed" : ` · ${ing.phase}`}
      </span>
    </div>
  ))}
```

> `animate-spin` is a Tailwind default utility; if unavailable, reuse the `olive-pulse` class present in `global.css`.

- [ ] **Step 3: Typecheck + lint**

Run: `npx tsc --noEmit && npm run lint`
Expected: no errors.

- [ ] **Step 4: Manual verification**

Upload a doc; confirm a "⟳ Processing · reading/chunking/…" row shows in the sidebar and is replaced by the real clickable chapter after Ready.

- [ ] **Step 5: Commit**

```bash
git add src/components/dashboard/SideBar/NotebookSidebar.tsx
git commit -m "feat(sidebar): live processing rows that resolve into real chapters"
```

---

## Self-Review Notes

- **Spec coverage:** Q1 named phases → Tasks 3,4,9. Q2 center+sidebar → Tasks 9,11,12. Q3 reload-resilient → Task 8 `seedInFlightIngestions`. Q4 failure+retry → Tasks 5,9,8. Q5 upload %/batch → Tasks 7,10,9. Q6 auto-open → Task 8 onmessage + Task 11. Q7 generic mechanism → keyed by document_id throughout; new-chapter flow verified in Task 11 step 5.
- **Root-cause bug** (payload dropped) → Task 1.
- **Type consistency:** `applyIngestionEvent`, `makeOptimisticIngestion`, `Ingestion`, `IngestionStatusEvent` defined in Task 6 and consumed unchanged in Tasks 8,9,10,12. Store actions named identically across Tasks 8/9/10/12. `push_ingestion_status` / phase constants defined in Task 2, consumed in Tasks 3,4.
- **Back-compat:** Task 1 keeps legacy `{"message": ...}` events working; Task 8 keeps handling `notebook_updated`/`document_ready`.
```
