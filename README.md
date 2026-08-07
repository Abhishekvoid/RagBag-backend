# RagBag — RAG Knowledge Assistant (Backend)

Production-oriented Django backend for an AI study workspace: upload books and study
material, then read, question, and revise them with a retrieval-grounded LLM.

> **This repository is the backend.** The Next.js client lives in
> [RagBag-frontend](https://github.com/Abhishekvoid/RagBag-frontend).

Documents are ingested page by page (PDF text layer → optional vision reconstruction →
OCR fallback), chunked, embedded through a self-hosted TEI service, and indexed into
Pinecone. Queries are contextualized against chat history, routed by intent, expanded
into multiple search queries, retrieved in parallel, reranked by a self-hosted
cross-encoder, and answered by Llama 3 on Groq with source attribution.

---

## ✨ What it does

| Feature | How it actually works |
|---|---|
| 📄 Page-aware ingestion | Each PDF page is rendered to an image and stored; text comes from the page's own text layer, a vision model when that layer is unusable, or Tesseract as a last resort |
| 🔎 Vision reconstruction | Pages the text layer can't be trusted for are rebuilt into clean Markdown by a vision model, with unreadable words marked inline as `[?word]` and the original scan kept alongside for verification |
| 🧠 Retrieval pipeline | History-aware query rewriting → intent routing → 3-way query expansion → parallel vector search → keyword boost → dedupe → cross-encoder rerank |
| 💬 Grounded answers | Llama 3 (Groq) answers from retrieved context and returns source chips plus suggested follow-up questions |
| 🎓 Study automation | Flashcards, practice questions, note synthesis, notes → flashcards, and passage explanation, all generated per chapter |
| 📝 Smart notes | Chapter-scoped notes and a scratch pad that feed back into generation |
| ⚡ Live ingestion status | WebSocket updates through eight named phases — `reading → naming → page → chunking → embedding → storing → ready / failed` — with per-batch progress |
| 🛡️ Graceful degradation | Redis-backed circuit breakers, bounded concurrency slots, and retry policies in front of every external service; reranking and vision are enhancements, never hard dependencies |
| 🔐 Multi-tenancy | UUID keys throughout, JWT auth, and every vector query filtered by `user_id` + `chapter_id` |

---

## 🏗️ System architecture

<img width="720" height="960" alt="robot_system_design excalidraw" src="https://github.com/user-attachments/assets/709964ff-d243-488d-880a-4effe4d5a3ea" />

### Ingestion

```text
POST /auth/documents/  (multipart)
   └─ file → Supabase S3, Document row → PENDING
   └─ Celery: create_chapter_from_document  (no chapter yet — Groq names one)
             process_document_for_existing_chapter  (chapter known)
        ├─ per page: render PNG → store → text layer? vision? OCR?
        │            → DocumentPage(reconstructed_md, image_url, text_source)
        ├─ canonical text = pages joined, [?] markers stripped
        └─ process_document_ingestion
             ├─ token chunking (tiktoken cl100k_base, 200 / 50 overlap, cap 1000)
             ├─ embed in batches of 16 via TEI  → 384-dim
             ├─ upsert to Pinecone with {text, document_id, user_id, chapter_id, file_type}
             └─ status COMPLETED → WebSocket "ready"
```

Every phase pushes a status frame to the user's WebSocket group, so the client shows real
progress instead of a spinner.

### Retrieval and generation

```text
POST /auth/rag-chat/   { chapter, text }
   └─ readiness gate: chapter's latest document must be COMPLETED (409 while processing)
   └─ RagPipeline.run
        ├─ greeting? → canned reply, no retrieval
        ├─ contextualize_query   — rewrite follow-ups into standalone questions (last 5 turns)
        ├─ route_query           — summary | ambiguous | question
        └─ handle_rag_search
             ├─ query expansion  — Groq JSON, 3 strategic queries
             ├─ embed_texts      — TEI, one vector per query
             ├─ vector search    — queries fanned out concurrently, filtered by user + chapter
             ├─ keyword boost + dedupe by chunk id
             ├─ rerank           — TEI /rerank over the top ≤20 candidates
             ├─ retrieval evaluation (relevance heuristics, recorded)
             └─ answer           — Groq, strict Markdown prompt → spacing normalizer
   └─ persist ChatMessage(text, citations, suggestions) → return answer + sources + followups
```

Both the strict `user + chapter` filter and a `user`-only fallback are attempted, so a
mis-tagged chapter degrades to broader retrieval rather than an empty answer.

---

## 🗃️ Domain model

All primary keys are UUIDs.

```text
CustomUserModel (email as username)
 └─ Subject
      └─ Chapter                 ← the unit everything is scoped to
           ├─ Document           → status: PENDING | PROCESSING | COMPLETED | FAILED
           │    └─ DocumentPage  → page_number, image_url, reconstructed_md, text_source
           ├─ Note               → chapter-scoped notes + scratch pad
           ├─ GenerateQuestion
           └─ GenerateFlashCards → known / need_review
 └─ ChatSession
      └─ ChatMessage             → sender, text, citations, suggestions, tokens, error
```

Chapters may have no subject; the API surfaces those under a synthetic "Uncategorized"
section rather than forcing a taxonomy on the user.

---

## 🔌 API surface

All routes are mounted under `auth/`. JWT required unless noted.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `register/` · `oauth-signin/` | Email registration · Google sign-in |
| `GET` | `me/` | Current user |
| `GET POST` | `subjects/` · `chapters/` | List / create |
| `GET PATCH DELETE` | `subjects/<id>/` · `chapters/<id>/` | Detail |
| `GET POST` | `documents/` | List · multipart upload (dispatches ingestion) |
| `GET PATCH DELETE` | `documents/<id>/` | Detail |
| `GET` | `documents/<id>/pages/` · `documents/<id>/content/` | Reconstructed pages · canonical text |
| `POST` | `documents/<id>/rescan/` · `documents/<id>/retry/` | Re-run vision pass · retry failed ingestion |
| `POST` | `rag-chat/` | **Main chat endpoint** |
| `GET` | `chapters/<id>/messages/` | Paginated history |
| `POST GET` | `chapters/<id>/generate-questions/` · `questions/` | Generate · list |
| `POST GET` | `chapters/<id>/generate-flashcards/` · `flashcards/` | Generate · list |
| `GET PATCH DELETE` | `flashcards/<id>/` | Known / needs-review toggles |
| `GET POST` | `chapters/<id>/notes/` | Notes |
| `GET PATCH DELETE` | `notes/<id>/` | Note detail |
| `GET PUT` | `chapters/<id>/scratch/` | Scratch pad |
| `POST` | `chapters/<id>/explain/` | Explain a highlighted passage |
| `POST` | `chapters/<id>/notes-to-flashcards/` · `synthesize-notes/` | Notes → study material |
| `WS` | `ws/notifications/?token=<jwt>` | Ingestion status stream |
| `GET` | `/ping/` | Health check (unauthenticated) |

Throttled at 10/hour anonymous and 100/hour authenticated. Access tokens live 15 minutes,
refresh tokens 7 days. Uploads are capped at 50 MB and restricted by extension.

---

## 🛡️ Resilience layer

Every outbound LLM call goes through one front door, `utils/llm_gateway.ask_llm`, which
composes four independent protections:

```python
# utils/llm_gateway.py
async def ask_llm(groq_client, messages, *, model, json_mode=False, **kwargs):
    if not llm_circuit_breaker.allow_request():      # 1. Redis circuit breaker
        raise LLMUnavailable("LLM circuit is open")
    try:
        async with llm_slot_manager.slot():          # 2. bounded concurrency + queue
            response = await _call_llm_with_retry(   # 3. tenacity, exponential jitter
                groq_client, messages, model=model, json_mode=json_mode, **kwargs
            )
    except SystemOverload:                           # 4. shed load, don't pile up
        raise LLMUnavailable("System is overloaded")
    llm_circuit_breaker.record_success()
    return response
```

- **Circuit breaker** (`utils/circuit_breaker.py`) — state lives in Redis, so the web
  process and every Celery worker share one view of whether a dependency is down.
  Opens after 3 failures, half-opens after a 60s cooldown.
- **Slot manager** (`utils/llm_load_control.py`) — a semaphore plus a bounded waiting
  queue. LLM calls get 20 concurrent / 100 queued, embeddings 3 / 5. Overflow raises
  `SystemOverload` immediately instead of letting latency grow without bound.
- **Retries** — tenacity with exponential jitter: 5 attempts for embeddings, 3 for LLM
  and rerank calls, and only for genuinely transient error classes.
- **Degradation over failure** — every caller has a defined fallback. Query
  contextualization falls back to the raw query, routing defaults to `question`,
  reranking returns `None` and the pipeline keeps the vector + keyword ordering, and
  vision failures fall through to the text layer. A dead rerank service slows answers
  down; it does not break them.

Instrumentation lives in `utils/metrics/`: a rolling-window latency tracker (per stage,
with p95/p99), a token/cost accumulator, and a retrieval relevance evaluator. Pipeline
stages are wrapped in `latency_tracker.track_async(...)` and reported through structured
logs with a per-request correlation id.

---

## 🚀 Running locally

### Prerequisites

- Python 3.13
- Redis (Celery broker + Channels layer)
- A TEI embedding container (`BAAI/bge-small-en-v1.5`, 384-dim)
- Pinecone account (serverless index is created automatically on first use)
- Supabase project — Postgres + an S3-compatible storage bucket
- Groq API key
- Optional: Tesseract + Poppler for the OCR fallback, Ollama for vision reconstruction

### Setup

```bash
git clone https://github.com/Abhishekvoid/RagBag-backend.git
cd RagBag-backend

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

touch .env                       # fill in the keys listed under Configuration
python manage.py migrate
```

### Supporting services

```bash
# Embeddings — 384-dim, must match EMBEDDING_DIM
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id BAAI/bge-small-en-v1.5

# Reranker (optional — the pipeline degrades cleanly without it)
docker compose up -d             # BAAI/bge-reranker-base on :8081

# Vision reconstruction (optional)
ollama pull minicpm-v && ollama serve
```

### Run

```bash
# ASGI server — required for WebSockets
uvicorn core.asgi:application --reload --port 8000

# Worker — uploads will never finish processing without this
celery -A core worker -l info
```

---

## 🔧 Configuration

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Required — startup fails without it |
| `DEBUG` · `DJANGO_ALLOWED_HOSTS` | `False` in production; comma-separated hosts |
| `REDIS_URL` | Celery broker, result backend, Channels layer, circuit-breaker state |
| `SUPABASE_DB_{HOST,PORT,NAME,USER,PASSWORD}` | Postgres connection (`sslmode=require`) |
| `SUPABASE_PROJECT_ID` · `AWS_*` | S3-compatible document storage |
| `PINECONE_API_KEY` | Vector store |
| `PINECONE_INDEX` · `PINECONE_CLOUD` · `PINECONE_REGION` | Defaults: `studywise-documents`, `aws`, `us-east-1` |
| `EMBEDDING_DIM` | `384` — enforced against every TEI response |
| `GROQ_API_KEY` | LLM generation |
| `TEI_URL` · `TEI_TIMEOUT` | Embedding service, default `http://localhost:8080/embed` |
| `RERANK_URL` · `RERANK_TIMEOUT` | Rerank service, default `http://localhost:8081/rerank` |
| `VISION_ENABLED` · `VISION_MODEL` · `VISION_BASE_URL` · `VISION_API_KEY` · `VISION_MAX_PAGES` | Vision pass; off by default, defaults to Ollama + `minicpm-v` |
| `GOOGLE_OAUTH_CLIENT_ID` · `GOOGLE_OAUTH_CLIENT_SECRET` | Google sign-in |

The dev↔prod switch is values only, never code: the same keys point at localhost
containers locally and hosted endpoints in production.

---

## 🧪 Tests

```bash
python manage.py test accounts
```

Covers the page model and ordering constraints, the page pipeline, vision reconstruction
and its fallbacks, the metrics utilities, and the API views.

---

## 📁 Layout

```text
RagBag-backend/
├── core/                     # settings, ASGI + Channels routing, Celery app, URLs
├── accounts/                 # the application
│   ├── models.py             # Subject / Chapter / Document / DocumentPage / Note / chat
│   ├── views.py              # DRF API surface
│   ├── serializers.py        # validation, upload limits
│   ├── rag_pipeline.py       # retrieval + generation orchestration
│   ├── rag_service.py        # Pinecone queries, embedding helpers
│   ├── page_pipeline.py      # PDF → per-page canonical Markdown
│   ├── vision_ocr.py         # vision reconstruction, engine-agnostic
│   ├── tasks.py              # Celery ingestion
│   ├── realtime.py           # WebSocket status phases
│   ├── consumers.py          # Channels consumer
│   ├── middleware.py         # JWT auth for WebSocket connections
│   └── tests/
├── utils/
│   ├── llm_gateway.py        # ask_llm — breaker + slots + retry + metrics
│   ├── llm_load_control.py   # SlotManager, SystemOverload
│   ├── circuit_breaker.py    # Redis-backed, shared across processes
│   ├── tei_embedding.py      # embedding client
│   ├── tei_rerank.py         # rerank client (fails soft)
│   ├── formatting.py         # Markdown spacing normalizer
│   └── metrics/              # latency (p95/p99), cost, retrieval relevance
├── docker-compose.yml        # reranker service
└── requirements.txt          # runtime deps (no torch — ML runs in TEI containers)
```

`requirements_ml.txt` holds the local-only ML extras. The deployed image ships no
`torch` / `transformers` / `sentence-transformers` at all: embeddings and reranking are
separate TEI services reached over HTTP, which keeps the app image small and lets the
model tier scale on its own.

---

## ⚡ Performance notes

The pipeline is built so the expensive stages overlap rather than queue:

- Query expansion produces three queries that are embedded together and searched
  **concurrently**, not sequentially.
- Embedding is batched (16 chunks per request) during ingestion.
- Deduplication happens **before** reranking, and the rerank window is capped at 20
  candidates, so the cross-encoder never sees the full result set.
- Chunks are capped at 1000 per document, bounding worst-case ingestion cost.
- Generation dominates end-to-end latency; retrieval is a small fraction of it.

Per-stage timings are recorded live by `StageLatencyTracker` (count / avg / p95 / p99
over a rolling window) and emitted in structured logs — measure against your own
deployment rather than trusting a table.

---

## 🚢 Deployment

Designed to run as two processes from one image, with all state external:

```bash
# Web (HTTP + WebSocket)
gunicorn core.asgi:application -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:$PORT --workers 2 --timeout 120

# Worker
celery -A core worker -l info --concurrency 2
```

Postgres (Supabase), Redis (any managed provider), Pinecone, and the TEI services are all
addressed by environment variable, so the same image runs on Render, Railway, Fly, or
plain Docker. Run `python manage.py migrate` as a pre-deploy step. `DEBUG=False` turns on
HTTPS redirect, HSTS, and secure cookies. The OCR fallback needs `tesseract-ocr` and
`poppler-utils` at the OS level, so deploy from a Dockerfile rather than a bare Python
runtime.

---

## 🗺️ Roadmap

Honest list of what is not there yet:

| Gap | Plan |
|---|---|
| Answers are returned as one response | Token streaming over the existing WebSocket |
| Page numbers on citations are approximate (chunk text matched back to a page) | Carry page number through chunking directly |
| `[?word]` markers are read-only | Accept user corrections and re-embed the page |
| Summary intent returns a placeholder | Map-reduce summarization over chapter pages |
| No retrieval cache | Redis-backed semantic cache on the expansion + search stages |
| Single worker, single queue | Split ingestion and generation queues, scale independently |
| Cost tracking priced at zero | Pricing table keys don't match the deployed model id |
| Leftover Qdrant-era scratch files in the repo root | Clean up |

---

## 🛠️ Stack

**Core** — Django 5.2, Django REST Framework, Channels (ASGI), Celery, Redis,
PostgreSQL, SimpleJWT + djoser + allauth, Uvicorn / Gunicorn.

**AI** — Pinecone (serverless, 384-dim, cosine), TEI with `BAAI/bge-small-en-v1.5`
(embeddings) and `BAAI/bge-reranker-base` (reranking), Groq `llama-3.1-8b-instant`,
MiniCPM-V via Ollama (vision), PyMuPDF / PyPDF2 / pdf2image / Tesseract (extraction),
tiktoken (chunking).

**Infra** — Supabase Postgres + S3-compatible storage, Docker, structured JSON logging.

---

## 📝 License

MIT.

---

## 📞 Contact

**Abhishek Rajput**

- 📧 `abhishek.rajput7202@gmail.com`
- 💼 [linkedin.com/in/abhishek-rajput-4ba60221a](https://linkedin.com/in/abhishek-rajput-4ba60221a)
- 💻 [github.com/Abhishekvoid](https://github.com/Abhishekvoid)

Open to backend engineering, AI infrastructure, distributed systems, and platform roles.
