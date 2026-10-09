# Deploying the RAG-tutor backend

The app is 12-factor: **the same code runs locally and in production — only env vars
differ.** Locally, `.env` supplies `localhost` URLs. In production you set the same
keys (with hosted URLs) in the platform dashboard. Nothing in code changes.

## Architecture (what runs where)

| Component | Local dev | Production |
|---|---|---|
| Web (Django ASGI) | `uvicorn ... --reload` | Docker web service (gunicorn + uvicorn worker) |
| Celery worker | `celery -A core worker -P solo` | Docker **worker** service (paid — see below) |
| Redis (broker + Channels) | `localhost:6379` | Upstash / managed Redis (`REDIS_URL`) |
| Postgres | Supabase | Supabase (same) |
| Object storage | Supabase S3 | Supabase S3 (same) |
| Embeddings (TEI) | local container `:8080` | hosted TEI endpoint (`TEI_URL`) |
| Reranker (TEI) | local container `:8081` | hosted TEI endpoint (`RERANK_URL`, optional) |
| Vision OCR (Ollama) | local `:11434` | **off** (`VISION_ENABLED=false`) |

## Files that make it deployable

- `Dockerfile` — Python 3.13 + poppler-utils + tesseract-ocr (needed by pdf2image/pytesseract).
- `.dockerignore` — keeps venvs/secrets out of the image.
- `render.yaml` (repo root) — Blueprint: web + worker services sharing one env group.
- `.env.example` — every env var, documented, secret-free.

Verified: image builds, `collectstatic` runs in-image, container boots and `/ping/` → 200.

## Before you go live — MUST DO

1. **Rotate every secret** currently in `.env` (they were committed to your working
   tree): Supabase DB password, Supabase/AWS S3 keys, Groq, Pinecone, Google OAuth.
   Set the new values as platform env vars, never in the repo.
2. **Provision Redis** — e.g. Upstash free tier — and set `REDIS_URL`.
3. **Host the TEI embedding service** and set `TEI_URL`. This is required; RAG returns
   nothing without it. (Optional: `RERANK_URL` for the reranker — the pipeline falls
   back to vector+keyword order if absent.)
4. `GROQ_API_KEY` is required at **boot** (views.py builds the pipeline at import), not
   just at request time — make sure it's set or the web service won't start.

## Render deploy (Blueprint)

1. Push the repo to GitHub.
2. Render → **New → Blueprint** → pick this repo. It reads `render.yaml`.
3. Fill every `sync: false` env var in the dashboard (see `.env.example` for meanings).
4. Deploy. `preDeployCommand` runs `migrate` once per release before traffic shifts.
5. Set `DJANGO_ALLOWED_HOSTS` to your `*.onrender.com` host and `CORS_ALLOWED_ORIGINS`
   /`CSRF_TRUSTED_ORIGINS` to your frontend origin.

**Cost note:** Render **background workers require a paid plan** (no free tier). The web
service can run on `free` (with cold-start spin-down); the Celery worker cannot, and
without it uploads never get processed.

## Other platforms

The Dockerfile is platform-neutral. Railway / Fly.io / Google Cloud Run all build it the
same way — provide the same env vars and a start command:
`gunicorn core.asgi:application -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`.
Run the Celery worker as a second process/service with
`celery -A core worker -l info --concurrency 2`.

## Local: build & run the prod image

```bash
cd backend
docker build -t rag-tutor-backend .
docker run -p 8000:8000 \
  -e SECRET_KEY=dev -e DEBUG=True -e GROQ_API_KEY=$GROQ_API_KEY \
  rag-tutor-backend
# probe: curl http://localhost:8000/ping/  ->  {"status": "ok"}
```
