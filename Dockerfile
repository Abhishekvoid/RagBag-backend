# RAG-tutor Django backend — production image.
#
# Why Docker (not Render's native Python runtime): the ingestion pipeline shells
# out to OS binaries that pip cannot install:
#   poppler-utils -> pdf2image (renders PDF pages to images)
#   tesseract-ocr -> pytesseract (OCR fallback for scanned pages)
#
# No torch/transformers here — embeddings and reranking are external TEI services
# reached over TEI_URL / RERANK_URL (see requirements.txt).

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an unprivileged user. Created after the pip install so the site-packages
# layer stays root-owned and read-only to the app — a compromised worker cannot
# rewrite its own dependencies.
RUN useradd --system --create-home --uid 10001 appuser

# Collect static at build time. The dummy env vars exist ONLY for this one command
# (settings.py refuses to import without SECRET_KEY; DEBUG=True skips the prod-only
# PINECONE guard). Real values are injected by the platform at runtime and override
# anything set here. collectstatic touches neither the DB nor Pinecone.
RUN SECRET_KEY=build-time-only DEBUG=True python manage.py collectstatic --noinput

# The app only needs to WRITE where uploads are staged before going to S3 and
# where whitenoise reads compressed static. Everything else stays root-owned and
# read-only to appuser.
RUN mkdir -p /app/media /app/staticfiles \
    && chown -R appuser:appuser /app/media /app/staticfiles

USER appuser

# Render/Railway/Fly inject $PORT. Default 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# ASGI (not WSGI) because the app serves WebSockets via Django Channels.
# exec-form + `exec` so gunicorn is PID 1 and receives SIGTERM for graceful
# shutdown; `sh -c` keeps $PORT / $WEB_CONCURRENCY expansion. (Render overrides
# this via startCommand; it's the default for local `docker run`.)
CMD ["sh", "-c", "exec gunicorn core.asgi:application -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --workers ${WEB_CONCURRENCY:-2} --timeout 120"]
