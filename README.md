# RAG Knowledge Assistant
Production-grade AI-powered document intelligence platform

## 📖 Overview

RAG Knowledge Assistant is a production-oriented, full-stack AI platform for intelligent document interaction. Upload PDFs, books, or study materials and interact with them through conversational AI with source citations, automated flashcard generation, and AI-powered quiz workflows.

Built with async ingestion pipelines, distributed task orchestration, real-time WebSocket updates, and retrieval-focused backend architecture.

> Semantic Search + RAG + LLM Generation + Study Automation — unified into a scalable AI knowledge workspace.

---

## ✨ Key Features

| Feature | Description | Impact |
|---|---|---|
| 📄 Intelligent Document Ingestion | OCR extraction + semantic chunking → auto-indexed into vector database | Upload once, retrieve instantly |
| 🔍 Hybrid Search | Vector similarity + cross-encoder reranking for relevance | Improved retrieval quality with reduced hallucinations |
| 💬 Conversational AI | Query expansion + contextualization → Llama 3 generation | Fast, contextual answers with citations |
| 🎓 Study Automation | AI-generated flashcards + quizzes from uploaded documents | Streamlined study workflows |
| ⚡ Real-time Updates | WebSocket streams for async ingestion status | Live indexing updates without refresh |
| 📊 Chat Memory | Multi-turn conversations with persistent document context | Natural conversational experience |
| 🔐 Multi-tenant Architecture | User isolation, secure document storage, scoped retrieval | Production-oriented backend design |

---

##  Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 13+
- Redis 7+
- Qdrant (Docker or Cloud)


### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/rag-knowledge-assistant.git
cd rag-knowledge-assistant

# =========================
# Backend Setup
# =========================
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# =========================
# Frontend Setup
# =========================
cd ../frontend

# Install frontend dependencies
npm install

# =========================
# Environment Variables
# =========================
cp .env.example .env

# Fill in environment variables
# (See Configuration section below)

# =========================
# Start Infrastructure Services
# =========================
docker-compose up -d

# Starts:
# - PostgreSQL
# - Redis
# - Qdrant

# =========================
# Run Backend
# =========================
cd backend

# Apply migrations
python manage.py migrate

# Start Celery worker
celery -A core worker -l info

# Start Django ASGI server
python manage.py runserver

# =========================
# Run Frontend
# =========================
cd ../frontend

npm run dev
````

### Access Application

Frontend:

```bash
http://localhost:3000
```

Backend API:

```bash
http://localhost:8000
```

## System Architecture

High-Level Overview

<img width="720" height="960" alt="robot_system_design excalidraw" src="https://github.com/user-attachments/assets/709964ff-d243-488d-880a-4effe4d5a3ea" />


## 🔄 Data Flow: Document Upload → Retrieval → Generation

### Step 1 — Document Upload

```text
User Uploads PDF
        ↓
Django REST API
        ↓
Store File in Supabase Storage
        ↓
Queue Celery Ingestion Task
````

---

### Step 2 — Async Ingestion Pipeline (Celery Worker)

```text
OCR Extraction
        ↓
Semantic Chunking (256–512 token windows)
        ↓
Embedding Generation (TEI Service)
        ↓
Store Embeddings in Qdrant
        ↓
Update PostgreSQL Metadata
        ↓
Broadcast WebSocket Update
        ↓
Frontend Receives "Ready" Status
```

---

### Step 3 — Retrieval Pipeline

```text
User Query
        ↓
Contextualize Query using Chat History
        ↓
Query Expansion (3–4 variations)
        ↓
Generate Query Embeddings
        ↓
Parallel Vector Search in Qdrant
        ↓
Deduplicate Retrieved Chunks
```

---

### Step 4 — Reranking + Generation

```text
Cross-Encoder Reranking
        ↓
Filter High-Relevance Chunks
        ↓
Build Context Prompt
        ↓
Llama 3 Generation (Groq)
        ↓
Stream Response with Citations
        ↓
Store Chat History
```

---

# 📋 RAG Pipeline (Detailed)

## Ingestion Pipeline

```python
# Step 1: OCR Extraction
documents = pytesseract.image_to_pdf_or_hocr(image_file)
text = pytesseract.image_to_string(image_file)

# Step 2: Semantic Chunking
chunks = semantic_chunker.chunk(
    text,
    chunk_size=512,
    overlap=128,
    split_on_sentences=True
)

# Step 3: Embedding Generation
embeddings = tei_client.embed(
    [chunk.text for chunk in chunks],
    model="sentence-transformers/bge-large-en-v1.5"
)

# Step 4: Vector Storage
qdrant_client.upsert(
    collection_name="documents",
    points=[
        Point(
            id=chunk.id,
            vector=embedding,
            payload={
                "document_id": doc.id,
                "text": chunk.text,
                "page": chunk.page,
                "chapter": chunk.chapter
            }
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
)
```

---

## Retrieval Pipeline

```python
# Step 1: Contextualize Query
def contextualize_query(query, chat_history):
    history_context = "\n".join([
        f"{msg.sender}: {msg.text}"
        for msg in chat_history[-5:]
    ])

    rewritten = llm.generate(
        f"Rewrite this query given context:\n"
        f"{history_context}\n"
        f"Query: {query}"
    )

    return rewritten


# Step 2: Query Expansion
def expand_query(query):
    variations = llm.generate_variations(
        query,
        count=3
    )

    return [query] + variations


# Step 3: Vector Retrieval
def retrieve_chunks(queries):
    all_results = []

    for q in queries:
        q_embedding = tei_client.embed(q)

        results = qdrant_client.search(
            collection_name="documents",
            query_vector=q_embedding,
            limit=8,
            with_payload=True
        )

        all_results.extend(results)

    # Deduplicate chunks
    unique_chunks = {
        r.payload["id"]: r
        for r in all_results
    }

    return list(unique_chunks.values())


# Step 4: Cross-Encoder Reranking
def rerank_chunks(query, chunks):
    scores = cross_encoder.predict([
        [query, chunk.payload["text"]]
        for chunk in chunks
    ])

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        chunk
        for chunk, score in ranked
        if score >= 4
    ]


# Step 5: Generation
def generate_answer(query, context_chunks):

    context = "\n---\n".join([
        f"[{c.payload['page']}] {c.payload['text']}"
        for c in context_chunks
    ])

    prompt = f'''
    Based on this context:
    {context}

    Answer the question: {query}

    Include source citations like [page X].
    '''

    response = groq_client.generate(
        prompt,
        model="llama-3.1-8b-instant",
        stream=True
    )

    return response
```

# ⚙️ Production Architecture

## Async Task Processing (Celery + Redis)

### Why Celery?

- Long-running OCR, embedding, and indexing tasks do not block the API layer
- Horizontally scalable worker architecture
- Automatic retry handling for transient failures
- Observable async task lifecycle with structured logs and status tracking

---

## Task Flow

```python
@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3}
)
def ingest_document(self, document_id: int):
    """Async ingestion: OCR → Chunk → Embed → Index."""

    with transaction.atomic():
        doc = Document.objects.select_for_update().get(id=document_id)
        doc.status = "PROCESSING"
        doc.save()

    try:
        # Extract text
        text = extract_text(doc.file)

        # Semantic chunking
        chunks = semantic_chunk(
            text,
            chunk_size=512
        )

        # Generate embeddings
        embeddings = embed_batch(chunks)

        # Insert into Qdrant
        qdrant_client.upsert(
            collection_name="documents",
            points=[...indexed chunks...]
        )

        # Update document status
        with transaction.atomic():
            doc.status = "INDEXED"
            doc.chunks_count = len(chunks)
            doc.save()

        # Notify frontend
        send_websocket_update(doc.user_id, {
            "document_id": document_id,
            "status": "INDEXED",
            "chunks": len(chunks)
        })

    except Exception as e:
        doc.status = "FAILED"
        doc.error_message = str(e)
        doc.save()
        raise
````

---

## Retry Strategy

```python
@shared_task(
    autoretry_for=(TimeoutError, ConnectionError),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=5
)
def process_chunk(chunk_id, embedding):

    # Automatically retries on:
    # - network failures
    # - timeout errors
    # - transient service failures

    pass
```

---

# 🔌 WebSocket Real-Time Updates

## Why WebSockets?

* Real-time ingestion status updates
* Push-based UI synchronization
* Live document processing feedback
* Eliminates constant frontend polling

---

## WebSocket Consumer

```python
# consumers.py

class DocumentConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.user_id = self.scope["user"].id
        self.group_name = f"user_{self.user_id}"

        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        await self.accept()

    async def document_status_update(self, event):

        await self.send(
            text_data=json.dumps({
                "type": "document_indexed",
                "document_id": event["document_id"],
                "status": event["status"],
                "chunks": event["chunks"]
            })
        )
```

---

## Celery → WebSocket Notification

```python
# tasks.py

def notify_document_ready(document_id, user_id):

    channel_layer = get_channel_layer()

    async_to_sync(channel_layer.group_send)(
        f"user_{user_id}",
        {
            "type": "document_status_update",
            "document_id": document_id,
            "status": "INDEXED"
        }
    )
```

---

# 🛡️ Resilience & Failure Handling

The system includes retry-aware ingestion tasks and lightweight failure-handling patterns for external AI services.

## Failure Handling Goals

* Prevent failed OCR or embedding jobs from crashing ingestion pipelines
* Retry transient embedding and vector DB failures
* Maintain deterministic document states
* Preserve frontend synchronization during async processing

---

## Lightweight Circuit Breaker Pattern

```python
class CircuitBreaker:

    def __init__(
        self,
        failure_threshold=3,
        recovery_timeout=60
    ):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = "CLOSED"

    def call(self, func, *args, **kwargs):

        if self.state == "OPEN":

            if (
                time.time() - self.last_failure_time
                > self.recovery_timeout
            ):
                self.state = "HALF_OPEN"

            else:
                raise CircuitBreakerOpen(
                    "Service temporarily unavailable"
                )

        try:
            result = func(*args, **kwargs)

            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception:
            self.failure_count += 1

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.last_failure_time = time.time()

            raise
```

---

# ⚡ Concurrency Control

```python
# Limit concurrent LLM requests

MAX_CONCURRENT_LLM_CALLS = 20

llm_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_LLM_CALLS
)

async def generate_with_rate_limit(
    query,
    context
):

    async with llm_semaphore:

        return await groq_client.generate_async(
            query,
            context
        )
```

---

# 📈 Observability & Instrumentation

The backend includes lightweight instrumentation utilities for monitoring:

* Embedding latency
* Async task duration
* Retrieval pipeline timing
* Batch processing throughput
* P95 / P99 latency windows
* Retrieval relevance heuristics

Custom observability helpers were built using rolling-window latency tracking and retrieval evaluation utilities integrated into the RAG workflow.

## ⚡ Performance Optimizations

Key optimizations used to achieve low-latency RAG responses:

- Parallel query expansion using async workflows
- Local embedding inference via TEI
- Retrieval deduplication before reranking
- Reranking only top candidate chunks
- Streaming LLM responses for reduced perceived latency

---

# 🛠️ Tech Stack

## Backend

| Category | Technology | Purpose |
|---|---|---|
| Framework | Django 4.2+ | API layer, business logic, ORM |
| API | Django REST Framework | REST endpoints |
| Async Tasks | Celery | Background ingestion workflows |
| Message Broker | Redis | Task queue + caching |
| Database | PostgreSQL 13+ | Metadata, chat history |
| API Server | Uvicorn (ASGI) | High-performance async server |
| WebSocket | Django Channels | Real-time updates |

---

## AI / RAG

| Component | Technology | Purpose |
|---|---|---|
| Vector Database | Qdrant | Store + retrieve embeddings |
| Embeddings | TEI (Text Embeddings Inference) | Fast local embeddings |
| OCR | Tesseract | Extract text from PDFs |
| Chunking | Recursive + Semantic Chunking | Intelligent document splitting |
| Reranking | Cross-Encoder (Sentence Transformers) | Relevance scoring |
| LLM | Llama 3 via Groq | Low-latency response generation |

---

## Frontend

| Component | Technology | Purpose |
|---|---|---|
| Framework | Next.js 14+ | React app + SSR |
| Language | TypeScript | Type-safe frontend |
| Styling | TailwindCSS | Utility-first styling |
| State Management | Zustand | Lightweight global state |
| Real-time Updates | WebSocket + Polling | Async ingestion synchronization |
| UI Components | shadcn/ui | Reusable component system |

---

## Infrastructure

| Component | Technology | Purpose |
|---|---|---|
| Storage | Supabase (PostgreSQL + S3) | Database + file storage |
| Containerization | Docker | Reproducible environments |
| Orchestration | Docker Compose | Local/staging deployment |
| Monitoring | Lightweight instrumentation utilities | Latency + retrieval observability |
| Logging | Structured JSON logs | Debugging + ingestion tracing |

---

# 📁 Project Structure

```text
rag-knowledge-assistant/
├── backend/
│   ├── core/
│   │   ├── settings.py
│   │   ├── asgi.py
│   │   └── urls.py
│   │
│   ├── documents/
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── serializers.py
│   │   └── tasks.py
│   │
│   ├── chat/
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── consumers.py
│   │   └── tasks.py
│   │
│   ├── retrieval/
│   │   ├── pipeline.py
│   │   ├── query_expansion.py
│   │   └── cross_encoder.py
│   │
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   │
│   ├── utils/
│   │   ├── observability.py
│   │   ├── retry.py
│   │   └── circuit_breaker.py
│   │
│   ├── manage.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── types/
│   │   └── utils/
│   │
│   ├── package.json
│   └── tsconfig.json
│
├── docker-compose.yml
├── .env.example
├── README.md
└── LICENSE
````

---

# 🔧 Configuration

## Environment Variables

```bash
# =========================
# Django
# =========================
DEBUG=False
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1

# =========================
# PostgreSQL
# =========================
DATABASE_URL=postgresql://user:password@postgres:5432/rag_db

# =========================
# Redis
# =========================
REDIS_URL=redis://redis:6379/0

# =========================
# Qdrant
# =========================
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=your-qdrant-key

# =========================
# Embeddings (TEI)
# =========================
TEI_EMBEDDING_API=http://tei-embedding:8080
EMBEDDING_MODEL=sentence-transformers/bge-large-en-v1.5

# =========================
# LLM (Groq)
# =========================
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama-3.1-8b-instant

# =========================
# Supabase Storage
# =========================
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_API_KEY=your-supabase-key
SUPABASE_BUCKET=documents

# =========================
# Frontend
# =========================
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

---

# 🚀 Deployment

## Docker Compose (Development)

```yaml
version: '3.8'

services:

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password

    volumes:
      - postgres_data:/var/lib/postgresql/data

    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine

    ports:
      - "6379:6379"

  qdrant:
    image: qdrant/qdrant:v1.7.0

    ports:
      - "6333:6333"

    volumes:
      - qdrant_data:/qdrant/storage

  tei:
    image: ghcr.io/huggingface/text-embeddings-inference:0.4.0

    ports:
      - "8080:80"

    environment:
      MODEL_ID: sentence-transformers/bge-large-en-v1.5

  backend:
    build: ./backend

    command: >
      sh -c "
      python manage.py migrate &&
      uvicorn core.asgi:application
      --host 0.0.0.0
      "

    ports:
      - "8000:8000"

    environment:
      - DEBUG=True
      - DATABASE_URL=postgresql://user:password@postgres:5432/rag_db
      - REDIS_URL=redis://redis:6379/0

    depends_on:
      - postgres
      - redis
      - qdrant
      - tei

  celery:
    build: ./backend

    command: celery -A core worker -l info

    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/rag_db
      - REDIS_URL=redis://redis:6379/0

    depends_on:
      - postgres
      - redis

  frontend:
    build: ./frontend

    command: npm run dev

    ports:
      - "3000:3000"

    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000

    depends_on:
      - backend

volumes:
  postgres_data:
  qdrant_data:
```

---

## Run Services

```bash
docker-compose up -d
```

---

# 🧪 Testing

```bash
# Backend tests
pytest backend/tests/ -v --cov=backend

# Frontend tests
npm test

# Integration tests
pytest tests/integration/ -v
```

# 📈 Scalability & Future Improvements

## Current Limitations & Roadmap

| Limitation | Planned Improvement | Priority |
|---|---|---|
| Single-worker Celery setup | Horizontal worker scaling | High |
| Single Qdrant instance | Distributed Qdrant cluster | Medium |
| Local embedding inference only | Multi-GPU embedding pipeline | Medium |
| Lightweight monitoring only | Expanded observability stack | High |
| No semantic caching layer | Redis-based retrieval cache | High |
| File upload limits | Chunked uploads + resumable ingestion | Medium |

---

## Scaling Strategy

### Phase 1 — Current Architecture

- Horizontal Celery worker scaling
- Redis-backed async task queues
- PostgreSQL optimization for chat/query workloads
- Retrieval deduplication + reranking optimization
- Lightweight latency instrumentation

---

### Phase 2 — Distributed Retrieval Infrastructure

- Distributed Qdrant deployment
- Multi-GPU embedding inference
- Parallel embedding batch processing
- CDN-backed frontend asset delivery
- Improved ingestion throughput optimization

---

### Phase 3 — Advanced Orchestration

- Container orchestration with Kubernetes
- Auto-scaling workers based on queue depth
- Distributed tracing + expanded observability
- Advanced caching strategies
- Multi-region deployment architecture

---

# 🎯 Performance Characteristics

## Query-to-Answer Latency Breakdown

Average end-to-end response latency:

```text
~1.8s average RAG response time
````

### Approximate Pipeline Breakdown

| Stage                   | Average Latency | Contribution |
| ----------------------- | --------------- | ------------ |
| Query Contextualization | ~200ms          | 11%          |
| Query Expansion         | ~180ms          | 10%          |
| Vector Search           | ~45ms           | 2%           |
| Deduplication           | ~25ms           | 1%           |
| Cross-Encoder Reranking | ~95ms           | 5%           |
| LLM Generation          | ~1200ms         | 67%          |

---

## Performance Optimizations

* Parallel query expansion using async workflows
* Local embedding inference (TEI)
* Retrieval deduplication before reranking
* Limited reranking candidate windows
* Streaming LLM responses
* Batched embedding generation
* Async ingestion orchestration

---

# 🤝 Contributing

Contributions are welcome.

## Development Workflow

```bash id="9xwsm1"
# Fork repository

# Create feature branch
git checkout -b feature/amazing-feature

# Commit changes
git commit -m "Add amazing feature"

# Push branch
git push origin feature/amazing-feature
```

Then open a Pull Request.

---

# 📝 License

MIT License — see the `LICENSE` file for details.

---

# 🙏 Acknowledgments

Inspired by technologies and ecosystems including:

* LangChain — Retrieval orchestration patterns
* Qdrant — Vector database architecture
* Groq — Low-latency inference
* Django — Backend framework ecosystem
* Next.js — Modern React infrastructure

---

# 📞 Contact

## Abhishek Rajput

* 📧 Email: `abhishek.rajput7202@gmail.com`
* 💼 LinkedIn: `linkedin.com/in/abhishek-rajput-4ba60221a`
* 💻 GitHub: `github.com/Abhishekvoid`

---

## Open To

* Backend Engineering
* AI Infrastructure
* Distributed Systems
* Platform Engineering


