import os
import logging
import tiktoken
import io
from celery import shared_task
from celery.exceptions import Retry, SoftTimeLimitExceeded
from django.conf import settings
from django.core.files.storage import default_storage

import PyPDF2
import docx
from pptx import Presentation
from dotenv import load_dotenv
from .models import Document, Chapter, DocumentPage
from .ingestion_versions import (
    document_lease, reserve_version, activate_version, fail_version,
    verify_vectors, version_prefix, LeaseLost,
)
from .ai_clients import (
    get_pinecone_index,
    get_pinecone_sparse_index,
    llm_client,
    LLM_MODEL,
    HYBRID_SEARCH_ENABLED,
    SPARSE_INPUT_PASSAGE,
    pinecone_client,
    PINECONE_SPARSE_INDEX,
)
from .rag_service import embed_sparse
from .page_pipeline import build_document_pages, canonical_text_for_document, DocumentOversizedError
from .vision_ocr import strip_uncertainty_markers
from .realtime import (
    push_ingestion_status,
    PHASE_READING, PHASE_NAMING, PHASE_CHUNKING,
    PHASE_EMBEDDING, PHASE_STORING, PHASE_READY, PHASE_FAILED,
)


import pytesseract
from pdf2image import convert_from_bytes
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.db import close_old_connections, transaction
import uuid


from utils.embedding import EmbeddingClient
from utils.token_budget import split_for_embedding
from utils.deadline import DeadlineExceeded, check_deadline, ingestion_deadline, timeout_for

# ---------------------------------------------

BATCH_SIZE = 100
logger = logging.getLogger(__name__)

load_dotenv()
TOKENIZER_NAME = "cl100k_base"
MAX_CHUNKS_PER_DOCUMENT = 1000

_tokenizer = None

def _get_clients():
    """Return (pinecone_index, tokenizer, llm_client).

    The Pinecone index is created lazily on first use and shared across the
    worker process. The LLM client is the shared one from ai_clients
    (OpenRouter primary), so the worker follows the same provider choice as
    the web process.
    """
    global _tokenizer
    pinecone_index = get_pinecone_index()
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    if llm_client is None:
        raise ValueError(
            "No LLM provider configured. Set OPENROUTER_API_KEY."
        )
    return pinecone_index, _tokenizer, llm_client



# ---- HELPER FUNCTIONS -------

def get_text_from_file(document_path, file_type):
    # ... (this function remains the same) ...
    """
    Extracts text from a file, with detailed logging for debugging.
    """
    text = ""
    print(f"--- Starting text extraction for {document_path} ---")
    with default_storage.open(document_path, 'rb') as f:
        file_content_bytes = f.read()
        print(f"Read {len(file_content_bytes)} bytes from storage.")
        in_memory_file = io.BytesIO(file_content_bytes)

        if file_type == 'pdf':
            # 1. Attempt with PyPDF2
            print("Attempting extraction with PyPDF2...")
            try:
                reader = PyPDF2.PdfReader(in_memory_file)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    print(f"  PyPDF2 - Page {i+1} extracted {len(page_text)} characters.")
                    text += page_text
            except Exception as e:
                print(f"  PyPDF2 failed with an error: {e}")
                text = ""

            # 2. Fallback to OCR if PyPDF2 failed
            if not text.strip():
                print("PyPDF2 returned no text. Falling back to OCR...")
                try:
                    images = convert_from_bytes(file_content_bytes)
                    print(f"  pdf2image converted PDF into {len(images)} image(s).")
                    full_ocr_text = ""
                    for i, image in enumerate(images):
                        ocr_text_per_page = pytesseract.image_to_string(image)
                        print(f"  Tesseract OCR - Page {i+1} extracted {len(ocr_text_per_page)} characters.")
                        full_ocr_text += ocr_text_per_page + "\n"
                    text = full_ocr_text
                except Exception as ocr_error:
                    print(f"  OCR processing failed with an error: {ocr_error}")
                    text = ""

        elif file_type == 'docx':
            doc = docx.Document(in_memory_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_type == 'pptx':
            prs = Presentation(in_memory_file)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, 'text'):
                        text += shape.text + "\n"
        elif file_type == 'txt':
            text = in_memory_file.read().decode('utf-8', errors='ignore')
    print(f"--- Finished extraction. Total characters found: {len(text)} ---")
    return text

def chunk_text_by_token(text, tokenizer, chunk_size=200, chunk_overlap=50):
    
    if not text or not tokenizer: return []
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - chunk_overlap
    return chunks

def _sparse_index_or_none(correlation_id=""):
    """The sparse index handle, or None when hybrid is off or unreachable.

    Returning None rather than raising keeps the sparse half strictly optional
    at ingestion: a document that lands in the dense index but not the sparse
    one is still fully answerable, and shows up as `sparse_empty` at query time
    rather than as a failed upload.
    """
    if not HYBRID_SEARCH_ENABLED:
        return None
    try:
        return get_pinecone_sparse_index()
    except DeadlineExceeded:
        raise
    except Exception as e:
        logger.warning(f"[{correlation_id}] sparse index unavailable: {e}")
        return None


def _delete_document_vectors(index, document_id, correlation_id=""):
    """Delete legacy prefixed IDs. Full deletion also removes all other versions."""
    if index is None:
        return 0

    removed = 0
    try:
        for id_page in index.list(prefix=f"{document_id}#"):
            # `index.list()` does NOT yield plain id strings. On pinecone 9.x a
            # page is a ListResponse wrapping ListItem objects, and handing that
            # straight to delete() raises "Type is not JSON serializable:
            # ListResponse" — inside the try below, where it was swallowed as a
            # warning. The purge therefore never deleted anything, silently, for
            # every re-ingest and every document deletion: orphaned vectors
            # stayed queryable and kept being fused into answers.
            #
            # Both shapes are accepted because the SDK has returned bare strings
            # in the past and the cost of tolerating that is one getattr.
            ids = [getattr(item, "id", item) for item in (id_page or [])]
            if ids:
                index.delete(ids=ids)
                removed += len(ids)
    except Exception as e:
        logger.warning(
            f"[{correlation_id}] could not clear existing vectors for "
            f"{document_id}: {e}"
        )
        return removed

    if removed:
        logger.info(f"[{correlation_id}] cleared {removed} existing vectors")
    return removed


def _sparse_index_for_cleanup():
    """Cleanup must reach an existing sparse index even when hybrid is disabled.

    Do not use the ingestion getter: it creates a missing index, which cleanup
    should never do. A service failure propagates so retired versions stay queued.
    """
    if pinecone_client is None:
        raise RuntimeError("Pinecone is not configured")
    if not pinecone_client.has_index(PINECONE_SPARSE_INDEX):
        return None
    return pinecone_client.Index(PINECONE_SPARSE_INDEX)


def _purge_document_vectors(dense_index, document_id, correlation_id=""):
    """Deletion only: remove legacy IDs and every version from BOTH indexes."""
    removed = 0
    try:
        sparse_index = _sparse_index_for_cleanup()
    except Exception:
        logger.warning("Sparse document deletion unavailable", exc_info=True)
        sparse_index = None
    for index in (dense_index, sparse_index):
        if index is None:
            continue
        removed += _delete_document_vectors(index, document_id, correlation_id)
        try:
            # Metadata deletion also catches pre-prefix UUIDs and new version IDs.
            index.delete(filter={"document_id": {"$eq": str(document_id)}})
        except Exception:
            logger.warning("Could not purge all document versions: %s", document_id, exc_info=True)
    return removed


def extract_document_text(doc, *, version=None, check_lease=lambda: None):
    """PDF -> page pipeline (vision/layer canonical text); other types -> legacy extractor."""
    if doc.file_type == "pdf":
        build_document_pages(doc, version=version, check_lease=check_lease)
        return canonical_text_for_document(doc, version=version)
    return get_text_from_file(doc.file.name, doc.file_type)


def build_chunk_metadata(document, chunk, page_number=None, *, version=None):
    metadata = {
        "text": chunk,
        "document_id": str(document.id),
        "user_id": str(document.user_id),
        "file_type": document.file_type,
    }
    if document.chapter_id:
        metadata["chapter_id"] = str(document.chapter_id)
    if page_number is not None and page_number > 0:
        metadata["page_number"] = page_number
    if version is not None:
        metadata["version"] = version
    return metadata


def _version_chunks(doc, text, tokenizer, version):
    """Chunk actual page text, so IDs never depend on a substring page guess."""
    pages = list(doc.pages.filter(version=version).order_by("page_number"))
    sources = [(p.page_number, strip_uncertainty_markers(p.reconstructed_md)) for p in pages]
    if not sources:
        sources = [(0, text)]
    prepared = []
    for page_number, content in sources:
        check_deadline()
        pieces = [piece for chunk in chunk_text_by_token(content, tokenizer)
                  if len(chunk.strip()) > 10
                  for piece in split_for_embedding(chunk.strip())]
        for position, piece in enumerate(pieces):
            prepared.append((f"{version_prefix(doc.id, version)}p{page_number}_c{position}",
                             page_number, piece))
        if len(prepared) > MAX_CHUNKS_PER_DOCUMENT:
            raise DocumentOversizedError(f"Document exceeds the {MAX_CHUNKS_PER_DOCUMENT}-chunk limit.")
    if not prepared:
        raise ValueError("No readable chunks available for ingestion")
    return prepared


def _queue_version_cleanup(document_id):
    try:
        prune_document_versions.apply_async(
            args=[str(document_id)], countdown=settings.INDEX_VERSION_GRACE_SECONDS,
        )
    except Exception:
        logger.exception("Could not queue version cleanup; run prune_index_versions to recover")


@shared_task
def create_chapter_from_document(document_id):
    # All extraction, naming, and indexing mutations now share the same lease.
    process_document_ingestion.delay(str(document_id))


@shared_task
def process_document_for_existing_chapter(document_id, chapter_id):
    process_document_ingestion.delay(str(document_id), chapter_id=str(chapter_id))


@shared_task(bind=True, max_retries=3, default_retry_delay=60,
             soft_time_limit=900, time_limit=930)
@ingestion_deadline
def process_document_ingestion(self, document_id: str, *, rescan=False, chapter_id=None):
    close_old_connections()
    doc = revision = lease = None
    try:
        with document_lease(document_id) as lease:
            if lease is None:
                logger.info("Duplicate ingestion skipped for document %s", document_id)
                return {"status": "skipped", "reason": "ingestion_in_progress"}
            doc, revision = reserve_version(document_id, lease)
            vector_index, tokenizer, llm = _get_clients()
            push_ingestion_status(doc.user_id, doc.id, PHASE_READING)
            if (rescan or not doc.extracted_text or
                    (doc.file_type == "pdf" and not doc.pages.filter(version=doc.active_version).exists())):
                text = extract_document_text(doc, version=revision.version, check_lease=lease.check)
            else:
                text = doc.extracted_text
                # Reindex cached text without exposing pending pages to readers.
                DocumentPage.objects.bulk_create([
                    DocumentPage(document=doc, version=revision.version,
                                 page_number=p.page_number, image_url=p.image_url,
                                 s3_object_key=p.s3_object_key,
                                 reconstructed_md=p.reconstructed_md, text_source=p.text_source)
                    for p in doc.pages.filter(version=doc.active_version)
                ])
            if not text.strip():
                raise ValueError("No text available for ingestion")
            revision.extracted_text = text
            revision.save(update_fields=["extracted_text"])

            if chapter_id is not None or doc.chapter_id is None:
                if chapter_id is not None:
                    chapter = Chapter.objects.get(id=chapter_id, user_id=doc.user_id)
                    title = doc.title
                else:
                    completion = llm.chat.completions.create(
                        model=LLM_MODEL, max_tokens=1000,
                        timeout=timeout_for(45),
                        messages=[{"role": "user", "content":
                                   "Give this study material a short title (4-5 words), without quotes:\n" + text[:4000]}],
                    )
                    title = (completion.choices[0].message.content or "").strip().strip('"')
                    if not title:
                        raise ValueError("Chapter title generation returned no text")
                    chapter = None
                with transaction.atomic():
                    current = Document.objects.select_for_update().get(id=doc.id)
                    lease.check()
                    if current.ingestion_token != lease.token:
                        raise LeaseLost("A newer ingestion owns this document")
                    if chapter is None:
                        chapter = Chapter.objects.create(user_id=doc.user_id, name=title)
                    current.chapter = chapter
                    current.title = title
                    current.save(update_fields=["chapter", "title"])
                doc.chapter = chapter
                doc.title = title
                push_ingestion_status(doc.user_id, doc.id, PHASE_NAMING,
                                      chapter_id=chapter.id, title=title)

            chunks = _version_chunks(doc, text, tokenizer, revision.version)
            revision.chunk_count = len(chunks)
            revision.save(update_fields=["chunk_count"])
            embedding_client = EmbeddingClient()
            sparse_index = _sparse_index_or_none()
            batch_size = 16
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            push_ingestion_status(doc.user_id, doc.id, PHASE_CHUNKING, total_batches=total_batches)
            probe_vector = None
            for offset in range(0, len(chunks), batch_size):
                check_deadline()
                lease.check()
                batch = chunks[offset:offset + batch_size]
                texts = [chunk for _, _, chunk in batch]
                batch_number = offset // batch_size + 1
                push_ingestion_status(doc.user_id, doc.id, PHASE_EMBEDDING,
                                      batch=batch_number, total_batches=total_batches)
                try:
                    embeddings = async_to_sync(embedding_client.embed_texts)(texts)
                    if len(embeddings) != len(batch):
                        raise ValueError("Embedding provider returned an incomplete batch")
                    check_deadline()
                    lease.check()
                    points = [{"id": point_id, "values": vector,
                               "metadata": build_chunk_metadata(doc, chunk, page_number=page,
                                                                version=revision.version)}
                              for (point_id, page, chunk), vector in zip(batch, embeddings)]
                    vector_index.upsert(vectors=points, timeout=timeout_for(10))
                    probe_vector = embeddings[0]
                except (DeadlineExceeded, LeaseLost, SoftTimeLimitExceeded):
                    raise
                except Exception as exc:
                    raise RuntimeError(f"Embedding/index batch {batch_number} failed: {exc}") from exc
                if sparse_index is not None:
                    try:
                        sparse_vectors = async_to_sync(embed_sparse)(texts, input_type=SPARSE_INPUT_PASSAGE)
                        if len(sparse_vectors) != len(batch):
                            raise ValueError("Sparse provider returned an incomplete batch")
                        lease.check()
                        points = [{"id": point_id, "sparse_values": sparse,
                                   "metadata": build_chunk_metadata(doc, chunk, page_number=page,
                                                                    version=revision.version)}
                                  for (point_id, page, chunk), sparse in zip(batch, sparse_vectors)
                                  if sparse["indices"]]
                        if points:
                            sparse_index.upsert(vectors=points, timeout=timeout_for(10))
                    except (DeadlineExceeded, LeaseLost, SoftTimeLimitExceeded):
                        raise
                    except Exception:
                        logger.warning("Sparse batch unavailable; new version remains dense-capable", exc_info=True)
            lease.check()
            verify_vectors(vector_index, [key for key, _, _ in chunks], doc.id,
                           revision.version, lease, probe_vector=probe_vector)
            push_ingestion_status(doc.user_id, doc.id, PHASE_STORING)
            check_deadline()
            activate_version(doc.id, revision, lease)
    except Document.DoesNotExist:
        logger.info("Document %s no longer exists; ingestion stopped", document_id)
        return {"status": "deleted"}
    except Exception as exc:
        logger.exception("Document ingestion failed: %s", document_id)
        if revision is not None:
            try:
                fail_version(document_id, revision, lease, exc)
                _queue_version_cleanup(document_id)
            except Exception:
                logger.exception("Could not record ingestion failure")
        if doc is not None:
            # A failed rebuild does not turn the active document into a failed upload.
            if doc.status != Document.STATUS_COMPLETED and not doc.active_version:
                push_ingestion_status(doc.user_id, doc.id, PHASE_FAILED, error=str(exc))
        if isinstance(exc, (DeadlineExceeded, SoftTimeLimitExceeded, DocumentOversizedError, LeaseLost)):
            raise
        raise self.retry(exc=exc)

    _queue_version_cleanup(document_id)
    push_ingestion_status(doc.user_id, doc.id, PHASE_READY,
                          chapter_id=doc.chapter_id, title=doc.title)
    # Notification/broker failures must never retry an already published version.
    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(f"user_{doc.user_id}", {
            "type": "send_notification", "message": "document_ready", "document_id": str(doc.id),
        })
    except Exception:
        logger.warning("Document ready notification failed", exc_info=True)
    return {"status": "completed", "version": revision.version}


@shared_task
def rescan_document_with_vision(document_id):
    process_document_ingestion.delay(str(document_id), rescan=True)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def prune_document_versions(self, document_id):
    from datetime import timedelta
    from django.utils import timezone
    cutoff = timezone.now() - timedelta(seconds=settings.INDEX_VERSION_GRACE_SECONDS)
    try:
        with document_lease(document_id) as lease:
            if lease is None:
                raise self.retry(countdown=60)
            doc = Document.objects.filter(id=document_id).first()
            if doc is None:
                return
            versions = doc.index_versions.filter(retired_at__lte=cutoff).exclude(
                version=doc.active_version,
            )
            if doc.pending_version is not None:
                versions = versions.exclude(version=doc.pending_version)
            if not versions.exists():
                return
            # Cleanup is strict: losing an index connection retains the DB record
            # so a later task/maintenance command can finish the same work.
            dense = get_pinecone_index()
            sparse = _sparse_index_for_cleanup()
            for revision in versions:
                lease.check()
                for index in (dense, sparse):
                    if index is None:
                        continue
                    if revision.version == 0:
                        index.delete(filter={"$and": [
                            {"document_id": {"$eq": str(doc.id)}},
                            {"user_id": {"$eq": str(doc.user_id)}},
                            {"version": {"$exists": False}},
                        ]})
                    else:
                        for page in index.list(prefix=version_prefix(doc.id, revision.version)):
                            lease.check()
                            ids = [getattr(item, "id", item) for item in page]
                            if ids:
                                index.delete(ids=ids)
                lease.check()
                doc.pages.filter(version=revision.version).delete()
                revision.delete()
    except Retry:
        raise
    except Exception as exc:
        raise self.retry(exc=exc)


@shared_task
def cleanup_document_data(document_ids, file_names):
    """Best-effort cleanup of a deleted chapter/subject's residual data.

    Removes each document's Pinecone vectors (located by the ``<doc_id>#`` id
    prefix set at ingestion) and its file from storage. Runs after the DB rows
    are already gone, so failures here are logged but never surface to the user.
    """
    document_ids = document_ids or []
    file_names = file_names or []

    if document_ids:
        try:
            index = get_pinecone_index()
        except Exception as e:
            logger.error(f"cleanup_document_data: cannot reach Pinecone: {e}")
            index = None

        if index is not None:
            for document_id in document_ids:
                _purge_document_vectors(index, document_id, "cleanup")

    for name in file_names:
        if not name:
            continue
        try:
            if default_storage.exists(name):
                default_storage.delete(name)
        except Exception as e:
            logger.error(f"cleanup_document_data: failed to delete file {name}: {e}")
