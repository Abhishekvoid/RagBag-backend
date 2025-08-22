import os
import logging
import tiktoken
import io
from celery import shared_task
from django.conf import settings
from django.core.files.storage import default_storage
from qdrant_client import QdrantClient, models
import google.generativeai as genai
import PyPDF2
import docx
from pptx import Presentation
from dotenv import load_dotenv
from qdrant_client.models import PointStruct
from groq import Groq
from .models import Document, Chapter

import pytesseract
from pdf2image import convert_from_bytes
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import uuid
# ---------------------------------------------

BATCH_SIZE = 100
logger = logging.getLogger(__name__)

load_dotenv()
QDRANT_URL = getattr(settings, "QDRANT_URL", "http://localhost:6333")
GOOGLE_API_KEY = getattr(settings, "GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
GROQ_API_KEY = getattr(settings, "GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "llama3-70b-8192" 
QDRANT_COLLECTION_NAME = "studywise_documents"
TOKENIZER_NAME = "cl100k_base"
MAX_CHUNKS_PER_DOCUMENT = 1000

_qdrant_client = None
_tokenizer = None
_groq_client = None

def _get_clients():
    global _qdrant_client, _tokenizer, _groq_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(QDRANT_URL)
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _qdrant_client, _tokenizer, _groq_client

def _initialize_google_ai():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set.")
    genai.configure(api_key=GOOGLE_API_KEY)

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

def chunk_text_by_token(text, tokenizer, chunk_size=384, chunk_overlap=50):
    # ... (this function remains the same) ...
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

# ----- CORRECTED "SMART CHAPTER" TASK -----
@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def create_chapter_from_document(self, document_id: str):
    logger.info(f"[{document_id}] Starting smart chapter creation...")
    doc = Document.objects.get(id=document_id)

    # --- NEW: Set status to PROCESSING immediately ---
    doc.status = Document.STATUS_PROCESSING
    doc.save(update_fields=['status'])

    try:
        _, _, groq_client = _get_clients()
        _initialize_google_ai()

        document_text = get_text_from_file(doc.file.name, doc.file_type)
        if not document_text:
            raise ValueError("No text could be extracted from the document.")

        prompt = f"Based on the following text, create a short, descriptive title (4-5 words max) for a new chapter. Do not use quotes.\n\nTEXT:\n{document_text[:4000]}\n\nTITLE:"
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
        )
        ai_generated_title = chat_completion.choices[0].message.content.strip().strip('"')

        new_chapter = Chapter.objects.create(user=doc.user, name=ai_generated_title)

        doc.chapter = new_chapter
        doc.title = ai_generated_title
        doc.save()

        # --- NEW: Send a success notification ---
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"user_{doc.user.id}",
            {"type": "send_notification", "message": "notebook_updated"}
        )

        # Trigger the ingestion task with the document ID
        process_document_ingestion.delay(str(doc.id))

        logger.info(f"[{document_id}] Successfully created chapter '{ai_generated_title}' and triggered ingestion.")

    except Exception as e:
        logger.error(f"[{document_id}] Chapter creation or ingestion trigger failed: {e}", exc_info=True)
        # --- NEW: On failure, update status and save error ---
        doc.status = Document.STATUS_FAILED
        doc.error_message = str(e)
        doc.save(update_fields=['status', 'error_message'])
        # You can optionally send a failure notification here
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"user_{doc.user.id}",
            {"type": "send_notification", "message": "document_failed", "document_id": str(doc.id)}
        )
        raise self.retry(exc=e)

# ----- CORRECTED DOCUMENT PROCESSING TASK --------
@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_document_ingestion(self, document_id: str):
    logger.info(f"[{document_id}] Starting document ingestion for RAG...")
    doc = Document.objects.get(id=document_id)
    
    try:
        qdrant_client, tokenizer, _ = _get_clients()
        _initialize_google_ai()
        
        # --- NEW: Text extraction happens here first and is saved to the database. ---
        if not doc.extracted_text:
            logger.info(f"[{document_id}] Extracting text for ingestion...")
            doc.extracted_text = get_text_from_file(doc.file.name, doc.file_type)
            # We save the text to the database to avoid redundant extraction.
            doc.save(update_fields=['extracted_text'])

        if not doc.extracted_text.strip():
            raise ValueError("No text available for ingestion.")

        text_chunks = chunk_text_by_token(doc.extracted_text, tokenizer)
        if not text_chunks:
            raise ValueError("Text could not be split into chunks.")
            
        logger.info(f"[{document_id}] Generating embeddings for {len(text_chunks)} chunks...")
        response = genai.embed_content(
            model=f"models/{EMBEDDING_MODEL}",
            content=text_chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        all_embeddings = response['embedding']
        vector_size = len(all_embeddings[0])

        try:
            qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
        except Exception:
            qdrant_client.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )

        points_batch = []
        for chunk, vector in zip(text_chunks, all_embeddings):
            payload = {
                "text": chunk,
                "document_id": str(document_id),
                "file_type": doc.file_type,
                "user_id": str(doc.user.id)
            }
            if doc.chapter:
                payload["chapter_id"] = str(doc.chapter.id)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            points_batch.append(point)
            
            if len(points_batch) >= BATCH_SIZE:
                qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=points_batch,
                    wait=True
                )
                points_batch = []
        
        if points_batch:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points_batch,
                wait=True
            )
            
        # --- NEW: On success, mark as COMPLETED ---
        doc.status = Document.STATUS_COMPLETED
        doc.error_message = None  # Clear any previous errors
        doc.save(update_fields=['status', 'error_message'])
        
        logger.info(f"[{document_id}] Ingestion successful.")

        # --- NEW: Send a success notification ---
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"user_{doc.user.id}",
            {"type": "send_notification", "message": "document_ready", "document_id": str(doc.id)}
        )

    except Exception as e:
        logger.error(f"[{document_id}] Ingestion failed: {e}", exc_info=True)
        # --- NEW: On failure, update status and save error ---
        doc.status = Document.STATUS_FAILED
        doc.error_message = str(e)
        doc.save(update_fields=['status', 'error_message'])
        
        # --- NEW: Send a failure notification ---
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"user_{doc.user.id}",
            {"type": "send_notification", "message": "document_failed", "document_id": str(doc.id)}
        )
        raise self.retry(exc=e)