import os
import logging
import tiktoken
import io # Import for in-memory file handling
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
from groq import Groq # Import Groq

# Import your Chapter and Document models
from .models import Document, Chapter

BATCH_SIZE = 100
logger = logging.getLogger(__name__)

load_dotenv()
QDRANT_URL = getattr(settings, "QDRANT_URL", "http://localhost:6333")
GOOGLE_API_KEY = getattr(settings, "GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
GROQ_API_KEY = getattr(settings, "GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "mixtral-8x7b-32768"
QDRANT_COLLECTION_NAME = "studywise_documents"
TOKENIZER_NAME = "cl100k_base"
MAX_CHUNKS_PER_DOCUMENT = 1000

# Client initialization moved inside tasks to be process-safe
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
    text = ""
    with default_storage.open(document_path, 'rb') as f:
        # Read file into an in-memory stream for robustness
        in_memory_file = io.BytesIO(f.read())
        
        if file_type == 'pdf':
            reader = PyPDF2.PdfReader(in_memory_file)
            for page in reader.pages:
                text += page.extract_text() or ""
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
    return text

def chunk_text_by_token(text, tokenizer, chunk_size=384, chunk_overlap=50):
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

# ----- NEW "SMART CHAPTER" TASK -----
@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def create_chapter_from_document(self, document_id: str):
    logger.info(f"[{document_id}] Starting smart chapter creation...")
    doc = None
    try:
        _, _, groq_client = _get_clients()
        _initialize_google_ai()

        doc = Document.objects.get(id=document_id)
        doc.status = "PROCESSING"
        doc.save()

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

        process_document_ingestion.delay(str(doc.id))

        logger.info(f"[{document_id}] Successfully created chapter '{ai_generated_title}'")
    except Exception as e:
        logger.error(f"[{document_id}] Smart chapter creation failed: {e}", exc_info=True)
        if doc:
            doc.status = 'FAILED'
            doc.save()
        raise self.retry(exc=e)

# ----- ORIGINAL DOCUMENT PROCESSING TASK --------
@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_document_ingestion(self, document_id: str):
    logger.info(f"[{document_id}] Starting document ingestion for RAG...")
    doc = None
    try:
        qdrant_client, tokenizer, _ = _get_clients()
        _initialize_google_ai()
        
        doc = Document.objects.get(id=document_id)
        if not doc.extracted_text:
            logger.info(f"[{document_id}] Extracting text for ingestion...")
            doc.extracted_text = get_text_from_file(doc.file.name, doc.file_type)
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

        # ✅ FIX: points_batch is now a local variable
        points_batch = []
        for i, (chunk, vector) in enumerate(zip(text_chunks, all_embeddings)):
            points_batch.append(PointStruct(
                id=f"{document_id}_{i}",
                vector=vector,
                payload={
                    "text": chunk,
                    "document_id": str(document_id),
                    "file_type": doc.file_type,
                    "user_id": str(doc.user.id) # Important for filtering
                }
            ))

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
            
        doc.status = 'COMPLETED'
        doc.save(update_fields=['status'])
        logger.info(f"[{document_id}] Ingestion successful.")
    except Document.DoesNotExist:
        logger.error(f"[{document_id}] Document not found.")
    except Exception as e:
        logger.error(f"[{document_id}] Ingestion failed: {e}", exc_info=True)
        if doc:
            doc.status = 'FAILED'
            doc.save(update_fields=['status'])
        raise self.retry(exc=e)