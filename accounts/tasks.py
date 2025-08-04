import os
import logging
import tiktoken
from celery import shared_task
from django.conf import settings
from django.core.files.storage import default_storage
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import UpdateStatus
import google.generativeai as genai
import PyPDF2
import docx
from pptx import Presentation
from dotenv import load_dotenv

from .models import Document

logger = logging.getLogger(__name__)

load_dotenv()
QDRANT_URL = getattr(settings, "QDRANT_URL", "http://localhost:6333")
GOOGLE_API_KEY = getattr(settings, "GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
EMBEDDING_MODEL = "text-embedding-004"
QDRANT_COLLECTION_NAME = "studywise_documents"
EMBEDDING_BATCH_SIZE = 100
TOKENIZER_NAME = "cl100k_base"
MAX_CHUNKS_PER_DOCUMENT = 1000 # Process a maximum of 1000 chunks

_qdrant_client = None
_tokenizer = None

def _get_clients():

    global _qdrant_client, _tokenizer
    if _qdrant_client is None:
        logger.info("Initializing Qdrant client...")
        _qdrant_client = QdrantClient(QDRANT_URL)
    if _tokenizer is None:
        logger.info("Initializing TikToken Tokenizer")
        _tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    return _qdrant_client, _tokenizer

def _initialize_google_ai():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in Django settings or .env file.")
    genai.configure(api_key=GOOGLE_API_KEY)

# ---- HELPER FUNCTIONS -------

def get_text_from_file(document_path, file_type):
    text =""
    with default_storage.open(document_path, 'rb') as f:
        if file_type == 'pdf':
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif file_type == 'docx':
            doc = docx.Document(f)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_type =='pptx':
            prs = Presentation(f)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            for run in paragraph.runs:
                                text += run.text
                            text += '\n'
        elif file_type == 'txt':
            text = f.read().decode('utf-8', errors='ignore')
    return text

def chunk_text_by_token(text, tokenizer, chunk_size=384, chunk_overlap=50):

    if not text or not tokenizer: return[]
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


# ----- MAIN CELERY TASK --------

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_document_ingestion(self, document_id: str):

    try:
        qdrant_client, tokenizer = _get_clients()
        _initialize_google_ai()
    except Exception as e:
        logger.critical(f"Failed to initialize clients for task {self.request.id}. Error: {e}")
        raise self.retry(exc=e)
    
    logger.info(f"starting ingestion pipeline for Document ID: {document_id}")
    doc = None
    try: 
        doc = Document.objects.get(id= document_id)
        doc.status = "Processing"
        doc.save(Update_fields=['status'])

        document_text = get_text_from_file(doc.file.name, doc.file_type)
        if not document_text.strip():
            raise ValueError("No text could be extracted from the document.")
        
        doc.extracted_text = document_text
        doc.save(update_fields=['extracted_text'])
        logger.info(f"-> Text extracted successfully.")

        text_chunks = chunk_text_by_token(document_text, tokenizer)
        if len(text_chunks) > MAX_CHUNKS_PER_DOCUMENT:
            logger.warning(f"documents {document_id} has {len(text_chunks)} chunks, excesding the limit of {MAX_CHUNKS_PER_DOCUMENT}. Truncating.")
            text_chunks = text_chunks[:MAX_CHUNKS_PER_DOCUMENT]
        logger.info(f"-> Text split into {len(text_chunks)} tokens-based chunks.")

        embedding_info = genai.get_model(f"models/{EMBEDDING_MODEL}")
        vector_size = embedding_info.output_dimensionality


        try: 
            qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        except Exception:
            logger.info(f"Qdrant collection not found. Creating with vector size {vector_size}.")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

        all_embeddings = []
        for i in range(0, len(text_chunks), EMBEDDING_BATCH_SIZE):
            batch = text_chunks[i:i + EMBEDDING_BATCH_SIZE]
            result = genai.embed_content(model=f"model/{EMBEDDING_MODEL}", content=batch, task_type="RETRIEVAL_DOCUMENT")

            if 'embedding' not in result:
                raise ValueError ("Google AI API response did not contain 'embedding' key.")
            all_embeddings.extend(result['embedding'])
        logger.info("-> ALL embeddings generated.")

        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id = f"{document_id}_{i}",
                    vector=all_embeddings[i],
                    payload={"text": text_chunks[i], 'document_id': str(document_id) }
                ) for i in range (len(text_chunks))
            ],
            wait=True               
        )
        logger.info("-> Vectors sucessfully stored in Qdrant.")

        doc.status = 'COMPLETED'
        doc.save(update_fields = ['status'])
        logger.info(f"SUCCESS: Ingestion pipeline finished for Document ID: {document_id}")

    except Document.DoesNotExist:
        logger.error(f"Document with ID {document_id} not found.")
    except Exception as e:
        logger.error(f"ERROR during ingestion for Document ID {document_id}: {e}", exc_info=True)
        if doc:
            doc.status = 'FAILED'
            doc.save(update_fields=['status'])
        raise self.retry(exc=e)