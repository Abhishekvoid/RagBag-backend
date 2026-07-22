import os
from dotenv import load_dotenv
import logging
from groq import Groq, AsyncGroq
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logger = logging.getLogger(__name__)

def _clean_env(name: str):
    v = os.getenv(name)
    if not v:
        return None
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v


GROQ_API_KEY = _clean_env("GROQ_API_KEY")

PINECONE_API_KEY = _clean_env("PINECONE_API_KEY")
PINECONE_INDEX = _clean_env("PINECONE_INDEX") or "studywise-documents"
PINECONE_CLOUD = _clean_env("PINECONE_CLOUD") or "aws"
PINECONE_REGION = _clean_env("PINECONE_REGION") or "us-east-1"
EMBEDDING_DIM = int(_clean_env("EMBEDDING_DIM") or 384)


if GROQ_API_KEY:
    logger.info("GROQ_API_KEY loaded (masked): %s...%s", GROQ_API_KEY[:4], GROQ_API_KEY[-4:])
else:
    logger.warning("GROQ_API_KEY not found")


# Pinecone client is cheap to construct and does no network I/O until used.
pinecone_client = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None

_index = None


def get_pinecone_index():
    """Return the shared Pinecone index handle, creating the index on first use.

    Idempotent and lazy: safe to call from both the Django request path and the
    Celery worker. The index handle is thread-safe for query/upsert calls, so we
    reuse a single cached instance across the process.
    """
    global _index
    if _index is not None:
        return _index

    if pinecone_client is None:
        raise RuntimeError("PINECONE_API_KEY is not set; cannot connect to Pinecone.")

    if not pinecone_client.has_index(PINECONE_INDEX):
        logger.info("Pinecone index '%s' not found. Creating (%s/%s, dim=%d)...",
                    PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_DIM)
        pinecone_client.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        logger.info("Pinecone index '%s' ready.", PINECONE_INDEX)

    _index = pinecone_client.Index(PINECONE_INDEX)
    return _index


groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
