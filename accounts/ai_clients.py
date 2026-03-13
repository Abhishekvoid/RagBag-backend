import os
from dotenv import load_dotenv
import logging
from groq import Groq, AsyncGroq
from qdrant_client import QdrantClient, AsyncQdrantClient

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
QDRANT_URL = _clean_env("QDRANT_URL") or "http://localhost:6333"

QDRANT_API_KEY = _clean_env("QDRANT_API_KEY")


if GROQ_API_KEY:
    logger.info("GROQ_API_KEY loaded (masked): %s...%s", GROQ_API_KEY[:4], GROQ_API_KEY[-4:])
else:
    logger.warning("GROQ_API_KEY not found")


qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

async_qdrant_client = AsyncQdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None