import os
from dotenv import load_dotenv
import logging

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
GOOGLE_API_KEY = _clean_env("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


if GROQ_API_KEY:
    logger.info("GROQ_API_KEY loaded (masked): %s...%s", GROQ_API_KEY[:4], GROQ_API_KEY[-4:])
else:
    logger.warning("GROQ_API_KEY not found")

if GOOGLE_API_KEY:
    logger.info("GOOGLE_API_KEY loaded (masked): %s...%s", GOOGLE_API_KEY[:4], GOOGLE_API_KEY[-4:])
else:
    logger.warning("GOOGLE_API_KEY not found")

from groq import Groq, AsyncGroq
from qdrant_client import QdrantClient, AsyncQdrantClient
import google.generativeai as genai

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

groq_client       = Groq(api_key=GROQ_API_KEY)       if GROQ_API_KEY   else None
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)  if GROQ_API_KEY   else None
qdrant_client     = QdrantClient(QDRANT_URL)
async_qdrant_client = AsyncQdrantClient(QDRANT_URL)