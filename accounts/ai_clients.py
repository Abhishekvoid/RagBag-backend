import os
from dotenv import load_dotenv
import logging
from openai import OpenAI, AsyncOpenAI
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


OPENROUTER_API_KEY = _clean_env("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = _clean_env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"

# OpenRouter is the sole LLM provider.
#
# Model choice is deliberate and verified against the live OpenRouter catalog:
#   ANSWER_MODEL  Nemotron 3 Ultra (550B) — best prose quality, 1M context, but it
#                 does NOT support response_format. Asking it for JSON mode returns
#                 HTTP 200 with content=None, so it is used for prose answers only.
#   LLM_MODEL     Nemotron 3 Super (120B) — declares AND honours response_format
#                 plus structured outputs, with ~3x less reasoning overhead than
#                 Ultra. Used for routing, query expansion and every JSON call.
#
# Both are reasoning models: they spend completion tokens on hidden reasoning
# before emitting content, which is why max_tokens budgets downstream are larger
# than they were for the old Groq llama models.
ANSWER_MODEL = _clean_env("OPENROUTER_ANSWER_MODEL") or "nvidia/nemotron-3-ultra-550b-a55b:free"
LLM_MODEL = _clean_env("OPENROUTER_LLM_MODEL") or "nvidia/nemotron-3-super-120b-a12b:free"

PINECONE_API_KEY = _clean_env("PINECONE_API_KEY")
PINECONE_INDEX = _clean_env("PINECONE_INDEX") or "studywise-documents"
PINECONE_CLOUD = _clean_env("PINECONE_CLOUD") or "aws"
PINECONE_REGION = _clean_env("PINECONE_REGION") or "us-east-1"
EMBEDDING_DIM = int(_clean_env("EMBEDDING_DIM") or 384)


if OPENROUTER_API_KEY:
    logger.info("OPENROUTER_API_KEY loaded")
else:
    logger.warning("OPENROUTER_API_KEY not found — no LLM provider configured")


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


# OpenRouter speaks the OpenAI wire protocol, so the stock OpenAI SDK (already a
# dependency, used by vision_ocr) is all that is needed — no extra package.
#
# The rest of the app codes against llm_client / async_llm_client rather than
# these names, so swapping providers stays a one-file change.
llm_client = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    if OPENROUTER_API_KEY else None
)
async_llm_client = (
    AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    if OPENROUTER_API_KEY else None
)
