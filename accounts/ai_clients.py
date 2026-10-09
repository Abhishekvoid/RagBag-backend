import os
from dotenv import load_dotenv
import logging
from openai import OpenAI, AsyncOpenAI
from pinecone import Pinecone, AsyncPinecone, ServerlessSpec, RetryConfig
from contextlib import asynccontextmanager
from utils.deadline import current_deadline, timeout_for

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

# --- Hybrid retrieval (dense + learned-sparse, fused with RRF) ----------------
#
# The sparse half lives in its OWN index, and that is a constraint rather than a
# preference. Pinecone only accepts sparse values inside a dense index when that
# index's metric is `dotproduct`; ours is `cosine`, so co-locating them would
# mean rebuilding and re-embedding the entire corpus. More importantly, a
# single-index sparse-dense query returns ONE already-fused list, scored by
# Pinecone's own internal weighting — there would be no second ranking left for
# RRF to operate on. Two indexes, two rankings, fusion under our control.
#
# Why learned sparse and not BM25: BM25 ranks on IDF, which is a statistic of
# the corpus. Every search here is hard-filtered to a single user's chapter
# inside a multi-tenant index that grows on every upload, so there is no stable
# corpus to compute IDF over — a fitted encoder is stale the moment anyone
# uploads, and refitting means re-encoding everything. pinecone-sparse-english-v0
# is a neural model with no corpus statistics to maintain, and it still gives us
# the exact-term matching (acronyms, formula names, proper nouns) that a 384-dim
# dense model reliably loses.
PINECONE_SPARSE_INDEX = _clean_env("PINECONE_SPARSE_INDEX") or "studywise-documents-sparse"
SPARSE_EMBED_MODEL = _clean_env("SPARSE_EMBED_MODEL") or "pinecone-sparse-english-v0"

# Kill switch. Set HYBRID_SEARCH_ENABLED=false to fall back to dense-only
# retrieval without a deploy — the pipeline already treats an absent sparse
# ranking as a degraded-but-valid state, so this changes result quality and
# nothing else.
HYBRID_SEARCH_ENABLED = (_clean_env("HYBRID_SEARCH_ENABLED") or "true").lower() not in (
    "false", "0", "no", "off",
)

# `input_type` is REQUIRED by this model and is not symmetric: passages and
# queries are encoded differently, and sending the wrong one degrades retrieval
# silently rather than erroring. Named here so neither call site can drift.
SPARSE_INPUT_PASSAGE = "passage"
SPARSE_INPUT_QUERY = "query"

# Allowed values are 512 and 2048. Ingestion already splits every chunk to at
# most 510 non-whitespace characters for the dense model, which is a proven
# upper bound of 512 WordPiece tokens (see utils/token_budget), so 2048 puts
# truncation structurally out of reach instead of relying on it not to trigger.
# The model's default is 512 with truncate=END — silent truncation, the exact
# failure mode token_budget exists to prevent.
SPARSE_MAX_TOKENS_PER_SEQUENCE = 2048


if OPENROUTER_API_KEY:
    logger.info("OPENROUTER_API_KEY loaded")
else:
    logger.warning("OPENROUTER_API_KEY not found — no LLM provider configured")


# Pinecone client is cheap to construct and does no network I/O until used.
pinecone_client = (
    Pinecone(api_key=PINECONE_API_KEY, timeout=10, retry_config=RetryConfig(max_retries=0))
    if PINECONE_API_KEY else None
)


def pinecone_call(method, *args, **kwargs):
    """SDK control/inference methods lack per-call timeouts in Pinecone 9.

    A temporary client gives each call the remaining budget without mutating
    the shared client used by concurrent requests. Index query/upsert methods
    accept their own timeout and continue to reuse the shared connection pool.
    """
    if not current_deadline():
        target = pinecone_client
        for part in method.split("."):
            target = getattr(target, part)
        return target(*args, **kwargs)
    with Pinecone(api_key=PINECONE_API_KEY, timeout=timeout_for(10),
                  retry_config=RetryConfig(max_retries=0)) as client:
        target = client
        for part in method.split("."):
            target = getattr(target, part)
        return target(*args, **kwargs)


def async_pinecone_client():
    return AsyncPinecone(api_key=PINECONE_API_KEY, timeout=timeout_for(10),
                         retry_config=RetryConfig(max_retries=0))


@asynccontextmanager
async def query_index(name):
    # Native async I/O is cancellable; a timed-out to_thread query would keep
    # async_to_sync waiting for its executor thread to finish on loop shutdown.
    async with async_pinecone_client() as client:
        index = await client.index(name=name)
        async with index:
            yield index

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

    if not pinecone_call("has_index", PINECONE_INDEX):
        logger.info("Pinecone index '%s' not found. Creating (%s/%s, dim=%d)...",
                    PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_DIM)
        pinecone_call("create_index",
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            timeout=-1,
        )
        logger.info("Pinecone index '%s' ready.", PINECONE_INDEX)

    if current_deadline():
        description = pinecone_call("describe_index", PINECONE_INDEX)
        _index = pinecone_client.Index(host=description.host)
    else:
        _index = pinecone_client.Index(PINECONE_INDEX)
    return _index


_sparse_index = None


def get_pinecone_sparse_index():
    """Return the shared sparse index handle, creating the index on first use.

    Mirrors get_pinecone_index() so both halves of retrieval have the same
    lazy, idempotent, thread-safe lifecycle. A sparse index takes no
    `dimension` — the vocabulary is unbounded — and MUST be dotproduct, which
    is the only metric defined over sparse vectors.

    Raises rather than returning None so the caller has to make an explicit
    decision about degradation; rag_service catches this and falls back to
    dense-only, which is the one place that policy belongs.
    """
    global _sparse_index
    if _sparse_index is not None:
        return _sparse_index

    if pinecone_client is None:
        raise RuntimeError("PINECONE_API_KEY is not set; cannot connect to Pinecone.")

    if not pinecone_call("has_index", PINECONE_SPARSE_INDEX):
        logger.info("Pinecone sparse index '%s' not found. Creating (%s/%s)...",
                    PINECONE_SPARSE_INDEX, PINECONE_CLOUD, PINECONE_REGION)
        pinecone_call("create_index",
            name=PINECONE_SPARSE_INDEX,
            metric="dotproduct",
            vector_type="sparse",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            timeout=-1,
        )
        logger.info("Pinecone sparse index '%s' ready.", PINECONE_SPARSE_INDEX)

    if current_deadline():
        description = pinecone_call("describe_index", PINECONE_SPARSE_INDEX)
        _sparse_index = pinecone_client.Index(host=description.host)
    else:
        _sparse_index = pinecone_client.Index(PINECONE_SPARSE_INDEX)
    return _sparse_index


# OpenRouter speaks the OpenAI wire protocol, so the stock OpenAI SDK (already a
# dependency, used by vision_ocr) is all that is needed — no extra package.
#
# The rest of the app codes against llm_client / async_llm_client rather than
# these names, so swapping providers stays a one-file change.
llm_client = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, max_retries=0, timeout=45)
    if OPENROUTER_API_KEY else None
)
async_llm_client = (
    AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, max_retries=0, timeout=45)
    if OPENROUTER_API_KEY else None
)
