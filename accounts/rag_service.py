# backend/rag_service.py
import logging
import asyncio
import uuid
from types import SimpleNamespace
from typing import List, Union, Optional
from .ai_clients import get_pinecone_index
from utils.embedding import EmbeddingClient

embedding_client = EmbeddingClient()
logger = logging.getLogger(__name__)


async def embed_texts(texts: Union[str, List[str]]) -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]
    embeddings = await embedding_client.embed_texts(texts)
    return embeddings


def _clean_metadata(payload: dict) -> dict:
    """Pinecone metadata cannot hold null values, so drop any None entries."""
    return {k: v for k, v in (payload or {}).items() if v is not None}


def _to_result(match):
    """Normalize a Pinecone match into an object with the same shape the pipeline
    expects from the old Qdrant points: mutable `.score`, dict `.payload`, `.id`.
    """
    if isinstance(match, dict):
        _id = match.get("id")
        score = match.get("score", 0.0)
        metadata = match.get("metadata") or {}
    else:
        _id = getattr(match, "id", None)
        score = getattr(match, "score", 0.0)
        metadata = getattr(match, "metadata", None) or {}
    return SimpleNamespace(id=_id, score=float(score or 0.0), payload=dict(metadata))


def _query_one(vector: List[float], filter: Optional[dict], top_k: int):
    """Blocking single-vector query against Pinecone (run in a thread)."""
    index = get_pinecone_index()
    return index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        filter=filter,
    )


async def search_vectors(
    vectors: List[List[float]],
    filter: Optional[dict],
    limit_per_vector: int = 5,
):
    if not vectors:
        logger.warning("No vectors provided")
        return []

    # Pinecone's Python SDK is synchronous; fan out across threads to keep the
    # same concurrent, per-vector query behavior the async Qdrant path had.
    tasks = [
        asyncio.to_thread(_query_one, v, filter, limit_per_vector)
        for v in vectors
    ]

    try:
        results = await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Pinecone query failed: {e}", exc_info=True)
        return []

    logger.info(f"Executed {len(results)} parallel searches")

    flat = []
    for res in results:
        matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])
        for m in (matches or []):
            flat.append(_to_result(m))

    if not flat:
        return []

    # deduplicate by point id
    seen = set()
    unique = []
    for r in flat:
        if r.id not in seen:
            seen.add(r.id)
            unique.append(r)

    # sort by score desc
    unique.sort(key=lambda x: getattr(x, "score", 0), reverse=True)

    return unique[:20]


async def store_context(payload: dict, vector: List[float], id: str = None):
    """Optional: write a new point into Pinecone to act as cached context.

    payload: dict with text, chapter_id, user_id...
    vector: embedding vector for the payload
    """
    point_id = id or str(uuid.uuid4())
    index = get_pinecone_index()
    await asyncio.to_thread(
        index.upsert,
        vectors=[{
            "id": point_id,
            "values": vector,
            "metadata": _clean_metadata(payload),
        }],
    )
    logger.info("Stored context to Pinecone (maybe cache)")


def make_chapter_user_filter(chapter_id: str, user_id: str) -> dict:
    """Pinecone metadata filter matching a specific chapter and owner."""
    return {
        "chapter_id": {"$eq": str(chapter_id)},
        "user_id": {"$eq": str(user_id)},
    }
