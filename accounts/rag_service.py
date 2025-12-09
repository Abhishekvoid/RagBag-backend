# backend/rag_service.py
import logging
import asyncio
from typing import List
from .ai_clients import async_groq_client, async_qdrant_client
import google.generativeai as genai
from qdrant_client import models

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-004"  # your embedding model

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Produce embeddings for a list of texts using google genai embed_content.
    Returns list of vectors aligned with texts.
    """
    loop = asyncio.get_running_loop()
    # run in thread to avoid blocking event loop if genai is sync
    result = await loop.run_in_executor(None, lambda: genai.embed_content(
        model=f"models/{EMBEDDING_MODEL}", content=texts, task_type="RETRIEVAL_QUERY"
    ))
    return result["embedding"]

async def search_qdrant_vectors(vectors: List[List[float]], filter: models.Filter, limit_per_vector:int=5):
    """
    Batch-search qdrant for each vector and return combined results.
    """
    requests = [models.SearchRequest(vector=v, filter=filter, limit=limit_per_vector) for v in vectors]
    # async search_batch from async_qdrant_client
    results = await async_qdrant_client.search_batch(collection_name="studywise_documents", requests=requests)
    # flatten
    flat = [item for sub in results for item in sub]
    # dedupe by payload text
    seen = set()
    unique = []
    for r in flat:
        text = (r.payload or {}).get("text")
        if text and text not in seen:
            seen.add(text)
            unique.append(r)
    return unique

async def store_context_to_qdrant(payload: dict, vector: List[float], id: str = None):
    """
    Optional: write a new point into Qdrant to act as cached context.
    payload: dict with text, chapter_id, user_id...
    vector: embedding vector for the payload
    """
    point = models.PointStruct(id=id, vector=vector, payload=payload) if id else models.PointStruct(vector=vector, payload=payload)
    # upsert expects list of points
    await async_qdrant_client.upsert(collection_name="studywise_documents", points=[point])
    logger.info("Stored context to Qdrant (maybe cache)")

# small helper to build Qdrant filter
def make_chapter_user_filter(chapter_id: str, user_id: str):
    return models.Filter(must=[
        models.FieldCondition(key="chapter_id", match=models.MatchValue(value=str(chapter_id))),
        models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))
    ])
