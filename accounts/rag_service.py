# backend/rag_service.py
import logging
import asyncio
from typing import List, Union 
from .ai_clients import async_groq_client, async_qdrant_client
from utils.tei_embedding import TEIEmbeddingClient
from qdrant_client import models
QDRANT_COLLECTION_NAME = "studywise_documents"
tei_client = TEIEmbeddingClient()
logger = logging.getLogger(__name__)



async def embed_texts(texts: Union[str, List[str]]) -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]  
    embeddings = await tei_client.embed_texts(texts)
    return embeddings

async def search_qdrant_vectors(
    vectors: List[List[float]],
    filter: models.Filter | None,
    limit_per_vector: int = 5,
):
    if not vectors:
        logger.warning("No vectors provided")
        return []

    tasks = []

    for v in vectors:
        tasks.append(
            async_qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=v, 
                query_filter=filter,
                limit=limit_per_vector,
                with_payload=True,
                with_vectors=False,
            )
        )

    try:
        results = await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Qdrant query failed: {e}", exc_info=True)
        return []

    logger.info(f"Executed {len(results)} parallel searches")

    flat = []
    for res in results:
        flat.extend(res.points)  
    if not flat:
        return []

    # deduplicate
    seen = set()
    unique = []

    for r in flat:
        if r.id not in seen:
            seen.add(r.id)
            unique.append(r)

    # sort
    unique.sort(key=lambda x: getattr(x, "score", 0), reverse=True)

    return unique[:20]
    
    

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
