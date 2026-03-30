# backend/rag_service.py

import logging
import asyncio
from typing import List, Union

from qdrant_client import models
from .ai_clients import async_qdrant_client
from utils.tei_embedding import TEIEmbeddingClient

tei_client = TEIEmbeddingClient()
logger = logging.getLogger(__name__)

QDRANT_COLLECTION = "studywise_documents"
MAX_RESULTS = 20  # prevent overload



# EMBEDDINGS

async def embed_texts(texts: Union[str, List[str]]) -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]

    embeddings = await tei_client.embed_texts(texts)

    if not embeddings or not isinstance(embeddings, list):
        raise ValueError("Invalid embeddings returned from TEI")

    return embeddings



# VECTOR SEARCH (PRODUCTION)

async def search_qdrant_vectors(
    vectors: List[List[float]],
    filter: models.Filter | None,
    limit_per_vector: int = 5,
):
    if not vectors:
        logger.warning("No vectors provided to search")
        return []

    tasks = []

    for v in vectors:
        tasks.append(
            async_qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=models.Query(vector=v), 
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


    # FLATTEN RESULTS
    flat = []
    for res in results:
        if hasattr(res, "points") and res.points:
            flat.extend(res.points)

    logger.info(f"Total raw results: {len(flat)}")

    if not flat:
        return []


    # DEDUPLICATE (by id)
    seen_ids = set()
    unique = []

    for r in flat:
        if r.id not in seen_ids:
            seen_ids.add(r.id)
            unique.append(r)


    # SORT BY SCORE
    unique.sort(key=lambda x: getattr(x, "score", 0), reverse=True)


    # LIMIT RESULTS (IMPORTANT)
    final_results = unique[:MAX_RESULTS]

    logger.info(f"Final results after dedup + limit: {len(final_results)}")

    return final_results


# STORE CONTEXT (OPTIONAL CACHE)

async def store_context_to_qdrant(
    payload: dict,
    vector: List[float],
    id: str = None,
):
    try:
        point = (
            models.PointStruct(id=id, vector=vector, payload=payload)
            if id
            else models.PointStruct(vector=vector, payload=payload)
        )

        await async_qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[point],
        )

        logger.info("Stored context to Qdrant")

    except Exception as e:
        logger.error(f"Failed to store context: {e}", exc_info=True)



#  FILTER BUILDER

def make_chapter_user_filter(chapter_id: str, user_id: str):
    return models.Filter(
        must=[
            models.FieldCondition(
                key="chapter_id",
                match=models.MatchValue(value=str(chapter_id)),
            ),
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=str(user_id)),
            ),
        ]
    )