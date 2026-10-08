
import logging
import asyncio
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .ai_clients import (
    HYBRID_SEARCH_ENABLED,
    SPARSE_EMBED_MODEL,
    SPARSE_INPUT_PASSAGE,
    SPARSE_INPUT_QUERY,
    SPARSE_MAX_TOKENS_PER_SEQUENCE,
    get_pinecone_index,
    get_pinecone_sparse_index,
    pinecone_client,
)
from utils.embedding import EmbeddingClient
from utils.metrics.hybrid import hybrid_stats

embedding_client = EmbeddingClient()
logger = logging.getLogger(__name__)

RRF_K = 60


async def embed_texts(texts) -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]
    embeddings = await embedding_client.embed_texts(texts)
    return embeddings


# --- sparse embeddings -------------------------------------------------------

class SparseEmbeddingUnavailable(Exception):
    """The sparse embedding service could not be reached or returned garbage."""


def _sparse_params(input_type: str) -> Dict[str, Any]:
    return {
        "input_type": input_type,
        "truncate": "END",
        "max_tokens_per_sequence": SPARSE_MAX_TOKENS_PER_SEQUENCE,
    }


def _embed_sparse_blocking(texts: Sequence[str], input_type: str) -> List[Dict[str, list]]:
    """Blocking sparse embed. The Pinecone SDK is synchronous; callers thread it."""
    if pinecone_client is None:
        raise SparseEmbeddingUnavailable("PINECONE_API_KEY is not set")

    response = pinecone_client.inference.embed(
        model=SPARSE_EMBED_MODEL,
        inputs=list(texts),
        parameters=_sparse_params(input_type),
    )

    out = []
    for item in response.data:
        try:
            indices = item["sparse_indices"]
            values = item["sparse_values"]
        except (KeyError, TypeError) as e:
            raise SparseEmbeddingUnavailable(
                f"{SPARSE_EMBED_MODEL} returned no sparse vector "
                f"(is it a sparse model?): {e}"
            )
        if len(indices) != len(values):
            raise SparseEmbeddingUnavailable(
                f"malformed sparse vector: {len(indices)} indices, {len(values)} values"
            )
        out.append({"indices": list(indices), "values": list(values)})

    if len(out) != len(texts):
        raise SparseEmbeddingUnavailable(
            f"asked for {len(texts)} sparse vectors, got {len(out)}"
        )
    return out


async def embed_sparse(texts, *, input_type: str) -> List[Dict[str, list]]:

    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    if input_type not in (SPARSE_INPUT_PASSAGE, SPARSE_INPUT_QUERY):
        raise ValueError(
            f"input_type must be {SPARSE_INPUT_PASSAGE!r} or {SPARSE_INPUT_QUERY!r}, "
            f"got {input_type!r}"
        )

    try:
        return await asyncio.to_thread(_embed_sparse_blocking, texts, input_type)
    except SparseEmbeddingUnavailable:
        raise
    except Exception as e:
        raise SparseEmbeddingUnavailable(f"sparse embedding failed: {e}") from e


# --- result normalisation ----------------------------------------------------


def _clean_metadata(payload: dict) -> dict:
    """Pinecone metadata cannot hold null values, so drop any None entries."""
    return {k: v for k, v in (payload or {}).items() if v is not None}


def _to_result(match):
    """Normalise a Pinecone match into a mutable object with `.id`, `.score`,
    `.payload` — the shape the pipeline has always consumed."""
    if isinstance(match, dict):
        _id = match.get("id")
        score = match.get("score", 0.0)
        metadata = match.get("metadata") or {}
    else:
        _id = getattr(match, "id", None)
        score = getattr(match, "score", 0.0)
        metadata = getattr(match, "metadata", None) or {}
    return SimpleNamespace(id=_id, score=float(score or 0.0), payload=dict(metadata))


def _matches(response) -> list:
    raw = (
        response.get("matches")
        if isinstance(response, dict)
        else getattr(response, "matches", [])
    )
    return [_to_result(m) for m in (raw or [])]


# --- index queries -----------------------------------------------------------


def _query_dense_blocking(vector: List[float], filter: Optional[dict], top_k: int):
    return get_pinecone_index().query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        filter=filter,
    )


def _query_sparse_blocking(sparse_vector: Dict[str, list], filter: Optional[dict], top_k: int):
    return get_pinecone_sparse_index().query(
        sparse_vector=sparse_vector,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        filter=filter,
    )


async def search_dense_ranked(
    vectors: List[List[float]],
    filter: Optional[dict],
    limit_per_vector: int = 15,
) -> List[List[Any]]:
  
    if not vectors:
        return []

    tasks = [
        asyncio.to_thread(_query_dense_blocking, v, filter, limit_per_vector)
        for v in vectors
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    lists = []
    for i, response in enumerate(responses):
        if isinstance(response, BaseException):
            # One variant failing is survivable — the others still retrieve, and
            # RRF over fewer lists is a weaker ranking rather than a broken one.
            logger.warning("dense query %d/%d failed: %s", i + 1, len(vectors), response)
            continue
        lists.append(_matches(response))
    if not lists and responses:
        # Empty matches are valid evidence; zero successful requests are not.
        raise responses[0]
    return lists


async def search_sparse_ranked(
    sparse_vector: Dict[str, list],
    filter: Optional[dict],
    limit: int = 15,
) -> List[Any]:

    response = await asyncio.to_thread(
        _query_sparse_blocking, sparse_vector, filter, limit
    )
    return _matches(response)


# --- fusion ------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Any]],
    weights: Optional[Sequence[float]] = None,
    k: int = RRF_K,
) -> List[Any]:
    """Fuse ranked lists into one ordering."""
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"{len(weights)} weights for {len(ranked_lists)} lists"
        )

    scores: Dict[Any, float] = {}
    representative: Dict[Any, Any] = {}

    for ranked, weight in zip(ranked_lists, weights):
        seen_in_this_list = set()
        rank = 0
        for item in ranked:
            item_id = getattr(item, "id", None)
            if item_id is None:
                continue
         
            if item_id in seen_in_this_list:
                continue
            seen_in_this_list.add(item_id)

            rank += 1
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)

            if item_id not in representative:
                representative[item_id] = item

    fused = []
    for item_id, score in scores.items():
        item = representative[item_id]
        item.score = score
        fused.append(item)

    fused.sort(key=lambda r: (-r.score, str(r.id)))
    return fused


def modality_balanced_weights(dense_list_count: int, sparse_list_count: int):
    """Per-list weights that give dense and sparse one vote EACH"""
    dense_w = [1.0 / dense_list_count] * dense_list_count if dense_list_count else []
    sparse_w = [1.0 / sparse_list_count] * sparse_list_count if sparse_list_count else []
    return dense_w, sparse_w


# --- the public entry point --------------------------------------------------


async def hybrid_search(
    dense_vectors: List[List[float]],
    query_text: Optional[str],
    filter: Optional[dict],
    limit_per_vector: int = 15,
    top_n: int = 20,
) -> List[Any]:
    
    dense_error = None
    try:
        dense_lists = await search_dense_ranked(dense_vectors, filter, limit_per_vector)
    except Exception as exc:
        dense_error = exc
        dense_lists = []
        logger.warning("dense retrieval unavailable; trying sparse retrieval: %s", exc)

    sparse_lists: List[List[Any]] = []
    outcome = "disabled"

    if HYBRID_SEARCH_ENABLED and query_text:
        try:
            sparse_vec = await embed_sparse(query_text, input_type=SPARSE_INPUT_QUERY)
            hits = await search_sparse_ranked(sparse_vec[0], filter, limit_per_vector)
            if hits:
                sparse_lists = [hits]
                outcome = "hybrid_ok"
            else:
                # Service healthy, zero matches. Overwhelmingly this means the
                # chapter predates the sparse index and has not been through

                outcome = "sparse_empty"
                logger.info(
                    "sparse index returned no matches; answering dense-only "
                    "(chapter likely not backfilled — run reindex_hybrid)"
                )
        except Exception as e:
            outcome = "sparse_unavailable"
            logger.warning("sparse retrieval unavailable, falling back to dense: %s", e)

    hybrid_stats.record(outcome=outcome)

    if dense_error is not None and not sparse_lists:
        raise dense_error

    if not dense_lists and not sparse_lists:
        return []

    dense_w, sparse_w = modality_balanced_weights(len(dense_lists), len(sparse_lists))
    fused = reciprocal_rank_fusion(
        list(dense_lists) + list(sparse_lists),
        weights=list(dense_w) + list(sparse_w),
    )

    logger.info(
        "hybrid_search fused %d dense list(s) + %d sparse list(s) -> %d unique chunks (%s)",
        len(dense_lists), len(sparse_lists), len(fused), outcome,
    )
    return fused[:top_n]


async def search_vectors(
    vectors: List[List[float]],
    filter: Optional[dict],
    limit_per_vector: int = 5,
):
    """Dense-only retrieval, RRF-fused. Kept for callers that have no query text
    (and used by the eval harness to produce the dense-only baseline)."""
    dense_lists = await search_dense_ranked(vectors, filter, limit_per_vector)
    if not dense_lists:
        return []
    weights = [1.0 / len(dense_lists)] * len(dense_lists)
    return reciprocal_rank_fusion(dense_lists, weights=weights)[:20]


# --- writes ------------------------------------------------------------------


async def store_context(payload: dict, vector: List[float], id: str = None):
    """Optional: write a new point into Pinecone to act as cached context."""
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


def make_chapter_user_filter(chapter_id: str, user_id: str) -> Optional[dict]:
    """Use from sync code; None means no active documents, never no filter."""
    from .ingestion_versions import active_document_filter
    return active_document_filter(user_id, chapter_id)
