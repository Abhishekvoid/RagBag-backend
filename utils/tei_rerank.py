"""Reranking via a self-hosted TEI container (POST /rerank).

Mirrors EmbeddingClient: async httpx with bounded retries. Reranking is an
enhancement, never a hard dependency — any failure returns None and the caller
falls back to the existing vector + keyword ordering, so a query never fails
because the rerank service is down.

RERANK_URL unset means DISABLED, not "try localhost". The LEAN profile ships no
reranker container (bge-reranker-base needs ~5.5 GB of RSS, more than the whole
instance), so the default has to be "off" rather than a localhost address that
would burn three retries and a timeout on every single query.
"""
import os
import logging
import httpx
from tenacity import (
    retry,
    wait_exponential_jitter,
    retry_if_exception_type,
    stop_after_attempt,
)

logger = logging.getLogger(__name__)

RERANK_URL = (os.getenv("RERANK_URL") or "").strip()
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", 10.0))

RERANK_ERRORS = (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)


class TEIRerankClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=RERANK_TIMEOUT,
            limits=httpx.Limits(max_connections=50),
        )

    @retry(
        retry=retry_if_exception_type(RERANK_ERRORS),
        wait=wait_exponential_jitter(initial=1, max=5),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _post(self, query: str, texts: list) -> list:
        response = await self.client.post(
            RERANK_URL,
            json={"query": query, "texts": texts},
        )
        response.raise_for_status()
        return response.json()

    async def rerank(self, query: str, texts: list):
        """Return relevance scores aligned to `texts` order, or None on failure.

        TEI /rerank returns [{"index": i, "score": s}, ...]; we scatter the
        scores back into input order so the caller can zip them onto its results.
        """
        if not texts:
            return []
        if not RERANK_URL:
            # Disabled by configuration (LEAN). Not an error, and not worth a
            # log line per query — the caller's fallback is the intended path.
            return None
        try:
            data = await self._post(query, texts)
            scores = [0.0] * len(texts)
            for item in data:
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(scores):
                    scores[idx] = float(item.get("score", 0.0))
            return scores
        except Exception as e:
            logger.warning(f"Rerank service unavailable, falling back: {e}")
            return None


# Cheap to construct; does no network I/O until first use.
rerank_client = TEIRerankClient()
