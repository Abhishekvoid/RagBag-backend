"""Embedding client. Talks to whichever service holds the embedding model.

The model is fixed — BAAI/bge-small-en-v1.5, 384 dimensions — but WHERE it runs
is a deployment choice, and the two profiles differ:

  LEAN  a managed provider over HTTPS (no inference container to run, and no
        amd64-only image to pin the whole stack off Graviton)
  FULL  a self-hosted TEI container, as before

Both speak HTTP and return float vectors; they disagree only on the JSON
envelope. EMBEDDING_PROVIDER selects the adapter, so switching profiles is an
environment change rather than a code change.

The 384-dimension check is not a formality. Vectors of the wrong width would be
accepted by nothing downstream — the Pinecone index is created with a fixed
dimension — so a mismatch is a misconfigured provider or a silently swapped
model, and it must fail loudly rather than corrupt the index.
"""

import asyncio
import logging
import os
import time
from typing import Any, List

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .circuit_breaker import tei_circuit_breaker
from .llm_load_control import tei_slot_manager
from .token_budget import BGE_MAX_TOKENS, bge_token_upper_bound
from .deadline import stop_at_deadline, timeout_for, within_deadline

logger = logging.getLogger(__name__)

# tei        self-hosted TEI          {"inputs": [...]}  -> [[float, ...], ...]
# cloudflare Cloudflare Workers AI    {"text":   [...]}  -> {"result": {"data": [[...]]}}
# openai     OpenAI-compatible        {"input":  [...]}  -> {"data": [{"embedding": [...]}]}
EMBEDDING_PROVIDER = (os.getenv("EMBEDDING_PROVIDER") or "tei").strip().lower()

# POOLING IS NOT COSMETIC — it decides whether vectors from two providers live in
# the same space at all.
#
# bge-small-en-v1.5 ships `1_Pooling/pooling_config.json` with CLS pooling, and
# TEI honours it, so every vector already in the Pinecone index is CLS-pooled.
# Cloudflare Workers AI exposes the same weights but defaults to MEAN pooling,
# and its own docs state the two are not compatible with each other. Left at the
# default, Cloudflare returns 384 well-formed floats that pass every shape and
# dimension check in this module and are still wrong — silently unrelated to the
# indexed vectors, degrading retrieval rather than failing it.
#
# So it is sent explicitly on every request, never inferred from a default.
EMBEDDING_POOLING = (os.getenv("EMBEDDING_POOLING") or "cls").strip().lower()

EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8080/embed")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", 10.0))
EXPECTED_DIM = int(os.getenv("EMBEDDING_DIM", 384))

SUPPORTED_PROVIDERS = ("tei", "cloudflare", "openai")

EMBEDDING_ERRORS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.HTTPStatusError,
)


class EmbeddingServiceUnavailable(Exception):
    pass


class EmbeddingDimensionMismatch(ValueError):
    """The provider returned vectors of the wrong width. Never coerce these."""


class EmbeddingPoolingMismatch(ValueError):
    """The provider pooled differently than we asked. Never accept these."""


class EmbeddingInputTooLong(ValueError):
    """A caller tried to embed more than the model's 512-token window.

    Neither provider refuses these — TEI and Cloudflare both truncate to 512 and
    return HTTP 200, so an over-long input produces a plausible vector of the
    first half of the text and no error anywhere. That silence is the problem:
    the caller cannot tell a complete embedding from a beheaded one.

    Raised before the request goes out, so the truncation never happens. It is
    deliberately not in EMBEDDING_ERRORS — retrying identical oversized input
    would fail identically five times and burn the backoff for nothing.
    """


def build_request(texts: List[str], provider: str = None) -> dict:
    provider = provider or EMBEDDING_PROVIDER

    if provider == "cloudflare":
        # See EMBEDDING_POOLING above: omitting this yields mean-pooled vectors
        # that are indistinguishable from correct ones by shape alone.
        return {"text": texts, "pooling": EMBEDDING_POOLING}

    if provider == "openai":
        body = {"input": texts}
        if EMBEDDING_MODEL:
            body["model"] = EMBEDDING_MODEL
        return body

    return {"inputs": texts}


def parse_response(payload: Any) -> List[List[float]]:
    """Normalise every supported envelope down to a plain list of vectors."""
    # TEI: the response is already the bare list.
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        # Cloudflare wraps everything in an API envelope.
        if payload.get("success") is False:
            raise ValueError("embedding provider reported failure")

        result = payload.get("result")
        if isinstance(result, dict) and isinstance(result.get("data"), list):
            return result["data"]

        data = payload.get("data")
        if isinstance(data, list):
            # OpenAI style: [{"embedding": [...]}, ...]
            if data and isinstance(data[0], dict):
                return [item["embedding"] for item in data]
            return data

    raise ValueError("unrecognised embedding response shape")


def check_pooling(payload: Any, provider: str = None) -> None:
    """Fail if the provider echoes a pooling mode other than the one requested.

    Cloudflare returns `result.pooling`. Silence is not consent — a provider that
    stops echoing the field, or starts ignoring the parameter, must not quietly
    downgrade us to mean pooling, so a mismatch raises rather than warns. TEI has
    no such field (pooling is baked into the model config), so absence is fine.
    """
    provider = provider or EMBEDDING_PROVIDER

    if provider != "cloudflare" or not isinstance(payload, dict):
        return

    result = payload.get("result")
    if not isinstance(result, dict):
        return

    echoed = result.get("pooling")
    if echoed is not None and str(echoed).strip().lower() != EMBEDDING_POOLING:
        raise EmbeddingPoolingMismatch(
            f"provider pooled with '{echoed}', requested '{EMBEDDING_POOLING}'; "
            "vectors would not match the index"
        )


def auth_headers() -> dict:
    """Bearer header, or nothing for an unauthenticated self-hosted TEI."""
    if EMBEDDING_API_KEY:
        return {"Authorization": f"Bearer {EMBEDDING_API_KEY}"}
    return {}


class EmbeddingClient:
    def __init__(self):
        self.cb = tei_circuit_breaker
        self.slot_manager = tei_slot_manager
        # The httpx client is created PER EVENT LOOP, lazily, and never in
        # __init__.
        #
        # An AsyncClient binds its connection pool to whichever loop first uses
        # it. Ingestion calls `async_to_sync(client.embed_texts)(batch)` once per
        # batch, and async_to_sync spins up a NEW loop for each call and closes
        # it on return. So batch 1 filled the pool on loop A, loop A closed, and
        # batch 2 reached for those same connections and died with "Event loop
        # is closed" — every batch after the first, on every multi-batch
        # document. Single-batch documents ingested fine, which is why this hid
        # for so long: it only bit documents big enough to matter.
        #
        # Keying the pool by the running loop means each async_to_sync call gets
        # a pool that belongs to its own loop, while a long-lived loop (ASGI,
        # the query path) still reuses one pool across every request.
        # Values are (loop, client). The LOOP is kept, not just its id, because
        # eviction has to ask `loop.is_closed()` — an httpx client whose loop
        # died still reports `is_closed == False`, so pruning on the client
        # alone never fires and a 40-batch document would leak 40 pools.
        self._clients: dict[int, tuple] = {}

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=EMBEDDING_TIMEOUT, limits=httpx.Limits(max_connections=100)
        )

    def _get_client(self) -> httpx.AsyncClient:
        try:
            loop = asyncio.get_running_loop()
            key = id(loop)
        except RuntimeError:
            # Reached from synchronous code — `EmbeddingClient().client` before
            # any await. There is no loop to key on yet, so the pool is parked
            # under None and adopted by whichever loop first uses it (below).
            # This is the path tests take when they patch `.client.post` from a
            # plain test method, and the patched object must be the very one the
            # subsequent coroutine uses.
            loop, key = None, None

        entry = self._clients.get(key)
        if entry is not None and not entry[1].is_closed:
            return entry[1]

        if key is not None and entry is None:
            parked = self._clients.get(None)
            if parked is not None and not parked[1].is_closed:
                self._clients[key] = (loop, self._clients.pop(None)[1])
                return self._clients[key][1]

        client = self._new_client()
        self._clients[key] = (loop, client)
        # Evict pools whose loop has since closed, so a long-running worker does
        # not accumulate one per batch.
        for dead in [
            k
            for k, (lp, c) in self._clients.items()
            if k != key and (c.is_closed or (lp is not None and lp.is_closed()))
        ]:
            self._clients.pop(dead, None)
        return client

    @property
    def client(self) -> httpx.AsyncClient:
        """The connection pool for the caller's event loop.

        Kept as `.client` so every existing call site — and the tests that patch
        `.client.post` — work unchanged.
        """
        return self._get_client()

    @within_deadline
    @retry(
        retry=retry_if_exception_type(EMBEDDING_ERRORS),
        wait=wait_exponential_jitter(initial=2, max=10),
        stop=stop_after_attempt(5) | stop_at_deadline,
    )
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if isinstance(texts, str):
            texts = [texts]

        if await asyncio.to_thread(self.cb.is_open):
            raise EmbeddingServiceUnavailable("embedding circuit is open")

        # The last line of defence, and the reason the call-site audit is a
        # one-liner: every path into the embedding API passes through here, so
        # no caller — present or future — can send input the model would
        # silently truncate. Callers are still expected to split or reject
        # up front; reaching this raise means one of them forgot.
        for position, text in enumerate(texts):
            bound = bge_token_upper_bound(text)
            if bound > BGE_MAX_TOKENS:
                raise EmbeddingInputTooLong(
                    f"input {position} of {len(texts)} may reach {bound} tokens, "
                    f"over the model's {BGE_MAX_TOKENS}-token limit; split it "
                    f"with utils.token_budget.split_for_embedding first"
                )

        async with self.slot_manager.slot():
            start = time.perf_counter()

            response = await self.client.post(
                EMBEDDING_URL,
                json=build_request(texts),
                headers=auth_headers(),
                timeout=timeout_for(EMBEDDING_TIMEOUT),
            )

            # Log the status only. Response bodies from an auth failure can echo
            # back the credential that was sent.
            if response.status_code >= 400:
                logger.warning(
                    "embedding provider returned HTTP %s", response.status_code
                )
            response.raise_for_status()

            payload = response.json()
            check_pooling(payload)
            embeddings = parse_response(payload)

            if not isinstance(embeddings, list):
                raise ValueError("invalid embedding response format")

            if embeddings and not isinstance(embeddings[0], list):
                raise ValueError("provider returned invalid embedding structure")

            for vec in embeddings:
                if len(vec) != EXPECTED_DIM:
                    raise EmbeddingDimensionMismatch(
                        f"Embedding dimension mismatch: expected {EXPECTED_DIM}, "
                        f"got {len(vec)}"
                    )

            latency = (time.perf_counter() - start) * 1000
            logger.info(
                "embedding latency: %.2f ms | batch=%d | provider=%s",
                latency,
                len(texts),
                EMBEDDING_PROVIDER,
            )

            await asyncio.to_thread(self.cb.record_success)
            return embeddings

    async def close(self):
        """Close every per-loop pool, not just the current loop's.

        Going through `self.client` here would CREATE a pool for the calling
        loop and then close that, leaving the pools that actually hold sockets
        open — the opposite of what the caller asked for.
        """
        for _loop, client in list(self._clients.values()):
            if not client.is_closed:
                await client.aclose()
        self._clients.clear()


# Backwards-compatible alias: the class was TEI-specific before it grew adapters.
TEIEmbeddingClient = EmbeddingClient
