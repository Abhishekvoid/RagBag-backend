

import os
import time
import httpx
import logging
from typing import List
from .circuit_breaker import  tei_circuit_breaker
from .llm_load_control import  tei_slot_manager

from tenacity import (
    retry,
    wait_exponential_jitter,
    retry_if_exception_type,
    stop_after_attempt
)

logger = logging.getLogger(__name__)

TEI_URL = os.getenv("TEI_URL", "http://localhost:8080/embed")
TEI_TIMEOUT = float(os.getenv("TEI_TIMEOUT", 10.0))
EXPECTED_DIM = int(os.getenv("EMBEDDING_DIM", 768))

TEI_ERRORS = (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)

class EmbeddingServiceUnavailable(Exception):
    pass


class TEIEmbeddingClient:

    def __init__(self):
        self.cb = tei_circuit_breaker
        self.slot_manager = tei_slot_manager
        self.client  = httpx.AsyncClient(timeout=TEI_TIMEOUT, limits=httpx.Limits(max_connections=100))

    @retry (
            
            retry = retry_if_exception_type(TEI_ERRORS),
            wait=wait_exponential_jitter(initial=2, max=10),
            stop = stop_after_attempt(5),
        
    )
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:


        if not texts:
            return []
        
        if isinstance(texts, str):
            texts = [texts]

        if self.cb.is_open():
            raise EmbeddingServiceUnavailable("Tei circuit is open")
        
        async with self.slot_manager.slot():
            
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=TEI_TIMEOUT) as client:
                response = await self.client.post(
                        TEI_URL,
                        json = {"inputs": texts},
                    )

            response.raise_for_status()
            embeddings = response.json()

            if not isinstance(embeddings, list):
                    raise ValueError("invalid Tei response format")
                
            if embeddings and not isinstance(embeddings[0], list):
                    raise ValueError("tei returend invalid embedding structure")
                
            for vec in embeddings:
                if len(vec) != EXPECTED_DIM:
                        raise ValueError(
                            f"Embedding dimension mismatch: except{EXPECTED_DIM}, got{len(vec)}"
                        )
                    
            latency = (time.perf_counter() - start) * 1000
            logger.info(f"tei latency: {latency:.2f} ms | batch = {len(texts)}")

            self.cb.record_success()
            return embeddings
            
            
    
    async def close(self):
        await self.client.aclose()

        
    