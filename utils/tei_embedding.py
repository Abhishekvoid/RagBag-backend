

import os
import time
import httpx
import logging
from typing import List
from circuit_breaker import  tei_circuit_breaker
from llm_load_control import  tei_slot_manager

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
        

        if self.cb.is_open():
            raise EmbeddingServiceUnavailable("Tei circuit is open")
        
        async with self.slot_manager.slot():
            
            start = time.perf_counter()
            try:
                response = await self.client.post(
                        TEI_URL,
                        json={"inputs": texts},
                )

                response.raise_for_status()
                embeddings = response.json()
                result = [embeddings] if len(texts) == 1 else embeddings

                for vec in result:
                    if len(vec) != EXPECTED_DIM:
                        raise ValueError (
                            f"Embedding dimension mismatch: expected {EXPECTED_DIM}, got {len(vec)}"
                        )
                
                latency = (time.perf_counter() - start) * 1000
                logger.info(
                    f"TEI latency: {latency:.2f} ms | batch={len(texts)}"
                )
                
                self.cb.record_success()
                return result
            
            except TEI_ERRORS as e:
                self.cb.record_failure()
                logger.error(f"TEI failed: {e}")
                raise 
             
            except Exception as e:
                logger.error(f"unexpected TEI failure: {e}", exc_info=True)
                raise EmbeddingServiceUnavailable("TEI unavailable")
    
    async def close(self):
        await self.client.aclose()

        
    