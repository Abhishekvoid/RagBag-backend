from tenacity import (

    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential_jitter,
    before_sleep_log
)
import logging
import groq
from groq import AsyncGroq

logger = logging.getLogger(__name__)



LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "text-embedding-004"
QDRANT_COLLECTION_NAME = "studywise_documents"

RETRYABLE_ERRORS = (

    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.RateLimitError,
    groq.InternalServerError,
)


@retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        wait=wait_exponential_jitter(initial=1, max=10), # Wait 1s, 2s, 4s... + jitter
        stop=stop_after_attempt(3), # Give up after 3 tries
        before_sleep=before_sleep_log(logger, logging.WARNING), # Log warnings on retry
        reraise=True # If it fails 3 times, raise the error so we can handle fallback
    )
async def _call_llm_with_retry(groq_client: AsyncGroq, messages, json_mode=False, **kwargs):

        params = {
            "messages": messages,
            **kwargs
            
        }
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        return await groq_client.chat.completions.create(**params)