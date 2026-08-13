from tenacity import (

    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential_jitter,
    before_sleep_log
)
import logging
import openai


logger = logging.getLogger(__name__)


EMBEDDING_MODEL = "gemini-embedding-001"

RETRYABLE_ERRORS = (

    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class EmptyCompletion(Exception):
    """Provider returned HTTP 200 but no message content.

    This is not hypothetical: asking an OpenRouter model that does not support
    `response_format` for JSON mode returns a well-formed 200 whose
    `message.content` is None. Treating that as success would push None into
    json.loads() far away from the cause, so it is raised as a failure here and
    handled like any other provider error (retry, then fall back).
    """


def _content_or_raise(response):
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise EmptyCompletion(f"malformed completion response: {exc!r}") from exc
    if content is None:
        raise EmptyCompletion("provider returned 200 with content=None")
    return response


@retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS + (EmptyCompletion,)),
        wait=wait_exponential_jitter(initial=1, max=10), # Wait 1s, 2s, 4s... + jitter
        stop=stop_after_attempt(3), # Give up after 3 tries
        before_sleep=before_sleep_log(logger, logging.WARNING), # Log warnings on retry
        reraise=True # If it fails 3 times, raise so the caller/breaker sees it
    )
async def _call_llm_with_retry(client, messages, json_mode=False, **kwargs):
    """Call the LLM provider, retrying transient failures and empty completions.

    There is no secondary provider: OpenRouter is the only one configured, so an
    exhausted retry budget propagates to the caller and trips the circuit
    breaker in llm_gateway.
    """
    params = {
        "messages": messages,
        **kwargs

    }

    if json_mode:
        params["response_format"] = {"type": "json_object"}

    return _content_or_raise(await client.chat.completions.create(**params))
