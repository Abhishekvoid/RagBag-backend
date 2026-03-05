from .llm_wrapper import _call_llm_with_retry
from .circuit_breaker import llm_circuit_breaker, redis_client
import logging
from .llm_load_control import SystemOverload, llm_slot_manager

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    pass


async def ask_llm(groq_client, messages, *, model, json_mode=False, **kwargs):

    # circuit breaker gate
    if llm_circuit_breaker.is_open():
        logger.warning("LLM circuit OPEN - blocking request")
        raise LLMUnavailable("LLM temporarily unavailable")

    try:
        async with llm_slot_manager.slot():

            response = await _call_llm_with_retry(
                groq_client,
                messages=messages,
                model=model,
                json_mode=json_mode,
                **kwargs
            )

            # success → reset breaker
            llm_circuit_breaker.record_success()
            return response

    except SystemOverload:
        raise LLMUnavailable("system under heavy load")

    except Exception:

        llm_circuit_breaker.record_failure()

        logger.exception(
            "LLM call failed",
            extra={
                "circuit_state": redis_client.hget(llm_circuit_breaker.key, "state"),
                "failures": redis_client.hget(llm_circuit_breaker.key, "failures"),
            }
        )

        raise