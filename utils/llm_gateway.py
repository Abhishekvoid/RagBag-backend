from .llm_wrapper import _call_llm_with_retry
from .circuit_breaker import llm_circuit_breaker, redis_client
import logging
from .llm_load_control import SystemOverload, llm_slot_manager
from .metrics.cost import cost_tracker

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    pass


async def ask_llm(client, messages, *, model, json_mode=False, **kwargs):

    # circuit breaker gate
    if llm_circuit_breaker.is_open():
        logger.warning("LLM circuit OPEN - blocking request")
        raise LLMUnavailable("LLM temporarily unavailable")

    try:
        async with llm_slot_manager.slot():

            response = await _call_llm_with_retry(
                client,
                messages=messages,
                model=model,
                json_mode=json_mode,
                **kwargs
            )

            try:
                usage = getattr(response, "usage", None)

                if usage:
                    input_tokens = getattr(usage, "prompt_tokens", 0)
                    output_tokens = getattr(usage, "completion_tokens", 0)

                    cost_data = cost_tracker.track(
                        model= model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )

                    logger.info(
                        "llm_cost_tracker",
                        extra={
                            "stage": kwargs.get("stage", "unknown"),
                            "model": model,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cost_usd": cost_data["cost_usd"],
                            "daily_total": cost_data["daily_total"],
                        }
                    )
            except Exception:
                logger.warning("cost tracking failed", exc_info=True)
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