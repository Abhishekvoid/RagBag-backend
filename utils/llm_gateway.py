from .llm_wrapper import _call_llm_with_retry
from .circuit_breaker import ServiceUnavailable
import logging
from .metrics.cost import cost_tracker
from .deadline import DeadlineExceeded, within_deadline

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    pass


@within_deadline
async def ask_llm(client, messages, *, model, json_mode=False, **kwargs):

    try:
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
        return response

    except DeadlineExceeded:
        raise

    except ServiceUnavailable:
        raise LLMUnavailable("system under heavy load")

    except Exception:


        logger.exception(
            "LLM call failed",
            extra={
                "service": "llm",
            }
        )

        raise
