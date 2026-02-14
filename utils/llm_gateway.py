from .llm_wrapper import _call_llm_with_retry
from .circuit_breaker import CircuitBreaker
import logging
from llm_load_control import llm_slot, SystemOverloaded

logger =  logging.getLogger(__name__)

class LLMUnavailable(Exception):
    pass


async def ask_llm(groq_client,messages,*,model,json_mode=False,**kwargs):


    # circuit breaker gate

    if not CircuitBreaker.allow_request():
        logging.warning("LLM circuit OPEN - blocking  request")
        raise LLMUnavailable("LLM temporarily unavailable")


    try:

        async with llm_slot():

            response = await _call_llm_with_retry(groq_client,messages=messages,model=model,json_mode=json_mode,**kwargs)

            # sucess -> reset breaker
            CircuitBreaker.record_sucess()
            return response
    except SystemOverloaded:
        raise LLMUnavailable("system under heavy load")
    except Exception as e:

        CircuitBreaker.record_failure()

        logger.exception(
            "LLM call is failed",
                extra={
                "circuit_state": CircuitBreaker.state,
                "failures": CircuitBreaker.failures,

                } 
        )
        raise