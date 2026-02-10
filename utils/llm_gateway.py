from .llm_wrapper import _call_llm_with_retry
from .circuit_breaker import llm_circuit_breaker
import logging

logger =  logging.getLogger(__name__)

class LLMUnavailable(Exception):
    pass


async def ask_llm(groq_client,messages,*,model,json_mode=False,**kwargs):


    # circuit breaker gate

    if not llm_circuit_breaker.allow_request():
        logging.warning("LLM circuit OPEN - blocking  request")
        raise LLMUnavailable("LLM temporarily unavailable")


    try:

        response = await _call_llm_with_retry(groq_client,messages=messages,model=model,json_mode=json_mode,**kwargs)

        # sucess -> reset breaker
        llm_circuit_breaker.record_sucess()
        return response
    
    except Exception as e:

        llm_circuit_breaker.record_failure()

        logger.exception(
            "LLM call is failed",
                extra={
                "circuit_state": llm_circuit_breaker.state,
                "failures": llm_circuit_breaker.failures,

                } 
        )
        raise