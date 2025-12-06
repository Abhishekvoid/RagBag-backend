# utils/timing.py
import time
import logging
from functools import wraps

logger = logging.getLogger("core")  # or "rag" if you prefer

def time_sync(name=None):
    def decorator(fn):
        n = name or fn.__qualname__
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("TIMER %s %.2fms", n, elapsed_ms)
        return wrapper
    return decorator

def time_async(name=None):
    def decorator(fn):
        n = name or fn.__qualname__
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("TIMER %s %.2fms", n, elapsed_ms)
        return wrapper
    return decorator