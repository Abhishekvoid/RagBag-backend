
import asyncio
from contextlib import asynccontextmanager


MAX_CONCURRENT = 20
MAX_QUEUE = 100

_semaphore  = asyncio.Semaphore(MAX_CONCURRENT)
_waiting  = 0
_lock = asyncio.Lock()

class SystemOverloaded(Exception):
    pass

async def llm_slot():
    global _waiting

    async with _lock:
        if _waiting >=MAX_QUEUE:
             raise SystemOverloaded("system is overladed, try again later")
        _waiting +=  1

    try:

        async with _semaphore:
            yield
    finally:
        async with _lock:
            _waiting -= 1