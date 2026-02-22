
import asyncio
from contextlib import asynccontextmanager


MAX_CONCURRENT = 20
MAX_QUEUE = 100


class SystemOverload(Exception):
    pass

class SlotManager:

    def __init__(self, max_concurrent=MAX_CONCURRENT, max_queue=MAX_QUEUE):
    
    
        self._semaphore  = asyncio.Semaphore(max_concurrent)
        self._waiting  = 0
        self._lock = asyncio.Lock()
        self.max_queue = max_queue


    @asynccontextmanager
    async def slot(self):
        

        async with self._lock:
            if self._waiting >= self.max_queue:
                raise SystemOverload("system is overloading, try again later")
            self._waiting +=  1

        try:

            async with self._semaphore:
                yield
        finally:
            async with self._lock:
                self._waiting -= 1


tei_slot_manager = SlotManager(3, 5)
llm_slot_manager = SlotManager(20, 100)