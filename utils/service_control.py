"""One admitted provider attempt; retry loops must sit outside these contexts."""
import asyncio
import logging
from contextlib import contextmanager, asynccontextmanager, nullcontext
import redis
from .circuit_breaker import ServiceUnavailable
from .deadline import timeout_for, check_deadline
from .llm_load_control import redis_call, maintain_lease

logger = logging.getLogger(__name__)

class Attempt:
    def __init__(self, breaker, permit):
        self.breaker, self.permit = breaker, permit
        self.started = False

    def timeout(self, default):
        return timeout_for(min(default, self.breaker.probe_seconds - 5))

    def keep_probe(self):
        if not self.permit.probe:
            return nullcontext()
        return maintain_lease(lambda: self.breaker.renew_probe(self.permit),
                              self.breaker.probe_seconds / 3)

    def before_call(self):
        check_deadline()
        try:
            if not self.breaker.validate(self.permit):
                raise ServiceUnavailable("Service circuit changed while waiting")
        except redis.RedisError as exc:
            raise ServiceUnavailable("Service control unavailable") from exc
        self.started = True

    def finish(self, outcome):
        try:
            accepted = self.breaker.finish(self.permit, outcome)
        except redis.RedisError as exc:
            if outcome == "success":
                raise ServiceUnavailable("Service control unavailable") from exc
            logger.warning("Could not record provider failure", exc_info=True)
            return
        if outcome == "success" and self.permit.probe and not accepted:
            raise ServiceUnavailable("Recovery probe lease expired")

@contextmanager
def sync_attempt(slots, breaker):
    attempt = Attempt(breaker, breaker.acquire())
    try:
        with attempt.keep_probe(), slots.sync_slot():
            attempt.before_call()
            yield attempt
            check_deadline()
    except BaseException as exc:
        attempt.finish("failure" if attempt.started and isinstance(exc, Exception) else "cancelled")
        if attempt.permit.probe and isinstance(exc, Exception):
            from .deadline import DeadlineExceeded
            if not isinstance(exc, DeadlineExceeded):
                raise ServiceUnavailable("Recovery probe failed") from exc
        raise
    else:
        attempt.finish("success")

async def _acquire(breaker):
    task = asyncio.create_task(asyncio.to_thread(breaker.acquire))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            permit = await task
            await redis_call(Attempt(breaker, permit).finish, "cancelled")
        except Exception:
            pass
        raise

@asynccontextmanager
async def async_attempt(slots, breaker):
    attempt = Attempt(breaker, await _acquire(breaker))
    try:
        # Includes admission queue time: a probe cannot outlive its Redis lease.
        async with asyncio.timeout(attempt.timeout(80)):
            async with slots.slot():
                await redis_call(attempt.before_call)
                yield attempt
                check_deadline()
    except BaseException as exc:
        await redis_call(attempt.finish, "failure" if attempt.started and isinstance(exc, Exception) else "cancelled")
        if attempt.permit.probe and isinstance(exc, Exception):
            from .deadline import DeadlineExceeded
            if not isinstance(exc, DeadlineExceeded):
                raise ServiceUnavailable("Recovery probe failed") from exc
        raise
    else:
        await redis_call(attempt.finish, "success")
