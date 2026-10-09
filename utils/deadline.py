"""A monotonic budget propagated through async tasks and sync worker threads."""
import asyncio
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps


CHAT_BUDGET = 30.0
INGESTION_BUDGET = 900.0
_deadline = ContextVar("request_deadline", default=None)


class DeadlineExceeded(TimeoutError):
    pass


class RequestDeadline:
    def __init__(self, seconds):
        self.expires_at = time.monotonic() + seconds

    @property
    def remaining(self):
        return max(0.0, self.expires_at - time.monotonic())

    def check(self):
        if self.remaining <= 0:
            raise DeadlineExceeded("Request deadline exceeded")

    def timeout(self, default):
        remaining = self.remaining
        if remaining <= 0:
            raise DeadlineExceeded("Request deadline exceeded")
        return min(default, remaining)


def current_deadline():
    return _deadline.get()


@contextmanager
def deadline_scope(deadline):
    token = _deadline.set(deadline)
    try:
        yield deadline
    finally:
        _deadline.reset(token)


def check_deadline():
    if deadline := current_deadline():
        deadline.check()


def timeout_for(default):
    deadline = current_deadline()
    return deadline.timeout(default) if deadline else default


def stop_at_deadline(retry_state):
    """Tenacity calls stop after computing the next sleep, before sleeping."""
    deadline = current_deadline()
    return bool(deadline and deadline.remaining <= retry_state.upcoming_sleep)


def within_deadline(function):
    """Bound queue waits and retry sleeps as well as individual HTTP attempts."""
    @wraps(function)
    async def wrapped(*args, **kwargs):
        deadline = current_deadline()
        if deadline is None:
            return await function(*args, **kwargs)
        deadline.check()
        timer = asyncio.timeout(deadline.remaining)
        try:
            async with timer:
                result = await function(*args, **kwargs)
                deadline.check()
                return result
        except TimeoutError as exc:
            if timer.expired():
                raise DeadlineExceeded("Request deadline exceeded") from exc
            raise
    return wrapped


def ingestion_deadline(function):
    """A fresh budget per Celery execution, never inherited from a chat caller."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with deadline_scope(RequestDeadline(INGESTION_BUDGET)):
            return function(*args, **kwargs)
    return wrapped


@contextmanager
def database_deadline():
    """Bound SQL in the sync request thread, including sync_to_async ORM calls.

    Django's async_to_sync runs thread-sensitive ORM work back on this thread.
    PostgreSQL needs a server-side timeout: cancelling Python cannot stop SQL.
    Connection establishment also has a bounded connect_timeout in settings.
    """
    from django.db import connection

    previous = None
    if connection.vendor == "postgresql":
        with connection.cursor() as cursor:
            cursor.execute("SHOW statement_timeout")
            previous = cursor.fetchone()[0]

    def execute_bounded(execute, sql, params, many, context):
        if sql.lstrip().upper().startswith(("ROLLBACK", "RELEASE SAVEPOINT")):
            # Expiry must not prevent transaction cleanup after a failed write.
            return execute(sql, params, many, context)
        check_deadline()
        if previous is not None:
            milliseconds = max(1, int(timeout_for(5.0) * 1000))
            execute(f"SET statement_timeout = {milliseconds}", None, False, context)
        try:
            return execute(sql, params, many, context)
        except Exception as exc:
            if (getattr(exc, "pgcode", None) or getattr(exc.__cause__, "pgcode", None)) == "57014":
                raise DeadlineExceeded("Database operation timed out") from exc
            raise

    try:
        with connection.execute_wrapper(execute_bounded):
            yield
    finally:
        if previous is not None:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT set_config('statement_timeout', %s, false)", [previous])
            except Exception:
                # Never return a connection with an unknown timeout to the pool.
                connection.close()


def chat_deadline(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from rest_framework.response import Response
        with deadline_scope(RequestDeadline(CHAT_BUDGET)):
            try:
                with database_deadline():
                    return function(*args, **kwargs)
            except DeadlineExceeded:
                return Response({"error": "The request timed out. Please try again.",
                                 "retryable": True}, status=504)
    return wrapped
