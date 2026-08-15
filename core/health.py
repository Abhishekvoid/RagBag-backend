"""Liveness and readiness probes.

Two endpoints with deliberately different jobs:

  /ping/    liveness  — "is this process alive?" No I/O, no dependencies. This
                        is what a load balancer target group polls every few
                        seconds; making it touch Postgres would turn a database
                        blip into a rolling restart of every healthy task.

  /healthz  readiness — "can this process actually serve traffic?" Checks the
                        dependencies, and distinguishes the ones that make the
                        app useless from the ones that only degrade it.

HARD dependencies (Postgres, Redis) return 503: without them nothing works.
SOFT dependencies (S3, Celery, TEI embed/rerank) return 200 with a "degraded"
status. Reranking already falls back to vector+keyword ordering, and a queued
upload is not a reason to pull a web task out of service.

Responses never include exception messages — connection errors routinely embed
the host, user and password. Only the exception class name is reported.
"""

import logging
import os
from urllib.parse import urljoin, urlparse

from django.conf import settings
from django.db import connections
from django.http import JsonResponse

logger = logging.getLogger(__name__)

# Soft checks are best-effort; a slow dependency must not hold the probe open.
SOFT_TIMEOUT = 2.0


def ping(request):
    """Shallow liveness probe. Deliberately does no I/O."""
    return JsonResponse({"status": "ok"})


def _err(exc):
    """Report the failure class only — never the message, which leaks DSNs."""
    return {"status": "error", "error": type(exc).__name__}


def _check_postgres():
    with connections["default"].cursor() as cursor:
        cursor.execute("SELECT 1")
        cursor.fetchone()
    return {"status": "ok"}


def _check_redis():
    import redis

    client = redis.Redis.from_url(
        settings.REDIS_URL, socket_connect_timeout=SOFT_TIMEOUT
    )
    try:
        client.ping()
    finally:
        client.close()
    return {"status": "ok"}


def _check_s3():
    from django.core.files.storage import default_storage

    bucket = getattr(settings, "AWS_STORAGE_BUCKET_NAME", None)
    if not bucket:
        return {"status": "skipped", "reason": "not configured"}

    default_storage.connection.meta.client.head_bucket(Bucket=bucket)
    return {"status": "ok"}


def _check_celery():
    from core.celery import app

    replies = app.control.ping(timeout=SOFT_TIMEOUT)
    if not replies:
        return {"status": "error", "error": "no workers responded"}
    return {"status": "ok", "workers": len(replies)}


def _tei_health_url(endpoint_url):
    """TEI serves /health at the service root; our env vars point at /embed etc."""
    parsed = urlparse(endpoint_url)
    if not parsed.scheme:
        return None
    return urljoin(f"{parsed.scheme}://{parsed.netloc}", "/health")


def _check_tei(endpoint_url, name):
    import httpx

    health_url = _tei_health_url(endpoint_url)
    if not health_url:
        return {"status": "skipped", "reason": "not configured"}

    response = httpx.get(health_url, timeout=SOFT_TIMEOUT)
    response.raise_for_status()
    return {"status": "ok"}


# Registries hold the *name* of the check, not the function object, so the
# lookup happens at request time. Binding the callable here would freeze it at
# import and make the checks impossible to patch in tests.
HARD_CHECKS = {
    "postgres": ("_check_postgres", ()),
    "redis": ("_check_redis", ()),
}

SOFT_CHECKS = {
    "s3": ("_check_s3", ()),
    "celery": ("_check_celery", ()),
    "tei_embed": ("_check_tei", ("TEI_URL", "http://localhost:8080/embed", "embed")),
    "tei_rerank": (
        "_check_tei",
        ("RERANK_URL", "http://localhost:8081/rerank", "rerank"),
    ),
}


def _run(checks):
    results = {}
    failed = []
    for name, (func_name, args) in checks.items():
        try:
            func = globals()[func_name]
            if func_name == "_check_tei":
                env_var, default, label = args
                results[name] = func(os.getenv(env_var, default), label)
            else:
                results[name] = func(*args)

            if results[name].get("status") == "error":
                failed.append(name)
        except Exception as exc:  # noqa: BLE001 — a probe must never itself 500
            logger.warning("healthz check %s failed: %s", name, type(exc).__name__)
            results[name] = _err(exc)
            failed.append(name)
    return results, failed


def healthz(request):
    """Deep readiness probe. 503 only when a hard dependency is down."""
    hard, hard_failed = _run(HARD_CHECKS)
    soft, soft_failed = _run(SOFT_CHECKS)

    if hard_failed:
        status_text, http_status = "error", 503
    elif soft_failed:
        status_text, http_status = "degraded", 200
    else:
        status_text, http_status = "ok", 200

    return JsonResponse(
        {
            "status": status_text,
            "hard": hard,
            "soft": soft,
            "degraded": soft_failed,
        },
        status=http_status,
    )
