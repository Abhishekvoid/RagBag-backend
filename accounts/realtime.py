"""Real-time ingestion status contract.

One structured WebSocket event type, `ingestion_status`, is pushed to the
per-user Channels group `user_{id}` at each pipeline phase. The frontend
reduces these into an ingestions map keyed by document_id.
"""
import logging
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)

PHASE_READING = "reading"
PHASE_NAMING = "naming"
PHASE_CHUNKING = "chunking"
PHASE_EMBEDDING = "embedding"
PHASE_STORING = "storing"
PHASE_READY = "ready"
PHASE_FAILED = "failed"


def build_ingestion_status(document_id, phase, *, chapter_id=None, title=None,
                           batch=None, total_batches=None, error=None) -> dict:
    """Pure builder for an ingestion_status payload. Optional keys are
    omitted when None so the frontend reducer can treat missing == unchanged."""
    payload = {
        "type": "ingestion_status",
        "document_id": str(document_id),
        "phase": phase,
    }
    if chapter_id is not None:
        payload["chapter_id"] = str(chapter_id)
    if title is not None:
        payload["title"] = title
    if batch is not None:
        payload["batch"] = batch
    if total_batches is not None:
        payload["total_batches"] = total_batches
    if error is not None:
        payload["error"] = error
    return payload


def push_ingestion_status(user_id, document_id, phase, **extra) -> dict:
    """Build and broadcast an ingestion_status event to the user's group.
    Never raises — a telemetry failure must not break ingestion."""
    payload = build_ingestion_status(document_id, phase, **extra)
    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"user_{user_id}",
            {"type": "send_notification", "data": payload},
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("push_ingestion_status failed (%s): %s", phase, e)
    return payload
