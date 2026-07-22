"""Vision-based page reconstruction: image/text -> clean canonical markdown.

Engine-agnostic: talks to any **OpenAI-compatible** chat endpoint. Defaults to a
local **Ollama** server running **MiniCPM-V** (free, no rate limits). Point it at
a different host/model with the ``VISION_*`` env vars and nothing else changes.

Only the pure helpers (`page_needs_vision`, `strip_uncertainty_markers`) and
config live at import time. The client is built lazily, so importing this module
never needs a server or key.
"""
import os
import re
import base64
import logging

logger = logging.getLogger(__name__)


def _flag(name: str, default: str) -> str:
    v = os.getenv(name, default)
    return v.strip() if v else default


VISION_ENABLED = _flag("VISION_ENABLED", "false").lower() in ("1", "true", "yes")
# Ollama model tag — `ollama pull minicpm-v` (use the 4.6 tag if you pulled it).
VISION_MODEL = _flag("VISION_MODEL", "minicpm-v")
# Ollama's OpenAI-compatible endpoint. Swap for OpenRouter/Groq/etc. to change host.
VISION_BASE_URL = _flag("VISION_BASE_URL", "http://localhost:11434/v1")
# Ollama ignores the key, but the OpenAI SDK requires a non-empty string.
VISION_API_KEY = _flag("VISION_API_KEY", "ollama")
VISION_MAX_PAGES = int(_flag("VISION_MAX_PAGES", "500"))

# A page's own text layer is "usable" when it has enough letters and isn't mostly
# symbol garbage (some PDFs carry junk embedded OCR — presence != quality).
_MIN_LAYER_CHARS = 40
_MIN_ALNUM_RATIO = 0.55

_MARKER_RE = re.compile(r"\[\?([^\]]*)\]")


def page_needs_vision(layer_text: str) -> bool:
    """True when the PDF's own text layer is too sparse or too garbled to trust,
    so the page should be sent to the vision model instead."""
    text = (layer_text or "").strip()
    if len(text) < _MIN_LAYER_CHARS:
        return True
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    return (alnum / len(text)) < _MIN_ALNUM_RATIO


def strip_uncertainty_markers(md: str) -> str:
    """Turn inline ``[?word]`` uncertainty markers back into plain ``word`` for
    embedding and quote-matching (the markers are a reader-only affordance)."""
    return _MARKER_RE.sub(r"\1", md or "")


# --------------------------------------------------------------------------- #
# Vision model call (OpenAI-compatible; Ollama + MiniCPM-V by default).
# --------------------------------------------------------------------------- #

_client = None
_client_failed = False

_SYSTEM = (
    "You transcribe a single scanned or photographed page (printed or handwritten) "
    "into clean, faithful Markdown. Rules: reproduce the page's real words — never "
    "invent facts, sentences, or data that are not visibly on the page. Use Markdown "
    "headings and bullet points to match the page's structure. For any word you cannot "
    "read with confidence, keep your best guess wrapped in [?like_this]; if a word is "
    "fully illegible, write [?]. Output only the Markdown transcription — no commentary, "
    "no page numbers of your own, no code fences around the whole thing."
)


class VisionUnavailable(Exception):
    """Raised when the vision endpoint can't be reached; callers must fall back."""


def _get_client():
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        raise VisionUnavailable("vision client previously failed to initialize")
    try:
        from openai import OpenAI
        _client = OpenAI(base_url=VISION_BASE_URL, api_key=VISION_API_KEY)
        return _client
    except Exception as e:  # missing sdk / bad config
        _client_failed = True
        raise VisionUnavailable(str(e))


def reconstruct_page_markdown(image_png_bytes: bytes, *, page_number: int) -> str:
    """Send one page image to the vision model; return clean Markdown with inline
    [?word] uncertainty markers. Raises VisionUnavailable on any failure."""
    client = _get_client()
    b64 = base64.standard_b64encode(image_png_bytes).decode("utf-8")
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            temperature=0,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": f"Transcribe page {page_number} to clean Markdown."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
        )
    except Exception as e:
        logger.error("vision reconstruct failed (page %s): %s", page_number, e)
        raise VisionUnavailable(str(e))

    return (resp.choices[0].message.content or "").strip()
