"""A document's file -> persisted, page-structured canonical text."""
import logging
import tempfile
import uuid
import io
import os
from contextlib import closing
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

import fitz  # PyMuPDF — renders pages to images with no external binary (unlike poppler)

from .models import Document, DocumentPage
from utils.deadline import check_deadline, timeout_for, DeadlineExceeded
from .vision_ocr import (
    VISION_ENABLED, VISION_MAX_PAGES,
    page_needs_vision, reconstruct_page_markdown, strip_uncertainty_markers,
    VisionUnavailable,
)
from .realtime import push_ingestion_status, PHASE_PAGE

logger = logging.getLogger(__name__)


class DocumentOversizedError(ValueError):
    """Permanent input rejection: retrying cannot make the PDF smaller."""


def render_pdf_pages(pdf_path, dpi: int = 150):
    """Yield one bounded RGB page at a time; close the generator on failure."""
    with fitz.open(pdf_path) as pdf:
        total = len(pdf)
        limit = settings.MAX_PAGES_PER_DOCUMENT
        if total > limit:
            raise DocumentOversizedError(f"PDF has {total} pages; the limit is {limit}.")
        for number in range(total):
            check_deadline()
            page = pdf.load_page(number)
            pix = None
            try:
                scale = min(max(1, dpi), 150) / 72
                scale = min(scale, 2048 / max(page.rect.width, page.rect.height))
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale),
                                      colorspace=fitz.csRGB, alpha=False)
                yield number + 1, total, page.get_text(), pix.tobytes("png"), bool(page.get_image_info())
            finally:
                pix = None
                page = None


def store_page_image(document: Document, page_number: int, png_bytes: bytes) -> str:
    """Persist a rendered page image; return the durable storage key."""
    name = f"{document.user_id}/pages/{document.id}/p{page_number}_{uuid.uuid4().hex[:8]}.png"
    return default_storage.save(name, ContentFile(png_bytes))


def local_ocr(png: bytes) -> str:
    """Tesseract is an existing image dependency; kill it when the budget expires."""
    import pytesseract
    from PIL import Image
    with Image.open(io.BytesIO(png)) as image:
        result = pytesseract.image_to_string(
            image, lang=os.getenv("OCR_LANGUAGES", "eng"),
            timeout=timeout_for(float(os.getenv("OCR_TIMEOUT", "20"))),
        )
    check_deadline()
    return result.strip()


def readable_length(text):
    return sum(c.isalnum() for c in strip_uncertainty_markers(text))


def extract_page(layer, png, page_number, has_image):
    md, source, reason = layer.strip(), DocumentPage.SOURCE_LAYER, ""
    if page_needs_vision(layer):
        if VISION_ENABLED and page_number <= VISION_MAX_PAGES:
            try:
                md = reconstruct_page_markdown(png, page_number=page_number)
                source = DocumentPage.SOURCE_VISION
            except VisionUnavailable:
                source = DocumentPage.SOURCE_FALLBACK
        if has_image and (readable_length(md) < 20 or page_needs_vision(md)):
            try:
                recovered = local_ocr(png)
                if readable_length(recovered) > readable_length(md):
                    md, source = recovered, DocumentPage.SOURCE_OCR
            except DeadlineExceeded:
                raise
            except Exception:
                # No provider exception text enters the persisted/public warning.
                check_deadline()
                reason = "Local OCR unavailable or timed out"
                logger.warning("Local OCR failed for page %s", page_number, exc_info=True)
    needs_review = has_image and readable_length(md) < 20
    return md, source, needs_review, (reason or "Too little readable text after extraction") if needs_review else ""


def build_document_pages(document: Document, *, version=None, check_lease=lambda: None) -> int:
    """Render/detect/reconstruct each page and upsert DocumentPage rows.
    Returns the number of pages processed."""
    total = 0
    version = document.active_version if version is None else version
    with tempfile.TemporaryDirectory(prefix="studywise-pdf-") as directory:
        pdf_path = Path(directory) / "document.pdf"
        with default_storage.open(document.file.name, "rb") as source, pdf_path.open("wb") as target:
            for chunk in iter(lambda: source.read(64 * 1024), b""):
                check_deadline()
                target.write(chunk)
        with closing(render_pdf_pages(pdf_path)) as pages:
            for page_number, total, layer, png, has_image in pages:
                check_deadline()
                check_lease()
                object_key = store_page_image(document, page_number, png)
                md, text_source, needs_review, reason = extract_page(layer, png, page_number, has_image)
                check_lease()
                DocumentPage.objects.update_or_create(
                    document=document, version=version, page_number=page_number,
                    defaults={"s3_object_key": object_key, "image_url": "",
                              "reconstructed_md": md, "text_source": text_source,
                              "needs_review": needs_review, "review_reason": reason},
                )
                del png
                push_ingestion_status(document.user_id, document.id, PHASE_PAGE,
                                      page=page_number, total_pages=total)
    return total


def canonical_text_for_document(document: Document, *, version=None) -> str:
    """Concatenate pages' reconstructed markdown (markers stripped) for RAG/flashcards."""
    version = document.active_version if version is None else version
    parts = [strip_uncertainty_markers(p.reconstructed_md)
             for p in document.pages.filter(version=version, needs_review=False) if p.reconstructed_md.strip()]
    return "\n\n".join(parts)
