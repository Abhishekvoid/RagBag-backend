"""A document's file -> persisted, page-structured canonical text."""
import io
import logging
import uuid

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

import PyPDF2
import fitz  # PyMuPDF — renders pages to images with no external binary (unlike poppler)

from .models import Document, DocumentPage
from .vision_ocr import (
    VISION_ENABLED, VISION_MAX_PAGES,
    page_needs_vision, reconstruct_page_markdown, strip_uncertainty_markers,
    VisionUnavailable,
)
from .realtime import push_ingestion_status, PHASE_PAGE

logger = logging.getLogger(__name__)


def per_page_layer_text(pdf_bytes: bytes) -> list:
    """The PDF's own text layer, one string per page (may be empty per page)."""
    out = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            out.append(page.extract_text() or "")
    except Exception as e:
        logger.warning("per_page_layer_text failed: %s", e)
    return out


def render_pdf_pages(pdf_bytes: bytes, dpi: int = 150) -> list:
    """Render each PDF page to PNG bytes (for storage + vision input)."""
    out = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            out.append(page.get_pixmap(dpi=dpi).tobytes("png"))
    return out


def store_page_image(document: Document, page_number: int, png_bytes: bytes) -> str:
    """Persist a rendered page image to storage; return its url."""
    name = f"{document.user_id}/pages/{document.id}/p{page_number}_{uuid.uuid4().hex[:8]}.png"
    saved = default_storage.save(name, ContentFile(png_bytes))
    return default_storage.url(saved)


def build_document_pages(document: Document) -> int:
    """Render/detect/reconstruct each page and upsert DocumentPage rows.
    Returns the number of pages processed."""
    with default_storage.open(document.file.name, "rb") as f:
        pdf_bytes = f.read()

    layer_texts = per_page_layer_text(pdf_bytes)
    images = render_pdf_pages(pdf_bytes)
    total = len(images)

    for idx, png in enumerate(images):
        page_number = idx + 1
        layer = layer_texts[idx] if idx < len(layer_texts) else ""

        use_vision = (
            VISION_ENABLED
            and page_number <= VISION_MAX_PAGES
            and page_needs_vision(layer)
        )
        image_url = store_page_image(document, page_number, png)

        if use_vision:
            try:
                md = reconstruct_page_markdown(png, page_number=page_number)
                source = DocumentPage.SOURCE_VISION
            except VisionUnavailable:
                md = layer.strip()
                source = DocumentPage.SOURCE_FALLBACK
        else:
            md = layer.strip()
            source = DocumentPage.SOURCE_LAYER

        DocumentPage.objects.update_or_create(
            document=document, page_number=page_number,
            defaults={"image_url": image_url, "reconstructed_md": md, "text_source": source},
        )
        push_ingestion_status(document.user_id, document.id, PHASE_PAGE,
                              page=page_number, total_pages=total)
    return total


def canonical_text_for_document(document: Document) -> str:
    """Concatenate pages' reconstructed markdown (markers stripped) for RAG/flashcards."""
    parts = [strip_uncertainty_markers(p.reconstructed_md)
             for p in document.pages.all() if p.reconstructed_md.strip()]
    return "\n\n".join(parts)
