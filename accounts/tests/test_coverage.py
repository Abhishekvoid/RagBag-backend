import io
from unittest import mock
from django.test import SimpleTestCase, TestCase
from PIL import Image
from asgiref.sync import async_to_sync
from accounts import page_pipeline as pages
from accounts.models import Document, DocumentPage, CustomUserModel, Chapter, DocumentIndexVersion
from accounts.coverage import record_coverage, COVERAGE_ADVISORY
from accounts.rag_pipeline import RagPipeline
from accounts.serializers import DocumentSerializer, DocumentPageSerializer
from utils.deadline import RequestDeadline, deadline_scope, DeadlineExceeded

class OCRFallbackTests(SimpleTestCase):
    def test_inline_pdf_image_is_detected_even_without_image_resources(self):
        import fitz
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inline.pdf"
            with fitz.open() as pdf:
                page = pdf.new_page(width=72, height=72)
                page.insert_text((10, 20), "temporary")
                pdf.update_stream(page.get_contents()[0],
                    b'q 50 0 0 50 0 0 cm BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI Q')
                self.assertEqual(page.get_images(), [])
                pdf.save(path)
            rendered = list(pages.render_pdf_pages(path))
        self.assertEqual(rendered[0][2], "")
        self.assertTrue(rendered[0][4])

    def test_image_ocr_runs_for_disabled_failed_empty_and_capped_vision(self):
        for enabled, limit, vision in [(False, 500, ""), (True, 500, pages.VisionUnavailable("down")),
                                       (True, 500, ""), (True, 0, "")]:
            with self.subTest(enabled=enabled, limit=limit, vision=vision), \
                 mock.patch.object(pages, "VISION_ENABLED", enabled), \
                 mock.patch.object(pages, "VISION_MAX_PAGES", limit), \
                 mock.patch.object(pages, "reconstruct_page_markdown", side_effect=vision if isinstance(vision, Exception) else None, return_value=vision), \
                 mock.patch.object(pages, "local_ocr", return_value="Readable OCR equation and coaching notes") as ocr:
                text, source, review, reason = pages.extract_page("", b"png", 1, True)
                self.assertEqual(source, "ocr")
                self.assertFalse(review)
                self.assertEqual(reason, "")
                ocr.assert_called_once()

    def test_unreadable_image_flags_review_but_blank_page_does_not(self):
        with mock.patch.object(pages, "VISION_ENABLED", False), \
             mock.patch.object(pages, "local_ocr", return_value="??? a") as ocr:
            self.assertTrue(pages.extract_page("", b"png", 1, True)[2])
            self.assertFalse(pages.extract_page("", b"png", 2, False)[2])
            ocr.assert_called_once()

    def test_missing_ocr_is_reviewable_and_deadline_is_not_swallowed(self):
        with mock.patch.object(pages, "VISION_ENABLED", False), \
             mock.patch.object(pages, "local_ocr", side_effect=RuntimeError("missing binary")):
            self.assertEqual(pages.extract_page("", b"png", 1, True)[3], "Local OCR unavailable or timed out")
        with mock.patch.object(pages, "VISION_ENABLED", False), \
             mock.patch.object(pages, "local_ocr", side_effect=DeadlineExceeded("expired")):
            with self.assertRaises(DeadlineExceeded):
                pages.extract_page("", b"png", 1, True)

    def test_tesseract_receives_real_image_and_bounded_timeout(self):
        png = io.BytesIO()
        Image.new("RGB", (100, 100), "white").save(png, format="PNG")
        with deadline_scope(RequestDeadline(2)), mock.patch("pytesseract.image_to_string", return_value=" Notes ") as ocr:
            self.assertEqual(pages.local_ocr(png.getvalue()), "Notes")
        self.assertEqual(ocr.call_args.args[0].size, (100, 100))
        self.assertGreater(ocr.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(ocr.call_args.kwargs["timeout"], 2)

class CoveragePersistenceTests(TestCase):
    def setUp(self):
        self.user = CustomUserModel.objects.create_user(email="coverage@test.com", name="C", password="x")
        self.chapter = Chapter.objects.create(user=self.user, name="Book")
        self.doc = Document.objects.create(user=self.user, chapter=self.chapter, title="Notes",
            active_version=1, pending_version=2, ingestion_token="owner")
        self.revision = DocumentIndexVersion.objects.create(document=self.doc, version=2)
        DocumentPage.objects.create(document=self.doc, version=2, page_number=1, reconstructed_md="unreadable fragment",
            needs_review=True, review_reason="Too little readable text")
        DocumentPage.objects.create(document=self.doc, version=2, page_number=2, reconstructed_md="Readable material " * 10)

    def test_audit_persists_without_changing_active_coverage(self):
        with mock.patch("accounts.coverage.push_ingestion_status") as push:
            audit = record_coverage(self.doc, self.revision, mock.Mock(token="owner"))
        self.doc.refresh_from_db()
        self.assertEqual(audit["code"], "INCOMPLETE_COVERAGE")
        self.assertEqual(audit["pages"][0]["page_number"], 1)
        self.assertEqual(DocumentSerializer(self.doc).data["coverage_audit"], audit)
        self.assertEqual(self.doc.active_coverage, {})
        self.assertEqual(self.doc.active_version, 1)
        push.assert_called_once()
        page = self.doc.pages.get(version=2, page_number=1)
        self.assertTrue(DocumentPageSerializer(page).data["needs_review"])

    def test_unreadable_pages_excluded_from_text_chunks_and_summary(self):
        from accounts.tasks import _version_chunks
        from types import SimpleNamespace
        self.doc.active_version = 2
        self.doc.save(update_fields=["active_version"])
        self.assertNotIn("fragment", pages.canonical_text_for_document(self.doc))
        chunks = _version_chunks(self.doc, "stale fragment", SimpleNamespace(encode=list, decode="".join), 2)
        self.assertEqual({chunk[1] for chunk in chunks}, {2})
        material = async_to_sync(RagPipeline._chapter_pages)(self.chapter.id, self.user.id)
        self.assertTrue(material.incomplete_coverage)
        self.assertNotIn("fragment", "".join(material))
        pipeline = object.__new__(RagPipeline)
        pipeline.llm_client = None
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Summary"))])
        with mock.patch("accounts.rag_pipeline.ask_llm", return_value=response):
            result = async_to_sync(pipeline.handle_summary)(self.chapter.id, self.user.id)
        self.assertTrue(result.answer.startswith(COVERAGE_ADVISORY))

    def test_all_unreadable_pages_cannot_fall_back_to_stale_flat_text(self):
        from accounts.tasks import _version_chunks
        from types import SimpleNamespace
        self.doc.pages.filter(version=2).update(needs_review=True)
        with self.assertRaisesRegex(ValueError, "No readable chunks"):
            _version_chunks(self.doc, "Old material must never be reused", SimpleNamespace(encode=list, decode="".join), 2)
