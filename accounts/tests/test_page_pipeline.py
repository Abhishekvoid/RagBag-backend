from unittest import mock
import io
import tempfile
from pathlib import Path
import fitz

from django.test import TestCase, SimpleTestCase, override_settings

from accounts.models import CustomUserModel, Document, DocumentPage
from accounts import page_pipeline, vision_ocr


class BuildPagesTest(TestCase):
    def setUp(self):
        self.user = CustomUserModel.objects.create_user(email="a@b.com", password="x", name="A")
        self.doc = Document.objects.create(user=self.user, title="t", file="u/x.pdf", file_type="pdf")
        pdf = fitz.open()
        pdf.new_page()
        pdf.new_page().insert_text((50, 50), "Good clean layer text " * 4)
        data = pdf.tobytes()
        pdf.close()
        patcher = mock.patch.object(page_pipeline.default_storage, "open", side_effect=lambda *a: io.BytesIO(data))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(page_pipeline, "push_ingestion_status")
        patcher.start()
        self.addCleanup(patcher.stop)

    @mock.patch.object(page_pipeline, "store_page_image", return_value="https://s3/p.png")
    @mock.patch.object(page_pipeline, "reconstruct_page_markdown", return_value="## Vision page")
    def test_mixed_layer_and_vision(self, *_):
        with mock.patch.object(page_pipeline, "VISION_ENABLED", True):
            n = page_pipeline.build_document_pages(self.doc)
        self.assertEqual(n, 2)
        p1, p2 = list(self.doc.pages.all())
        self.assertEqual(p1.text_source, DocumentPage.SOURCE_VISION)   # empty layer -> vision
        self.assertEqual(p1.reconstructed_md, "## Vision page")
        self.assertEqual(p2.text_source, DocumentPage.SOURCE_LAYER)    # good layer -> skip vision
        self.assertIn("Good clean layer text", p2.reconstructed_md)

    @mock.patch.object(page_pipeline, "store_page_image", return_value="https://s3/p.png")
    @mock.patch.object(page_pipeline, "reconstruct_page_markdown",
                       side_effect=vision_ocr.VisionUnavailable("ollama down"))
    def test_vision_failure_falls_back(self, *_):
        with mock.patch.object(page_pipeline, "VISION_ENABLED", True):
            page_pipeline.build_document_pages(self.doc)
        p1 = self.doc.pages.get(page_number=1)
        self.assertEqual(p1.text_source, DocumentPage.SOURCE_FALLBACK)

    def test_canonical_text_strips_markers(self):
        DocumentPage.objects.create(document=self.doc, page_number=1,
                                    reconstructed_md="ATP in the [?thylakoid]",
                                    text_source=DocumentPage.SOURCE_VISION)
        self.assertEqual(page_pipeline.canonical_text_for_document(self.doc).strip(),
                         "ATP in the thylakoid")

    @override_settings(MAX_PAGES_PER_DOCUMENT=1)
    @mock.patch.object(page_pipeline, "store_page_image")
    def test_oversized_pdf_is_rejected_before_rendering_or_upload(self, store):
        with mock.patch.object(fitz.Page, "get_pixmap") as render:
            with self.assertRaises(page_pipeline.DocumentOversizedError):
                page_pipeline.build_document_pages(self.doc)
        render.assert_not_called()
        store.assert_not_called()
        self.assertFalse(self.doc.pages.exists())

    @mock.patch.object(page_pipeline, "store_page_image", return_value="https://s3/page.png")
    def test_previous_page_is_persisted_before_next_page_renders(self, _):
        def pages(path):
            yield 1, 2, "First page", b"one"
            self.assertTrue(self.doc.pages.filter(page_number=1).exists())
            yield 2, 2, "Second page", b"two"
        with mock.patch.object(page_pipeline, "render_pdf_pages", side_effect=pages), \
             mock.patch.object(page_pipeline, "VISION_ENABLED", False):
            self.assertEqual(page_pipeline.build_document_pages(self.doc), 2)

    def test_generator_and_temporary_file_close_on_upload_failure(self):
        paths = []
        closed = []
        def pages(path):
            paths.append(path)
            try:
                yield 1, 1, "Page", b"png"
            finally:
                closed.append(True)
        with mock.patch.object(page_pipeline, "render_pdf_pages", side_effect=pages), \
             mock.patch.object(page_pipeline, "store_page_image", side_effect=RuntimeError("storage down")):
            with self.assertRaises(RuntimeError):
                page_pipeline.build_document_pages(self.doc)
        self.assertEqual(closed, [True])
        self.assertFalse(paths[0].exists())


class RenderBoundsTests(SimpleTestCase):
    def test_large_landscape_and_portrait_pages_are_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "huge.pdf"
            with fitz.open() as pdf:
                pdf.new_page(width=5000, height=10000)
                pdf.new_page(width=10000, height=5000)
                pdf.new_page(width=72, height=72)
                pdf.save(path)
            sizes = []
            for _, _, _, png in page_pipeline.render_pdf_pages(path, dpi=900):
                pix = fitz.Pixmap(png)
                sizes.append((pix.width, pix.height))
            self.assertEqual(sizes, [(1024, 2048), (2048, 1024), (150, 150)])


class ChunkMetadataTest(TestCase):
    def test_metadata_and_page_lookup(self):
        from accounts import tasks
        user = CustomUserModel.objects.create_user(email="c@d.com", password="x", name="C")
        doc = Document.objects.create(user=user, title="t", file="u/x.pdf", file_type="pdf")
        page = DocumentPage.objects.create(document=doc, page_number=42,
                                           reconstructed_md="Mitochondria are the powerhouse of the cell",
                                           text_source=DocumentPage.SOURCE_VISION)
        pages = [page]
        n = tasks._page_for_chunk("Mitochondria are the powerhouse", pages)
        meta = tasks.build_chunk_metadata(doc, "Mitochondria are the powerhouse", page_number=n)
        self.assertEqual(meta["page_number"], 42)
        self.assertEqual(meta["document_id"], str(doc.id))
        self.assertIsNone(tasks._page_for_chunk("nowhere on any page", pages))
