from unittest import mock

from django.test import TestCase

from accounts.models import CustomUserModel, Document, DocumentPage
from accounts import page_pipeline, vision_ocr


class BuildPagesTest(TestCase):
    def setUp(self):
        self.user = CustomUserModel.objects.create_user(email="a@b.com", password="x", name="A")
        self.doc = Document.objects.create(user=self.user, title="t", file="u/x.pdf", file_type="pdf")

    @mock.patch.object(page_pipeline.default_storage, "open")
    @mock.patch.object(page_pipeline, "store_page_image", return_value="https://s3/p.png")
    @mock.patch.object(page_pipeline, "reconstruct_page_markdown", return_value="## Vision page")
    @mock.patch.object(page_pipeline, "render_pdf_pages", return_value=[b"img1", b"img2"])
    @mock.patch.object(page_pipeline, "per_page_layer_text",
                       return_value=["", "Good clean layer text " * 5])
    def test_mixed_layer_and_vision(self, *_):
        with mock.patch.object(page_pipeline, "VISION_ENABLED", True):
            n = page_pipeline.build_document_pages(self.doc)
        self.assertEqual(n, 2)
        p1, p2 = list(self.doc.pages.all())
        self.assertEqual(p1.text_source, DocumentPage.SOURCE_VISION)   # empty layer -> vision
        self.assertEqual(p1.reconstructed_md, "## Vision page")
        self.assertEqual(p2.text_source, DocumentPage.SOURCE_LAYER)    # good layer -> skip vision
        self.assertIn("Good clean layer text", p2.reconstructed_md)

    @mock.patch.object(page_pipeline.default_storage, "open")
    @mock.patch.object(page_pipeline, "store_page_image", return_value="https://s3/p.png")
    @mock.patch.object(page_pipeline, "reconstruct_page_markdown",
                       side_effect=vision_ocr.VisionUnavailable("ollama down"))
    @mock.patch.object(page_pipeline, "render_pdf_pages", return_value=[b"img1"])
    @mock.patch.object(page_pipeline, "per_page_layer_text", return_value=[""])
    def test_vision_failure_falls_back(self, *_):
        with mock.patch.object(page_pipeline, "VISION_ENABLED", True):
            page_pipeline.build_document_pages(self.doc)
        p1 = self.doc.pages.get()
        self.assertEqual(p1.text_source, DocumentPage.SOURCE_FALLBACK)

    def test_canonical_text_strips_markers(self):
        DocumentPage.objects.create(document=self.doc, page_number=1,
                                    reconstructed_md="ATP in the [?thylakoid]",
                                    text_source=DocumentPage.SOURCE_VISION)
        self.assertEqual(page_pipeline.canonical_text_for_document(self.doc).strip(),
                         "ATP in the thylakoid")


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
