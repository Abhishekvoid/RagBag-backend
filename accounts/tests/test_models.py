from django.db import IntegrityError
from django.test import TestCase

from accounts.models import CustomUserModel, Document, DocumentPage


class DocumentPageModelTest(TestCase):
    def setUp(self):
        self.user = CustomUserModel.objects.create_user(email="a@b.com", password="x", name="A")
        self.doc = Document.objects.create(user=self.user, title="t", file="u/x.pdf", file_type="pdf")

    def test_page_defaults_and_ordering(self):
        p2 = DocumentPage.objects.create(document=self.doc, page_number=2, text_source=DocumentPage.SOURCE_VISION)
        p1 = DocumentPage.objects.create(document=self.doc, page_number=1, text_source=DocumentPage.SOURCE_LAYER)
        pages = list(self.doc.pages.all())
        self.assertEqual(pages, [p1, p2])           # ordered by page_number
        self.assertEqual(p1.reconstructed_md, "")   # blank default
        self.assertEqual(p1.image_url, "")

    def test_page_number_unique_per_document(self):
        DocumentPage.objects.create(document=self.doc, page_number=1, text_source=DocumentPage.SOURCE_LAYER)
        with self.assertRaises(IntegrityError):
            DocumentPage.objects.create(document=self.doc, page_number=1, text_source=DocumentPage.SOURCE_VISION)
