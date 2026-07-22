from unittest import mock

from rest_framework.test import APITestCase

from accounts.models import CustomUserModel, Document, DocumentPage


class PageApiTest(APITestCase):
    def setUp(self):
        self.user = CustomUserModel.objects.create_user(email="a@b.com", password="x", name="A")
        self.client.force_authenticate(self.user)
        self.doc = Document.objects.create(user=self.user, title="t", file="u/x.pdf", file_type="pdf")
        DocumentPage.objects.create(document=self.doc, page_number=1, reconstructed_md="## Hi",
                                    image_url="https://s3/p1.png", text_source=DocumentPage.SOURCE_VISION)

    def test_list_pages(self):
        r = self.client.get(f"/auth/documents/{self.doc.id}/pages/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data[0]["page_number"], 1)
        self.assertEqual(r.data[0]["reconstructed_md"], "## Hi")

    @mock.patch("accounts.views.rescan_document_with_vision")
    def test_rescan_dispatches(self, task):
        r = self.client.post(f"/auth/documents/{self.doc.id}/rescan/")
        self.assertEqual(r.status_code, 202)
        task.delay.assert_called_once_with(str(self.doc.id))

    def test_cannot_see_others_pages(self):
        other = CustomUserModel.objects.create_user(email="z@z.com", password="x", name="Z")
        self.client.force_authenticate(other)
        r = self.client.get(f"/auth/documents/{self.doc.id}/pages/")
        self.assertEqual(r.status_code, 404)
