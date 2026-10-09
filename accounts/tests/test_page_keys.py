from importlib import import_module
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, urlsplit
from unittest import mock

from django.apps import apps
from django.db import connection
from django.test import TestCase, TransactionTestCase, override_settings
from django.db.migrations.executor import MigrationExecutor
from rest_framework.test import APIClient
from storages.backends.s3 import S3Storage

from accounts.models import CustomUserModel, Document, DocumentPage
from accounts.serializers import DocumentPageSerializer
from accounts.tests.test_storage import PROD_STORAGE

migration = import_module("accounts.migrations.0013_documentpage_s3_object_key")


@override_settings(**PROD_STORAGE)
class PageKeysTests(TestCase):
    def setUp(self):
        self.user = CustomUserModel.objects.create_user(email="keys@test.com", name="K", password="x")
        self.doc = Document.objects.create(user=self.user, title="Doc", active_version=1)
        self.key = f"{self.user.id}/pages/{self.doc.id}/p1_a b+%23.png"
        self.url = "https://ragbag-media-test.s3.ap-south-1.amazonaws.com/" + quote(self.key) + "?X-Amz-Signature=expired"
        self.page = DocumentPage.objects.create(document=self.doc, version=1, page_number=1,
                                                image_url=self.url)

    def test_backfill_is_idempotent_and_retains_legacy_url(self):
        editor = SimpleNamespace(connection=connection)
        migration.backfill_keys(apps, editor)
        migration.backfill_keys(apps, editor)
        self.page.refresh_from_db()
        self.assertEqual(self.page.s3_object_key, self.key)
        self.assertEqual(self.page.image_url, self.url)

    def test_path_style_and_local_urls(self):
        urls = [f"https://s3.ap-south-1.amazonaws.com/ragbag-media-test/{quote(self.key)}?token=old",
                f"/media/{quote(self.key)}"]
        for url in urls:
            self.assertEqual(migration.key_from_url(url, self.user.id, self.doc.id), self.key)
        with override_settings(AWS_S3_ENDPOINT_URL="https://storage.example/storage/v1/s3"):
            url = f"https://storage.example/storage/v1/s3/ragbag-media-test/{quote(self.key)}?old=1"
            self.assertEqual(migration.key_from_url(url, self.user.id, self.doc.id), self.key)

    def test_foreign_hosts_buckets_and_document_prefixes_are_not_resigned(self):
        for url in [self.url.replace("amazonaws.com", "amazonaws.com.attacker.invalid"),
                    self.url.replace("ragbag-media-test", "another-bucket"),
                    self.url.replace(str(self.doc.id), "another-document"),
                    "https://[invalid", ""]:
            self.assertEqual(migration.key_from_url(url, self.user.id, self.doc.id), "")

    def test_serialization_signs_a_fresh_fifteen_minute_url(self):
        self.page.s3_object_key = self.key
        storage = S3Storage()
        with mock.patch("accounts.serializers.default_storage", storage):
            url = DocumentPageSerializer(self.page).data["image_url"]
        query = parse_qs(urlsplit(url).query)
        self.assertEqual(query["X-Amz-Expires"], ["900"])
        self.assertNotEqual(query["X-Amz-Signature"], ["expired"])

    def test_unknown_legacy_url_is_not_served(self):
        self.assertEqual(DocumentPageSerializer(self.page).data["image_url"], "")

    def test_api_signs_only_owned_active_pages_on_every_read(self):
        self.page.s3_object_key = self.key
        self.page.save()
        DocumentPage.objects.create(document=self.doc, version=2, page_number=1,
                                    s3_object_key="pending/key.png")
        client = APIClient()
        client.force_authenticate(self.user)
        with mock.patch("accounts.serializers.default_storage", spec=S3Storage) as storage:
            storage.url.side_effect = ["fresh-one", "fresh-two"]
            self.assertEqual(client.get(f"/auth/documents/{self.doc.id}/pages/").data[0]["image_url"], "fresh-one")
            self.assertEqual(client.get(f"/auth/documents/{self.doc.id}/pages/").data[0]["image_url"], "fresh-two")
            self.assertEqual(storage.url.call_args_list, [mock.call(self.key, expire=900)] * 2)
            other = CustomUserModel.objects.create_user(email="foreign@test.com", name="F", password="x")
            client.force_authenticate(other)
            self.assertEqual(client.get(f"/auth/documents/{self.doc.id}/pages/").status_code, 404)
            self.assertEqual(storage.url.call_count, 2)

    @override_settings(STORAGES={"default": {"BACKEND": "django.core.files.storage.FileSystemStorage"}})
    def test_local_storage_still_works(self):
        self.page.s3_object_key = "1/pages/doc/p1_local.png"
        self.assertEqual(DocumentPageSerializer(self.page).data["image_url"], "/media/1/pages/doc/p1_local.png")


@override_settings(**PROD_STORAGE)
class PageKeyMigrationTests(TransactionTestCase):
    def test_upgrade_backfills_preexisting_rows_without_fetching_storage(self):
        before = [("accounts", "0012_document_index_versions")]
        after = [("accounts", "0013_documentpage_s3_object_key")]
        executor = MigrationExecutor(connection)
        executor.migrate(before)
        self.addCleanup(lambda: MigrationExecutor(connection).migrate(after))
        old_apps = executor.loader.project_state(before).apps
        user = old_apps.get_model("accounts", "CustomUserModel").objects.create(
            email="migration@test.com", name="M", password="!",
        )
        doc = old_apps.get_model("accounts", "Document").objects.create(user_id=user.id, title="Old book")
        key = f"{user.id}/pages/{doc.id}/p1_original.png"
        old_page = old_apps.get_model("accounts", "DocumentPage").objects.create(
            document_id=doc.id, page_number=1,
            image_url="https://ragbag-media-test.s3.ap-south-1.amazonaws.com/" + key + "?X-Amz-Signature=expired",
        )
        with mock.patch("django.core.files.storage.default_storage.open", side_effect=AssertionError("network forbidden")):
            MigrationExecutor(connection).migrate(after)
        self.assertEqual(DocumentPage.objects.get(id=old_page.id).s3_object_key, key)
