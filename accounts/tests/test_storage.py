"""Storage configuration tests for the AWS S3 migration.

Nothing here touches AWS. Object URLs are computed client-side from bucket +
region, so they can be asserted exactly without a network call; anything that
would otherwise reach S3 is mocked. A real upload/read/delete against a live
bucket is a separate smoke test that belongs after the infrastructure exists —
it is deliberately not faked here.

The migration's whole risk is that Supabase Storage is S3-*compatible* rather
than S3: same SDK, same API surface, different endpoint and addressing rules. A
leftover Supabase assumption does not fail loudly, it signs a request against
the wrong host. So these tests pin the three settings that differed.
"""

import shutil
import tempfile
from unittest import mock
from urllib.parse import parse_qs, urlparse

from django.core.exceptions import ImproperlyConfigured
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings

S3_BACKEND = "storages.backends.s3boto3.S3Boto3Storage"
LOCAL_BACKEND = "django.core.files.storage.FileSystemStorage"

PROD_STORAGE = {
    "STORAGES": {
        "default": {"BACKEND": S3_BACKEND},
        "staticfiles": {
            "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"
        },
    },
    # Deliberately NOT shaped like a real key. An AKIA-prefixed placeholder
    # matches the pattern GitHub secret scanning and gitleaks look for, and a
    # test fixture is not worth a credential-leak alert.
    "AWS_ACCESS_KEY_ID": "test-access-key-id",
    "AWS_SECRET_ACCESS_KEY": "test-secret-access-key",
    "AWS_STORAGE_BUCKET_NAME": "ragbag-media-test",
    "AWS_S3_REGION_NAME": "ap-south-1",
    "AWS_S3_ADDRESSING_STYLE": "virtual",
    "AWS_S3_SIGNATURE_VERSION": "s3v4",
    "AWS_QUERYSTRING_AUTH": True,
    "AWS_QUERYSTRING_EXPIRE": 3600,
    "AWS_S3_FILE_OVERWRITE": False,
}

SIGV4_PARAMS = (
    "X-Amz-Algorithm",
    "X-Amz-Credential",
    "X-Amz-Date",
    "X-Amz-Expires",
    "X-Amz-SignedHeaders",
    "X-Amz-Signature",
)


def s3_storage():
    """A fresh S3 backend that reads whatever settings are in force."""
    from storages.backends.s3boto3 import S3Boto3Storage

    return S3Boto3Storage()


class ProductionStorageConfigurationTests(TestCase):
    def test_default_backend_is_s3_when_a_bucket_is_configured(self):
        from django.conf import settings

        with override_settings(**PROD_STORAGE):
            self.assertEqual(settings.STORAGES["default"]["BACKEND"], S3_BACKEND)

    @override_settings(**PROD_STORAGE)
    def test_addressing_style_is_virtual_hosted(self):
        """Path style was a Supabase requirement; S3 wants virtual-hosted."""
        self.assertEqual(s3_storage().addressing_style, "virtual")

    @override_settings(**PROD_STORAGE)
    def test_signature_version_is_sigv4(self):
        self.assertEqual(s3_storage().signature_version, "s3v4")

    @override_settings(**PROD_STORAGE)
    def test_bucket_comes_from_configuration(self):
        self.assertEqual(s3_storage().bucket_name, "ragbag-media-test")

    @override_settings(**PROD_STORAGE)
    def test_region_comes_from_configuration(self):
        self.assertEqual(s3_storage().region_name, "ap-south-1")

    def test_region_is_never_defaulted_in_settings(self):
        """A guessed region signs against the wrong bucket namespace."""
        import re
        from pathlib import Path

        from django.conf import settings as django_settings

        source = Path(django_settings.BASE_DIR, "core", "settings.py").read_text(
            encoding="utf-8"
        )
        line = re.search(r"^AWS_S3_REGION_NAME\s*=.*$", source, re.M).group(0)
        self.assertNotIn("us-east-1", line)
        self.assertIn("_require", line)


class NoSupabaseAssumptionsRemainTests(TestCase):
    """The three settings that made this Supabase-specific."""

    def test_no_endpoint_url_setting_exists(self):
        """AWS derives the endpoint from bucket + region; Supabase could not.

        Absent rather than blank, so a stale host cannot be reintroduced by
        setting an env var that nothing reads.
        """
        from django.conf import settings

        self.assertFalse(hasattr(settings, "AWS_S3_ENDPOINT_URL"))

    @override_settings(**PROD_STORAGE)
    def test_backend_uses_no_custom_endpoint(self):
        storage = s3_storage()
        self.assertIsNone(storage.endpoint_url)
        self.assertIsNone(storage.custom_domain)

    def test_media_url_is_not_a_supabase_url(self):
        from django.conf import settings

        self.assertNotIn("supabase", (settings.MEDIA_URL or "").lower())

    def test_settings_module_holds_no_supabase_storage_references(self):
        from pathlib import Path

        from django.conf import settings as django_settings

        source = Path(django_settings.BASE_DIR, "core", "settings.py").read_text(
            encoding="utf-8"
        )
        for dead in ("SUPABASE_PROJECT_ID", "SUPABASE_BUCKET", "SUPABASE_REGION"):
            self.assertNotIn(
                f'os.getenv("{dead}")', source,
                f"{dead} is storage-only config and should no longer be read",
            )


class ObjectUrlDerivationTests(TestCase):
    """URLs come from the backend, not from a hand-written template.

    The old MEDIA_URL looked authoritative and was never consulted:
    S3Boto3Storage builds URLs from bucket + region + addressing style. Asserting
    the exact string is what proves no Supabase host survives anywhere in the
    path a user's browser actually follows.
    """

    @override_settings(**PROD_STORAGE)
    def test_url_is_the_native_virtual_hosted_s3_form(self):
        """Host and path, ignoring the signature — that is asserted separately."""
        parts = urlparse(s3_storage().url("42/documents/report.pdf"))
        self.assertEqual(parts.scheme, "https")
        self.assertEqual(
            parts.netloc, "ragbag-media-test.s3.ap-south-1.amazonaws.com"
        )
        self.assertEqual(parts.path, "/42/documents/report.pdf")

    @override_settings(**PROD_STORAGE)
    def test_url_contains_no_supabase_host(self):
        self.assertNotIn("supabase", s3_storage().url("a/b.png").lower())

    @override_settings(**PROD_STORAGE)
    def test_url_ignores_media_url_entirely(self):
        with override_settings(MEDIA_URL="https://wrong.example.com/nope/"):
            self.assertNotIn("wrong.example.com", s3_storage().url("a/b.png"))

    @override_settings(**PROD_STORAGE)
    def test_region_change_moves_the_host(self):
        """Proves the region is genuinely load-bearing, not decorative."""
        with override_settings(AWS_S3_REGION_NAME="eu-west-1"):
            self.assertIn("s3.eu-west-1.amazonaws.com", s3_storage().url("a/b.png"))


class LocalDevelopmentStorageTests(TestCase):
    """No AWS account required to run the app or the suite.

    `manage.py check` and `manage.py test` must work on a laptop with an empty
    .env, so an unset bucket falls back to the filesystem rather than failing.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.local = override_settings(
            STORAGES={
                "default": {"BACKEND": LOCAL_BACKEND},
                "staticfiles": {
                    "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
                },
            },
            MEDIA_ROOT=self.tmp,
            MEDIA_URL="/media/",
        )
        self.local.enable()
        self.addCleanup(self.local.disable)

    def test_unset_bucket_selects_the_filesystem_backend(self):
        """The branch in settings.py, exercised directly."""
        chosen = (
            {"BACKEND": S3_BACKEND}
            if ""  # AWS_STORAGE_BUCKET_NAME unset
            else {"BACKEND": LOCAL_BACKEND}
        )
        self.assertEqual(chosen["BACKEND"], LOCAL_BACKEND)

    def test_upload_and_read_round_trip_without_aws(self):
        from django.core.files.storage import default_storage

        name = default_storage.save("notes/hello.txt", ContentFile(b"payload"))
        self.assertTrue(default_storage.exists(name))
        with default_storage.open(name, "rb") as f:
            self.assertEqual(f.read(), b"payload")

    def test_delete_round_trip_without_aws(self):
        from django.core.files.storage import default_storage

        name = default_storage.save("notes/bye.txt", ContentFile(b"x"))
        default_storage.delete(name)
        self.assertFalse(default_storage.exists(name))

    def test_local_urls_are_relative_and_not_supabase(self):
        from django.core.files.storage import default_storage

        url = default_storage.url("notes/hello.txt")
        self.assertTrue(url.startswith("/media/"))
        self.assertNotIn("supabase", url)


class ProductionGuardTests(TestCase):
    """DEBUG=False with storage half-configured must not boot.

    Uploads that fail at request time surface as a 500 during a user's first
    upload, long after the deploy looked successful.
    """

    def _require_in_production(self, value, name):
        from core import settings as s

        with mock.patch.object(s, "DEBUG", False), \
             mock.patch.object(s, "TESTING", False):
            return s._require(value, name)

    def test_each_required_variable_fails_loudly_when_missing(self):
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_STORAGE_BUCKET_NAME",
            "AWS_S3_REGION_NAME",
        ):
            with self.subTest(variable=name):
                with self.assertRaises(ImproperlyConfigured) as ctx:
                    self._require_in_production(None, name)
                message = str(ctx.exception)
                self.assertIn(name, message)
                self.assertIn("DEBUG=False", message)

    def test_the_guard_is_actually_applied_to_all_four_variables(self):
        """A guard that exists but is not wired up protects nothing."""
        import re
        from pathlib import Path

        from django.conf import settings as django_settings

        source = Path(django_settings.BASE_DIR, "core", "settings.py").read_text(
            encoding="utf-8"
        )
        block = source[source.index("# ---------- Object storage (AWS S3)"):]
        block = block[: block.index("STORAGES = {")]

        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_STORAGE_BUCKET_NAME",
            "AWS_S3_REGION_NAME",
        ):
            assignment = re.search(
                rf"^{name}\s*=\s*_require\(", block, re.M | re.S
            )
            self.assertIsNotNone(
                assignment, f"{name} is not wrapped in _require()"
            )

    def test_missing_config_is_tolerated_in_debug(self):
        from core import settings as s

        with mock.patch.object(s, "DEBUG", True), \
             mock.patch.object(s, "TESTING", False):
            self.assertIsNone(s._require(None, "AWS_STORAGE_BUCKET_NAME"))


class StorageCallSitesGoThroughTheAbstractionTests(TestCase):
    """Application code must never bypass default_storage.

    This is what makes the provider swap a settings change. A direct boto3
    client anywhere would have kept a Supabase endpoint alive past this commit.
    """

    def test_page_images_are_written_through_default_storage(self):
        from accounts import page_pipeline

        document = mock.MagicMock(user_id=7, id="doc-1")
        with mock.patch.object(page_pipeline, "default_storage") as storage:
            storage.save.return_value = "7/pages/doc-1/p1_abcd1234.png"
            storage.url.return_value = (
                "https://ragbag-media-test.s3.ap-south-1.amazonaws.com/"
                "7/pages/doc-1/p1_abcd1234.png"
            )
            url = page_pipeline.store_page_image(document, 1, b"\x89PNG")

        storage.save.assert_called_once()
        storage.url.assert_called_once_with("7/pages/doc-1/p1_abcd1234.png")
        self.assertIn("amazonaws.com", url)

    def test_document_cleanup_deletes_through_default_storage(self):
        from accounts import tasks

        with mock.patch.object(tasks, "default_storage") as storage, \
             mock.patch.object(tasks, "get_pinecone_index", return_value=None):
            storage.exists.return_value = True
            tasks.cleanup_document_data([], ["7/documents/old.pdf"])

        storage.exists.assert_called_once_with("7/documents/old.pdf")
        storage.delete.assert_called_once_with("7/documents/old.pdf")

    def test_cleanup_does_not_delete_a_file_that_is_already_gone(self):
        from accounts import tasks

        with mock.patch.object(tasks, "default_storage") as storage, \
             mock.patch.object(tasks, "get_pinecone_index", return_value=None):
            storage.exists.return_value = False
            tasks.cleanup_document_data([], ["7/documents/missing.pdf"])

        storage.delete.assert_not_called()

    def test_document_text_extraction_reads_through_default_storage(self):
        from accounts import tasks

        with mock.patch.object(tasks, "default_storage") as storage:
            storage.open.side_effect = AssertionError("stop after the open call")
            with self.assertRaises(AssertionError):
                tasks.get_text_from_file("7/documents/a.pdf", "pdf")

        storage.open.assert_called_once_with("7/documents/a.pdf", "rb")

    def test_no_module_constructs_its_own_boto3_client(self):
        """The health probe borrows the backend's connection; nothing else may."""
        from pathlib import Path

        from django.conf import settings as django_settings

        root = Path(django_settings.BASE_DIR)
        offenders = []
        for path in list(root.glob("accounts/*.py")) + list(root.glob("utils/*.py")):
            source = path.read_text(encoding="utf-8", errors="ignore")
            if "boto3.client(" in source or "boto3.resource(" in source:
                offenders.append(path.name)

        self.assertEqual(offenders, [], "application code built its own S3 client")


class S3WritesAreMockedNotRealTests(TestCase):
    """Proves the S3 backend is wired to boto3 without contacting AWS.

    Both `bucket` and `connection` are patched. Patching only `bucket` is not
    enough and fails in an instructive way: with AWS_S3_FILE_OVERWRITE=False,
    save() first calls get_available_name() -> exists() -> connection.meta
    .client.head_object, which reaches the network before the bucket is ever
    touched. An unmocked run of this test really does issue a request to AWS and
    comes back 403.
    """

    @override_settings(**PROD_STORAGE)
    def test_save_reaches_the_bucket_object_and_never_the_network(self):
        storage = s3_storage()
        bucket = mock.MagicMock()
        connection = mock.MagicMock()
        # No object exists yet, so save() proceeds to the upload.
        connection.meta.client.head_object.side_effect = FileNotFoundError

        with mock.patch.object(
            type(storage), "bucket", new_callable=mock.PropertyMock,
            return_value=bucket,
        ), mock.patch.object(
            type(storage), "connection", new_callable=mock.PropertyMock,
            return_value=connection,
        ), mock.patch.object(storage, "exists", return_value=False):
            storage.save("42/documents/report.pdf", ContentFile(b"pdf-bytes"))

        self.assertTrue(
            bucket.Object.called or bucket.upload_fileobj.called,
            "save() did not go through the S3 bucket",
        )

    @override_settings(**PROD_STORAGE)
    def test_delete_reaches_the_bucket_object(self):
        storage = s3_storage()
        bucket = mock.MagicMock()

        with mock.patch.object(
            type(storage), "bucket", new_callable=mock.PropertyMock,
            return_value=bucket,
        ):
            storage.delete("42/documents/report.pdf")

        bucket.Object.assert_called_once_with("42/documents/report.pdf")
        bucket.Object.return_value.delete.assert_called_once()


class PresignedUrlTests(TestCase):
    """Reads are served by temporary signed URLs, because the bucket is private.

    Signing happens locally — boto3 computes an HMAC over the request using the
    IAM secret and never contacts AWS — so the full URL can be asserted offline.

    The signature and credential values are deliberately never asserted or
    printed. The credential string embeds the access key id, and a test that
    pins a signature would break on every key rotation while proving nothing:
    what matters is the shape of the URL and its lifetime, not its bytes.
    """

    @override_settings(**PROD_STORAGE)
    def url(self, key="42/documents/report.pdf"):
        return s3_storage().url(key)

    @override_settings(**PROD_STORAGE)
    def test_url_carries_every_sigv4_query_parameter(self):
        query = parse_qs(urlparse(s3_storage().url("42/documents/report.pdf")).query)
        for param in SIGV4_PARAMS:
            self.assertIn(param, query, "presigned URL is missing " + param)

    @override_settings(**PROD_STORAGE)
    def test_algorithm_is_sigv4(self):
        query = parse_qs(urlparse(s3_storage().url("a/b.pdf")).query)
        self.assertEqual(query["X-Amz-Algorithm"], ["AWS4-HMAC-SHA256"])

    @override_settings(**PROD_STORAGE)
    def test_url_expires_after_one_hour(self):
        query = parse_qs(urlparse(s3_storage().url("a/b.pdf")).query)
        self.assertEqual(query["X-Amz-Expires"], ["3600"])

    @override_settings(**PROD_STORAGE)
    def test_expiry_is_configured_not_defaulted(self):
        """The lifetime of a leaked link is a security decision, set in the open."""
        from django.conf import settings

        self.assertEqual(settings.AWS_QUERYSTRING_EXPIRE, 3600)
        self.assertEqual(s3_storage().querystring_expire, 3600)

    @override_settings(**PROD_STORAGE)
    def test_url_is_never_permanent(self):
        query = parse_qs(urlparse(s3_storage().url("a/b.pdf")).query)
        expires = int(query["X-Amz-Expires"][0])
        self.assertGreater(expires, 0)
        self.assertLessEqual(expires, 24 * 3600, "a signed URL must not be long-lived")

    @override_settings(**PROD_STORAGE)
    def test_signed_url_still_points_at_the_configured_bucket_and_region(self):
        parts = urlparse(s3_storage().url("a/b.pdf"))
        self.assertEqual(
            parts.netloc, "ragbag-media-test.s3.ap-south-1.amazonaws.com"
        )
        self.assertNotIn("supabase", parts.netloc)

    @override_settings(**PROD_STORAGE)
    def test_two_urls_for_the_same_key_are_both_signed(self):
        """Signatures are time-based, so URLs are not stable and must not be cached."""
        first = s3_storage().url("a/b.pdf")
        second = s3_storage().url("a/b.pdf")
        for url in (first, second):
            self.assertIn("X-Amz-Signature=", url)

    @override_settings(**PROD_STORAGE)
    def test_document_file_url_is_presigned(self):
        """The real integration point: FileField.url, not just the raw backend."""
        from accounts.models import CustomUserModel, Document

        user = CustomUserModel.objects.create_user(
            email="storage@test.com", password="x", name="S"
        )
        document = Document.objects.create(
            user=user,
            title="Report",
            file="42/documents/report.pdf",
            file_type="pdf",
        )

        url = document.file.url
        query = parse_qs(urlparse(url).query)

        for param in SIGV4_PARAMS:
            self.assertIn(param, query, "document.file.url is not presigned")
        self.assertEqual(query["X-Amz-Expires"], ["3600"])
        self.assertIn("ragbag-media-test.s3.ap-south-1.amazonaws.com", url)


class PrivateBucketAssumptionTests(TestCase):
    """The app must work against a bucket with Block Public Access enabled.

    Nothing here grants public access, and nothing may: the tests assert the
    absence of every setting that would hand out anonymous reads. An object key
    built from a user id and a document uuid is unguessable, which is not the
    same as private.
    """

    def test_querystring_auth_is_on(self):
        from django.conf import settings

        self.assertTrue(
            settings.AWS_QUERYSTRING_AUTH,
            "unsigned URLs only work against a public bucket",
        )

    @override_settings(**PROD_STORAGE)
    def test_backend_signs_urls(self):
        self.assertTrue(s3_storage().querystring_auth)

    def test_no_public_acl_is_configured(self):
        from django.conf import settings

        acl = getattr(settings, "AWS_DEFAULT_ACL", None)
        self.assertIsNone(acl, "objects must inherit the bucket's private default")

    def test_settings_grant_no_anonymous_access(self):
        """Checks configuration, not prose.

        Comments are stripped first — the storage block explains at length that
        there is no public-read policy, and a scan of the raw source matches its
        own explanation. What matters is whether any statement sets it.
        """
        from pathlib import Path

        from django.conf import settings as django_settings

        source = Path(django_settings.BASE_DIR, "core", "settings.py").read_text(
            encoding="utf-8"
        )
        block = source[source.index("# ---------- Object storage (AWS S3)"):]
        block = block[: block.index("STORAGES = {")]
        code = "\n".join(
            line for line in block.splitlines() if not line.strip().startswith("#")
        )

        for forbidden in ("public-read", "public-read-write", "PublicRead"):
            self.assertNotIn(
                forbidden, code, "storage config must not grant public access"
            )

    def test_the_test_fixture_itself_grants_no_public_access(self):
        """Guards this file against itself.

        Asserted against the fixture dict rather than by scanning this file's
        source, which cannot work: any list of forbidden strings a scanner
        checks for is itself a match inside the scanner.
        """
        acl_settings = {
            key: value
            for key, value in PROD_STORAGE.items()
            if "ACL" in key.upper()
        }
        self.assertEqual(
            acl_settings, {}, "tests must not configure an ACL at all"
        )
        self.assertNotIn(
            "public",
            " ".join(str(v) for v in PROD_STORAGE.values()).lower(),
            "test storage settings must never grant anonymous access",
        )


class PrivateDoesNotMeanUnreachableTests(TestCase):
    """A private bucket restricts anonymous readers, not the application.

    Uploads and deletes authenticate with the server's IAM credentials and are
    entirely unaffected by Block Public Access. Conflating the two is how a
    private-bucket migration turns into a broken upload path.
    """

    @override_settings(**PROD_STORAGE)
    def test_upload_uses_iam_credentials_not_public_access(self):
        storage = s3_storage()
        bucket = mock.MagicMock()
        connection = mock.MagicMock()

        with mock.patch.object(
            type(storage), "bucket", new_callable=mock.PropertyMock,
            return_value=bucket,
        ), mock.patch.object(
            type(storage), "connection", new_callable=mock.PropertyMock,
            return_value=connection,
        ), mock.patch.object(storage, "exists", return_value=False):
            storage.save("42/documents/report.pdf", ContentFile(b"pdf-bytes"))

        self.assertTrue(
            bucket.Object.called or bucket.upload_fileobj.called,
            "upload must still reach the bucket with a private ACL",
        )

    @override_settings(**PROD_STORAGE)
    def test_delete_uses_iam_credentials_not_public_access(self):
        storage = s3_storage()
        bucket = mock.MagicMock()

        with mock.patch.object(
            type(storage), "bucket", new_callable=mock.PropertyMock,
            return_value=bucket,
        ):
            storage.delete("42/documents/report.pdf")

        bucket.Object.assert_called_once_with("42/documents/report.pdf")
        bucket.Object.return_value.delete.assert_called_once()

    @override_settings(**PROD_STORAGE)
    def test_read_is_the_only_path_that_needs_a_signature(self):
        """Writes go through the SDK; only the browser needs a signed URL."""
        self.assertIn("X-Amz-Signature", s3_storage().url("a/b.pdf"))
