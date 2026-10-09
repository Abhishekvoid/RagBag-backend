"""Recover durable keys without fetching expired URLs or contacting storage."""
import logging
import re
from urllib.parse import unquote, urlsplit

from django.conf import settings
from django.db import migrations, models

logger = logging.getLogger(__name__)


def key_from_url(url, user_id, document_id):
    """Only recover known storage locations and this document's page prefix."""
    bucket = getattr(settings, "AWS_STORAGE_BUCKET_NAME", "") or ""
    endpoint = urlsplit(getattr(settings, "AWS_S3_ENDPOINT_URL", "") or "")
    media = urlsplit(getattr(settings, "MEDIA_URL", "/media/") or "/media/")
    try:
        parsed = urlsplit(url)
        if parsed.username or parsed.password or parsed.scheme not in ("", "http", "https"):
            return ""
        path = parsed.path
        host = parsed.hostname or ""
        if bucket and re.fullmatch(
            re.escape(bucket) + r"\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com(?:\.cn)?", host
        ):
            path = path.lstrip("/")
        elif bucket and re.fullmatch(r"s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com(?:\.cn)?", host):
            prefix = "/" + bucket + "/"
            if not path.startswith(prefix):
                return ""
            path = path[len(prefix):]
        elif bucket and endpoint.hostname and host == endpoint.hostname:
            prefix = endpoint.path.rstrip("/") + "/" + bucket + "/"
            if not path.startswith(prefix):
                return ""
            path = path[len(prefix):]
        elif parsed.netloc == media.netloc and path.startswith(media.path.rstrip("/") + "/"):
            path = path[len(media.path.rstrip("/")) + 1:]
        else:
            return ""
        key = unquote(path)
        prefix = f"{user_id}/pages/{document_id}/"
        if (len(key) > 512 or not key.startswith(prefix)
                or not re.fullmatch(r"p\d+_[^/\\\x00-\x1f]+\.png", key[len(prefix):])):
            return ""
        return key
    except (ValueError, TypeError):
        return ""


def backfill_keys(apps, schema_editor):
    Page = apps.get_model("accounts", "DocumentPage")
    database = schema_editor.connection.alias
    pages = Page.objects.using(database).filter(s3_object_key="").exclude(image_url="")
    batch = []
    skipped = 0
    for page in pages.select_related("document").iterator(chunk_size=500):
        key = key_from_url(page.image_url, page.document.user_id, page.document_id)
        if not key:
            skipped += 1
            continue
        page.s3_object_key = key
        batch.append(page)
        if len(batch) == 500:
            Page.objects.using(database).bulk_update(batch, ["s3_object_key"])
            batch.clear()
    if batch:
        Page.objects.using(database).bulk_update(batch, ["s3_object_key"])
    if skipped:
        logger.warning("Page key backfill left %d unrecognized legacy URLs for repair", skipped)


class Migration(migrations.Migration):
    dependencies = [("accounts", "0012_document_index_versions")]
    operations = [
        migrations.AddField(
            model_name="documentpage", name="s3_object_key",
            field=models.CharField(blank=True, max_length=512),
        ),
        migrations.RunPython(backfill_keys, migrations.RunPython.noop),
    ]
