"""Recover cleanup jobs that were interrupted or could not reach the broker."""
from django.core.management.base import BaseCommand, CommandError

from accounts.models import DocumentIndexVersion
from accounts.tasks import prune_document_versions


class Command(BaseCommand):
    help = "Prune retired index versions after the configured grace period."

    def add_arguments(self, parser):
        parser.add_argument("--document")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        versions = DocumentIndexVersion.objects.filter(retired_at__isnull=False)
        if options["document"]:
            versions = versions.filter(document_id=options["document"])
        for document_id in versions.values_list("document_id", flat=True).distinct():
            if options["dry_run"]:
                self.stdout.write(f"Would check retired versions for {document_id}")
            else:
                result = prune_document_versions.apply(args=[str(document_id)])
                if result.failed():
                    raise CommandError(f"Version cleanup failed for {document_id}: {result.result}")
                self.stdout.write(f"Checked retired versions for {document_id}")
