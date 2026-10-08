"""Backfill legacy or current documents through versioned ingestion.

The previous version remains searchable until the complete replacement is
verified and activated. Legacy UUID/prefix IDs are read as version zero and
retired only after a successful replacement and the cleanup grace period.
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import close_old_connections

from accounts.models import Document
from accounts.tasks import process_document_ingestion


class Command(BaseCommand):
    help = "Build and activate verified document versions without deleting live vectors."

    def add_arguments(self, parser):
        parser.add_argument(
            "--document",
            help="Reindex a single document by id (repairs one failed upload).",
        )
        parser.add_argument(
            "--status",
            default=Document.STATUS_COMPLETED,
            help=(
                "Which status to reindex (default: COMPLETED). FAILED documents "
                "are skipped by default because they have no usable text and "
                "would fail again."
            ),
        )
        parser.add_argument(
            "--sync",
            action="store_true",
            help=(
                "Run ingestion in this process instead of dispatching to Celery. "
                "Use when no worker is running; it is far slower and gives up "
                "the task's retry behaviour."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List what would be reindexed and exit.",
        )

    def handle(self, *args, **options):
        if options["document"]:
            documents = Document.objects.filter(id=options["document"])
            if not documents.exists():
                raise CommandError(f"No document with id {options['document']}")
        else:
            documents = Document.objects.filter(
                status=options["status"]
            ).order_by("created_at")

        total = documents.count()
        if not total:
            self.stdout.write(self.style.WARNING("Nothing to reindex."))
            return

        self.stdout.write(f"{total} document(s) to reindex.")

        if options["dry_run"]:
            for doc in documents:
                self.stdout.write(f"  would reindex {doc.id}  {doc.title or '(untitled)'}")
            self.stdout.write(self.style.WARNING("Dry run — nothing dispatched."))
            return

        # Read the whole work list NOW rather than iterating the queryset lazily.
        # A lazy queryset holds a server-side cursor for the entire run, and when
        # the pooler drops the connection mid-loop the iteration itself dies —
        # taking every document that had not been reached yet, with no record of
        # which ones they were.
        work = list(documents.values_list("id", "title"))

        succeeded = 0
        skipped = 0
        failures = []

        for doc_id, title in work:
            label = f"{doc_id}  {title or '(untitled)'}"
            # Each document gets a fresh connection. See the matching call in
            # process_document_ingestion for why.
            close_old_connections()
            try:
                if options["sync"]:
                    # .apply() runs the task body inline — but it CAPTURES the
                    # exception into an EagerResult instead of raising it, so the
                    # try/except below never sees a task-level failure. Reporting
                    # a dispatch as a success is how the previous run printed
                    # "80/80 reindexed" while 35 documents were dying.
                    result = process_document_ingestion.apply(args=[str(doc_id)])
                    if result.failed():
                        raise result.result
                    if isinstance(result.result, dict) and result.result.get("status") != "completed":
                        skipped += 1
                        self.stdout.write(f"  [skipped] {label}: {result.result['status']}")
                        continue
                else:
                    process_document_ingestion.delay(str(doc_id))
                succeeded += 1
                self.stdout.write(f"  [ok]   {label}")
            except Exception as e:
                # One document failing must not abandon the rest. Collect and
                # report at the end so the operator sees the whole picture
                # rather than the first problem.
                failures.append((doc_id, f"{type(e).__name__}: {e}"))
                self.stdout.write(self.style.ERROR(f"  [FAIL] {label}: {e}"))

        # The two modes promise different things and must not claim the same
        # one. --sync has actually run the work and knows the outcome; the async
        # path has only put a message on a queue and knows nothing beyond that.
        verb = "reindexed" if options["sync"] else "dispatched"
        style = self.style.SUCCESS if not failures else self.style.WARNING
        self.stdout.write(style(f"\n{succeeded}/{total} {verb}."))
        if skipped:
            self.stdout.write(f"{skipped} skipped (already running or deleted).")

        if failures:
            self.stdout.write(self.style.ERROR(f"{len(failures)} failed:"))
            for doc_id, error in failures:
                self.stdout.write(self.style.ERROR(f"  {doc_id}: {error}"))

        if not options["sync"]:
            self.stdout.write(
                "Tasks are queued — watch the Celery worker log for completion. "
                "Each existing active version remains available until its replacement is verified."
            )
