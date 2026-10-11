"""Version-specific coverage audits, kept separate from document readiness."""
import logging
from .models import Document
from .realtime import push_ingestion_status, PHASE_PAGE

logger = logging.getLogger(__name__)
COVERAGE_ADVISORY = "Some pages in these notes could not be read. This answer covers only the readable material."


def record_coverage(doc, revision, lease):
    pages = list(doc.pages.filter(version=revision.version, needs_review=True)
                 .values('page_number', 'review_reason'))
    audit = ({'code': 'INCOMPLETE_COVERAGE', 'version': revision.version, 'pages': pages} if pages else {})
    revision.coverage_audit = audit
    revision.save(update_fields=['coverage_audit'])
    lease.check()
    updated = Document.objects.filter(id=doc.id, ingestion_token=lease.token,
                                      pending_version=revision.version).update(coverage_audit=audit)
    if not updated:
        from .ingestion_versions import LeaseLost
        raise LeaseLost('A newer ingestion owns the coverage audit')
    doc.coverage_audit = audit
    if audit:
        logger.warning('INCOMPLETE_COVERAGE', extra={'document_id': str(doc.id), **audit})
        push_ingestion_status(doc.user_id, doc.id, PHASE_PAGE, coverage_audit=audit)
    return audit


class SourcePages(list):
    """Canonical summary pages plus version-matched coverage information."""
    incomplete_coverage = False
