"""
Celery Document Worker for LexFind.
Executes the document processing pipeline asynchronously.
"""
import logging
import os
import sys

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="process_document_task",
    bind=True,
    queue="lexfind_documents",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def process_document_task(self, document_id: str, blob_path: str) -> dict:
    logger.info(
        "[Task %s] Starting document processing | document_id=%s blob=%s",
        self.request.id, document_id, blob_path,
    )

    try:
        from app.services.document_processing_service import document_processing_service
        document_processing_service.process_document(document_id, blob_path)

        logger.info(
            "[Task %s] Document processing completed | document_id=%s",
            self.request.id, document_id,
        )
        return {"document_id": document_id, "status": "ready"}

    except Exception as exc:
        logger.exception(
            "[Task %s] Unexpected error during processing | document_id=%s: %s",
            self.request.id, document_id, exc,
        )
        raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
