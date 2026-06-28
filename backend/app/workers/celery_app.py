"""
Celery Application Factory for LexFind.
Handles asynchronous tasks like document processing via RabbitMQ and PostgreSQL.
"""
import logging
import os
import sys

from celery import Celery
from dotenv import load_dotenv

_API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

_dotenv_path = os.path.abspath(os.path.join(_API_DIR, "..", ".env"))
load_dotenv(dotenv_path=_dotenv_path, override=False)

logger = logging.getLogger(__name__)

_BROKER_URL = os.environ.get(
    "CELERY_BROKER_URL",
    "amqp://guest:guest@localhost:5672//",
)
_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND",
    "db+postgresql://postgres:postgres@localhost:5432/LexFind",
)

celery_app = Celery(
    "LexFind_worker",
    broker=_BROKER_URL,
    backend=_RESULT_BACKEND,
    include=["app.workers.document_worker"]
)

celery_app.conf.update(
    broker_connection_retry_on_startup=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="lexfind_documents",
    task_queues={
        "lexfind_documents": {
            "exchange": "lexfind_documents",
            "routing_key": "lexfind_documents",
        }
    },
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    result_expires=86400,
    timezone="UTC",
    enable_utc=True,
)

logger.info("Celery app '%s' configured | broker=%s", celery_app.main, _BROKER_URL)
