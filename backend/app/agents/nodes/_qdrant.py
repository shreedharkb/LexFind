"""
Shared Qdrant client singleton for all agent nodes.

The client is instantiated exactly once at first use and reused across
all subsequent requests — thread-safe and async-safe.
"""
import logging
import os
from typing import Optional

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legal_corpus"

_client: Optional[QdrantClient] = None


def get_qdrant() -> QdrantClient:
    """Return the module-level Qdrant client, creating it on first call."""
    global _client
    if _client is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        logger.info("Initialising Qdrant client → %s:%s", host, port)
        _client = QdrantClient(host=host, port=port)
    return _client
