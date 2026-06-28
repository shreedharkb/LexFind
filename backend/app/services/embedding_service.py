"""
Embedding Service — single consolidated SentenceTransformer wrapper.

Replaces duplicate embedding code across the application.

"""
import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_EMBEDDING_DIM = 768
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model loaded (%d-dim).", _EMBEDDING_DIM)
    return _model


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vectors.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


EMBEDDING_DIM = _EMBEDDING_DIM
