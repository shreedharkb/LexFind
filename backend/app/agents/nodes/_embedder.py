"""
Shared embedding helper for all agent nodes.

Delegates to the project-wide EmbeddingService so the SentenceTransformer
model is only ever loaded ONCE across the entire process (singleton).
"""
from app.services.embedding_service import embed_query as _embed_query


def embed(text: str) -> list[float]:
    """
    Encode a single string into a 768-dim float list suitable for Qdrant search.

    Reuses the existing project-wide SentenceTransformer singleton
    (sentence-transformers/all-mpnet-base-v2) so the model is not
    loaded again.
    """
    vector = _embed_query(text)
    return vector.tolist()
