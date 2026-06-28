"""
Retrieval Service — pgvector-based RAG retrieval scoped to session documents.

"""
import logging
import os
import uuid
from typing import List

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.embedding_service import embed_query
from app.db.crud.session_document_repository import get_attached_document_ids

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 8
MIN_SIMILARITY = 0.20


class RetrievedChunk:
    def __init__(self, chunk_id: str, document_id: str, document_title: str, document_filename: str, page_number: int, chunk_text: str, similarity: float):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_title = document_title
        self.document_filename = document_filename
        self.page_number = page_number
        self.chunk_text = chunk_text
        self.similarity = similarity

    def as_context_block(self) -> str:
        return f"--- SOURCE: {self.document_title}, Page {self.page_number} ---\n{self.chunk_text}\n--- END SOURCE ---"

    def as_citation(self) -> dict:
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "document_filename": self.document_filename,
            "page_number": self.page_number,
            "excerpt": self.chunk_text[:200],
        }


def retrieve_for_session(db: Session, session_id: uuid.UUID, query: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievedChunk]:
    document_ids = get_attached_document_ids(db, session_id)
    if not document_ids:
        return []

    query_vector = embed_query(query)
    vector_str = "[" + ",".join(f"{v:.6f}" for v in query_vector.tolist()) + "]"
    doc_id_strings = [str(d) for d in document_ids]

    sql = text("""
        SELECT
            dc.id AS chunk_id, dc.document_id, d.title AS document_title, d.blob_path AS document_blob_path,
            dc.page_number, dc.chunk_text,
            1 - (de.embedding <=> CAST(:query_vec AS vector)) AS similarity
        FROM document_embeddings de
        JOIN document_chunks dc ON dc.id = de.chunk_id
        JOIN documents d ON d.id = de.document_id
        WHERE de.document_id = ANY(CAST(:doc_ids AS uuid[]))
        ORDER BY de.embedding <=> CAST(:query_vec AS vector) ASC
        LIMIT :top_k
    """)

    rows = db.execute(sql, {"query_vec": vector_str, "doc_ids": "{" + ",".join(doc_id_strings) + "}", "top_k": top_k}).fetchall()

    return [
        RetrievedChunk(
            chunk_id=str(r.chunk_id), document_id=str(r.document_id), document_title=r.document_title,
            document_filename=os.path.basename(r.document_blob_path), page_number=r.page_number,
            chunk_text=r.chunk_text, similarity=float(r.similarity)
        )
        for r in rows
    ]


def build_context_prompt(chunks: List[RetrievedChunk]) -> str:
    if not chunks: return ""
    return "\n\n".join(c.as_context_block() for c in chunks)
