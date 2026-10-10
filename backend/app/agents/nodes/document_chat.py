"""
DocumentChat Node.
Answers queries based on attached documents.
Routes retrieval based on source_type (legal_case vs uploaded).
"""
import logging
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List, Dict

from app.agents.nodes._embedder import embed
from app.agents.state import LexFindState
from app.db.models import Document
from app.db.session import DatabaseSession
from app.services.qdrant_search_service import qdrant_hybrid_search

_dotenv_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=_dotenv_path, override=False)

logger = logging.getLogger(__name__)


TOP_K_QDRANT = 8
TOP_K_PGVECTOR = 8


def _search_qdrant_by_doc(db: Session, doc_id: str, question: str, query_vector: List[float]) -> List[Dict]:
    doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])
    
    points = qdrant_hybrid_search(
        query_text=question,
        qdrant_filter=doc_filter,
        limit=TOP_K_QDRANT,
    )

    chunk_ids = [r.payload.get("chunk_id") for r in points if r.payload.get("chunk_id")]
    chunk_texts = {}
    if chunk_ids:
        rows = db.execute(
            text("SELECT id::text, chunk_text FROM legal_chunks WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": chunk_ids}
        ).fetchall()
        chunk_texts = {r[0]: r[1] for r in rows}

    return [
        {
            "chunk_text": chunk_texts.get(r.payload.get("chunk_id", ""), "") or "",
            "document_id": r.payload.get("document_id") or doc_id,
            "chunk_id": r.payload.get("chunk_id") or "",
            "title": r.payload.get("title") or "Unknown",
            "court": r.payload.get("court") or "",
            "year": r.payload.get("year") or "",
            "citation": r.payload.get("citation") or "",
            "score": r.score,
            "source": "qdrant",
        }
        for r in points
    ]


def _search_pgvector_by_doc(db: Session, doc_id: str, query_vector: List[float]) -> List[Dict]:
    vector_str = "[" + ",".join(f"{v:.6f}" for v in query_vector) + "]"
    sql = text("""
        SELECT
            dc.id AS chunk_id, dc.document_id, d.title AS title, d.blob_path AS blob_path,
            dc.page_number, dc.chunk_text, 1 - (de.embedding <=> CAST(:query_vec AS vector)) AS score
        FROM document_embeddings de
        JOIN document_chunks dc ON dc.id = de.chunk_id
        JOIN documents d ON d.id = de.document_id
        WHERE de.document_id = CAST(:doc_id AS uuid)
        ORDER BY de.embedding <=> CAST(:query_vec AS vector) ASC
        LIMIT :top_k
    """)
    rows = db.execute(sql, {"query_vec": vector_str, "doc_id": doc_id, "top_k": TOP_K_PGVECTOR}).fetchall()
    import os as _os
    return [
        {
            "chunk_text": row.chunk_text, "document_id": str(row.document_id), "chunk_id": str(row.chunk_id),
            "title": row.title, "page_number": row.page_number, "blob_path": _os.path.basename(row.blob_path),
            "score": float(row.score), "source": "pgvector",
        }
        for row in rows
    ]


def _build_citations(chunks: List[Dict]) -> List[Dict]:
    citations = []
    for c in chunks:
        text = c.get("chunk_text", "")
        excerpt = text[:200] + "..." if len(text) > 200 else text
        
        if c.get("source") == "qdrant":
            citations.append({
                "document_id": c.get("document_id"), "chunk_id": c.get("chunk_id"),
                "document_title": c.get("title"), "court": c.get("court"),
                "year": c.get("year"), "citation": c.get("citation"),
                "score": c.get("score"), "excerpt": excerpt,
            })
        else:
            citations.append({
                "document_id": c.get("document_id"), "chunk_id": str(c.get("chunk_id")),
                "document_title": c.get("title"), "page_number": c.get("page_number"),
                "filename": c.get("blob_path"), "score": c.get("score"), "excerpt": excerpt,
            })
    return citations




def document_chat_node(state: LexFindState) -> LexFindState:
    question = state["question"]
    doc_ids = state.get("document_ids", [])

    if not doc_ids:
        return {**state, "answer": "No documents are attached to this session.", "citations": [], "retrieved_chunks": []}

    # Use enhanced_query from query_enhancement_node if available
    search_query = state.get("enhanced_query") or question
    query_vector = embed(search_query)
    all_chunks   = []
    attempts     = state.get("retrieval_attempts", 0)

    try:
        with DatabaseSession() as db:
            for doc_id in doc_ids:
                doc = db.query(Document).filter(Document.id == doc_id).first()
                if not doc: continue

                if doc.source_type == "legal_case":
                    chunks = _search_qdrant_by_doc(db, str(doc_id), question, query_vector)
                else:
                    chunks = _search_pgvector_by_doc(db, str(doc_id), query_vector)
                all_chunks.extend(chunks)
    except Exception as exc:
        logger.error("DocumentChat retrieval error: %s", exc)
        return {**state, "answer": "Retrieval failed. Please try again.", "citations": [], "retrieved_chunks": [], "error": str(exc)}

    if not all_chunks:
        return {
            **state,
            "answer":             "I could not find relevant content in the attached documents for your question.",
            "citations":          [],
            "retrieved_chunks":   [],
            "retrieval_attempts": attempts + 1,
        }

    citations = _build_citations(all_chunks)

    return {
        **state,
        "retrieved_chunks":   all_chunks,
        "citations":          citations,
        "retrieval_attempts": attempts + 1,
        "retrieval_passed":   None,  # grader will evaluate
        "prompt_messages":    None,  # response_generation_node handles this
    }
