"""
CorpusSearch Node (qdrant_retrieval_node).
Searches the full Qdrant corpus and builds retrieved_chunks.
The LLM call is handled downstream by response_generation_node.
"""
import logging
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client.http.models import Fusion, FusionQuery, Prefetch, SparseVector
from sqlalchemy import text

from app.agents.nodes._embedder import embed
from app.agents.nodes._qdrant import COLLECTION_NAME, get_qdrant
from app.agents.state import LexFindState
from app.db.session import DatabaseSession

_dotenv_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=_dotenv_path, override=False)

logger = logging.getLogger(__name__)

SEARCH_LIMIT = 15
TOP_DOCS = 5

_SYSTEM_PROMPT = (
    "You are a senior Indian legal expert. You are given excerpts from multiple "
    "Supreme Court judgments. Synthesise a comprehensive answer that references "
    "specific cases. When citing cases use [Case Name (Year)]. "
    "If cases have conflicting holdings, explain the evolution of law. "
    "Be precise and professional."
)


def _deduplicate(results: list) -> list:
    seen = {}
    for r in results:
        doc_id = r.payload.get("document_id")
        if doc_id and (doc_id not in seen or r.score > seen[doc_id].score):
            seen[doc_id] = r
    top = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    return top[:TOP_DOCS]


def _build_chunks(results: list, chunk_texts: dict) -> list:
    return [
        {
            "chunk_text":  chunk_texts.get(r.payload.get("chunk_id", ""), "") or "",
            "document_id": r.payload.get("document_id") or "",
            "chunk_id":    r.payload.get("chunk_id") or "",
            "title":       r.payload.get("title") or "Unknown",
            "court":       r.payload.get("court") or "",
            "year":        r.payload.get("year") or "",
            "citation":    r.payload.get("citation") or "",
            "score":       r.score,
        }
        for r in results
    ]


def _build_citations(chunks: list) -> list:
    citations = []
    for c in chunks:
        txt = c.get("chunk_text", "")
        citations.append({
            "document_id":    c["document_id"],
            "chunk_id":       c["chunk_id"],
            "document_title": c["title"],
            "court":          c["court"],
            "year":           c["year"],
            "citation":       c["citation"],
            "score":          c["score"],
            "excerpt":        txt[:200] + "..." if len(txt) > 200 else txt,
        })
    return citations


def _build_context(chunks: list) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"Case: {c['title']} ({c['year']})\n"
            f"Court: {c['court']}\n"
            f"Citation: {c['citation']}\n"
            f"Excerpt: {c['chunk_text']}"
        )
    return "\n\n---\n\n".join(parts)


def corpus_search_node(state: LexFindState) -> LexFindState:
    """Retrieve relevant chunks from Qdrant. Uses enhanced_query if available."""
    # Use enhanced_query from query_enhancement_node if available, else raw question
    search_query = state.get("enhanced_query") or state["question"]
    attempts     = state.get("retrieval_attempts", 0)

    try:
        dense_vec = embed(search_query)
        client    = get_qdrant()
        result    = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vec,
            limit=SEARCH_LIMIT,
            with_payload=True,
        )
        raw_results = result.points
    except Exception as exc:
        logger.error("CorpusSearch Qdrant error: %s", exc)
        return {
            **state,
            "answer":           "Search failed. Please try again.",
            "citations":        [],
            "retrieved_chunks": [],
            "retrieval_attempts": attempts + 1,
            "error":            str(exc),
        }

    top_results = _deduplicate(raw_results)

    chunk_ids   = [r.payload.get("chunk_id") for r in top_results if r.payload.get("chunk_id")]
    chunk_texts = {}
    if chunk_ids:
        try:
            with DatabaseSession() as db:
                rows = db.execute(
                    text("SELECT id::text, chunk_text FROM legal_chunks WHERE id = ANY(CAST(:ids AS uuid[]))"),
                    {"ids": chunk_ids},
                ).fetchall()
                chunk_texts = {r[0]: r[1] for r in rows}
        except Exception as exc:
            logger.error("Failed to fetch chunk texts from postgres: %s", exc)

    top_chunks = _build_chunks(top_results, chunk_texts)
    citations  = _build_citations(top_chunks)

    return {
        **state,
        "retrieved_chunks":   top_chunks,
        "citations":          citations,
        "retrieval_attempts": attempts + 1,
        "retrieval_passed":   None,  # will be set by retrieval_grader_node
        # Remove prompt_messages — response_generation_node handles that now
        "prompt_messages":    None,
    }


# Alias used by corpus_re_retrieval_node
qdrant_retrieval_node = corpus_search_node
