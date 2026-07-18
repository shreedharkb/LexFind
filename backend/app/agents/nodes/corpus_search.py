"""
CorpusSearch Node.
Searches the full Qdrant corpus and synthesises a multi-case answer.
"""
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from qdrant_client.http.models import Fusion, FusionQuery, Prefetch, SparseVector
from sqlalchemy import text

from app.agents.nodes._embedder import embed
from app.services.qdrant_search_service import _embed_sparse
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


def _clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(System:|Assistant:|AI:|Response:)\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _deduplicate(results: list) -> list:
    seen = {}
    for r in results:
        doc_id = r.payload.get("document_id")
        if doc_id and (doc_id not in seen or r.score > seen[doc_id].score):
            seen[doc_id] = r
    top = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    return top[:TOP_DOCS]


def _build_chunks(results: list, chunk_texts: dict) -> list[dict]:
    return [
        {
            "chunk_text": chunk_texts.get(r.payload.get("chunk_id", ""), "") or "",
            "document_id": r.payload.get("document_id") or "",
            "chunk_id": r.payload.get("chunk_id") or "",
            "title": r.payload.get("title") or "Unknown",
            "court": r.payload.get("court") or "",
            "year": r.payload.get("year") or "",
            "citation": r.payload.get("citation") or "",
            "score": r.score,
        }
        for r in results
    ]


def _build_citations(chunks: list[dict]) -> list[dict]:
    citations = []
    for c in chunks:
        text = c.get("chunk_text", "")
        excerpt = text[:200] + "..." if len(text) > 200 else text
        citations.append({
            "document_id": c["document_id"],
            "chunk_id": c["chunk_id"],
            "document_title": c["title"],
            "court": c["court"],
            "year": c["year"],
            "citation": c["citation"],
            "score": c["score"],
            "excerpt": excerpt,
        })
    return citations


def _build_context(chunks: list[dict]) -> str:
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
    question = state["question"]
    history = state.get("history", [])

    try:
        dense_vec = embed(question)
        client = get_qdrant()

        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vec,
            limit=SEARCH_LIMIT,
            with_payload=True,
        )
        raw_results = result.points
    except Exception as exc:
        logger.error("CorpusSearch Qdrant error: %s", exc)
        return {**state, "answer": "Search failed. Please try again.", "citations": [], "retrieved_chunks": [], "error": str(exc)}

    top_results = _deduplicate(raw_results)
    
    chunk_ids = [r.payload.get("chunk_id") for r in top_results if r.payload.get("chunk_id")]
    chunk_texts = {}
    if chunk_ids:
        try:
            with DatabaseSession() as db:
                rows = db.execute(
                    text("SELECT id::text, chunk_text FROM legal_chunks WHERE id = ANY(CAST(:ids AS uuid[]))"),
                    {"ids": chunk_ids}
                ).fetchall()
                chunk_texts = {r[0]: r[1] for r in rows}
        except Exception as exc:
            logger.error("Failed to fetch chunk texts from postgres: %s", exc)

    top_chunks = _build_chunks(top_results, chunk_texts)
    citations = _build_citations(top_chunks)

    if not top_chunks:
        return {**state, "answer": "I could not find relevant cases in the corpus for your question.", "citations": [], "retrieved_chunks": []}

    context = _build_context(top_chunks)
    user_prompt = f"Context from Supreme Court cases:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    try:
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        groq = Groq(api_key=api_key)

        response = groq.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1800,
            stream=False,
        )

        answer = _clean(response.choices[0].message.content or "")
        return {**state, "answer": answer, "citations": citations, "retrieved_chunks": top_chunks}

    except Exception as exc:
        logger.error("CorpusSearch LLM error: %s", exc)
        return {**state, "answer": "The AI encountered an error. Please try again.", "citations": citations, "retrieved_chunks": top_chunks, "error": str(exc)}
