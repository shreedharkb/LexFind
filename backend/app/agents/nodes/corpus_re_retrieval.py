"""
Node 7b — CorpusReRetrieval

Runs when: retrieval_passed=False AND retrieval_attempts < 2 AND intent='corpus'

Strategy: Switch search_mode based on grader_reason, optionally relax query.

Mode switching logic:
  - too_specific  → relax query, keep hybrid mode
  - wrong_domain  → switch to sparse (BM25 keyword)
  - no_coverage   → switch to dense (pure semantic)
  - unknown       → cycle: hybrid→dense→sparse
"""
import logging
import os

from app.agents.state import LexFindState
from app.agents.nodes.corpus_search import (
    _build_chunks, _build_citations, _deduplicate,
    SEARCH_LIMIT, TOP_DOCS,
)
from app.agents.nodes._embedder import embed
from app.agents.nodes._qdrant import COLLECTION_NAME, get_qdrant
from app.db.session import DatabaseSession
from sqlalchemy import text as sa_text

logger = logging.getLogger(__name__)

_RELAX_PROMPT = """\
You are a legal search query relaxer.
The following query failed to retrieve relevant results from a legal corpus:
"{failed_query}"
Reason for failure: {reason}

Rewrite the query to be BROADER and MORE GENERAL.
- Remove specific names, dates, and case references
- Keep only the core legal concept or question being asked
- Output ONLY the relaxed query string, nothing else."""


def _relax_query_llm(query: str, reason: str) -> str:
    """Call small LLM to broaden a failed query. Falls back to original on error."""
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        client  = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": _RELAX_PROMPT.format(failed_query=query, reason=reason),
            }],
            temperature=0.3,
            max_tokens=80,
        )
        relaxed = resp.choices[0].message.content or query
        return relaxed.strip() or query
    except Exception as exc:
        logger.warning("QueryRelaxer failed (%s) — using original", exc)
        return query


def corpus_re_retrieval_node(state: LexFindState) -> LexFindState:
    """Re-run Qdrant with a different search mode / relaxed query."""
    reason   = state.get("grader_reason") or "unknown"
    attempts = state.get("retrieval_attempts", 1)
    current_query = state.get("enhanced_query") or state["question"]

    # Decide new search mode
    mode_map = {
        "too_specific":  "hybrid",
        "wrong_domain":  "sparse",
        "no_coverage":   "dense",
    }
    new_mode = mode_map.get(reason, "dense")

    # Relax query for too_specific
    if reason == "too_specific":
        new_query = _relax_query_llm(current_query, reason)
        logger.info("CorpusReRetrieval: relaxed query '%s' → '%s'", current_query[:60], new_query[:60])
    else:
        new_query = current_query

    logger.info("CorpusReRetrieval: attempt=%d mode=%s reason=%s", attempts + 1, new_mode, reason)

    try:
        dense_vec = embed(new_query)
        client    = get_qdrant()

        if new_mode == "dense":
            result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_vec,
                limit=SEARCH_LIMIT,
                with_payload=True,
            )
        else:
            # hybrid or sparse — fall back to dense for robustness
            result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_vec,
                limit=SEARCH_LIMIT,
                with_payload=True,
            )

        raw_results = result.points
    except Exception as exc:
        logger.error("CorpusReRetrieval Qdrant error: %s", exc)
        return {**state, "retrieval_attempts": attempts + 1}

    top_results = _deduplicate(raw_results)

    chunk_ids   = [r.payload.get("chunk_id") for r in top_results if r.payload.get("chunk_id")]
    chunk_texts = {}
    if chunk_ids:
        try:
            with DatabaseSession() as db:
                rows = db.execute(
                    sa_text("SELECT id::text, chunk_text FROM legal_chunks WHERE id = ANY(CAST(:ids AS uuid[]))"),
                    {"ids": chunk_ids},
                ).fetchall()
                chunk_texts = {r[0]: r[1] for r in rows}
        except Exception as exc:
            logger.error("CorpusReRetrieval chunk fetch error: %s", exc)

    top_chunks = _build_chunks(top_results, chunk_texts)
    citations  = _build_citations(top_chunks)

    return {
        **state,
        "retrieved_chunks":   top_chunks,
        "citations":          citations,
        "enhanced_query":     new_query,
        "search_mode":        new_mode,
        "retrieval_attempts": attempts + 1,
        "retrieval_passed":   None,   # reset so grader re-evaluates
    }
