"""
Node 7a — DocumentReRetrieval

Runs when: retrieval_passed=False AND retrieval_attempts < 2 AND intent='document'

Strategy: Relax the enhanced_query and re-run pgvector retrieval on the session documents.
"""
import logging
import os

from app.agents.state import LexFindState

logger = logging.getLogger(__name__)

_RELAX_PROMPT = """\
You are a legal document search query relaxer.
The following query failed to retrieve relevant passages from a specific legal document:
"{failed_query}"
Reason for failure: {reason}

Rewrite the query to be BROADER — focus on the general topic rather than exact phrasing.
Output ONLY the relaxed query, nothing else."""


def _relax_query(query: str, reason: str) -> str:
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        client  = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": _RELAX_PROMPT.format(failed_query=query, reason=reason)}],
            temperature=0.3,
            max_tokens=80,
        )
        return (resp.choices[0].message.content or query).strip() or query
    except Exception as exc:
        logger.warning("DocumentReRetrieval relax failed (%s)", exc)
        return query


def document_re_retrieval_node(state: LexFindState) -> LexFindState:
    """Re-run document retrieval with a broader query."""
    from app.agents.nodes.document_chat import document_chat_node

    reason   = state.get("grader_reason") or "unknown"
    attempts = state.get("retrieval_attempts", 1)
    current_query = state.get("enhanced_query") or state["question"]

    new_query = _relax_query(current_query, reason)
    logger.info("DocumentReRetrieval: attempt=%d relaxed='%s'", attempts + 1, new_query[:60])

    # Re-run document_chat_node with the relaxed enhanced_query
    new_state = {
        **state,
        "enhanced_query":     new_query,
        "retrieval_attempts": attempts + 1,
        "retrieval_passed":   None,
    }
    return document_chat_node(new_state)
