"""
Node 6 — RecencyCheck

Keyword-only scan. No LLM call. Zero cost.
Detects if the query requires real-time or recent information.
Runs ONLY after a PASS from retrieval_grader_node.

Preserves needs_web_search=True if enhancer already flagged it.
"""
from app.agents.state import LexFindState

_RECENCY_SIGNALS = {
    "recent", "latest", "new", "current", "today", "news", "update",
    "amendment", "ordinance", "gazette", "notification", "this year",
    "this month", "2024", "2025", "2026", "recently", "just", "now",
    "bench", "upcoming", "pending", "introduced", "passed", "enacted",
}


def recency_check_node(state: LexFindState) -> LexFindState:
    """
    Set needs_web_search based on keyword scan of the enhanced query.
    Preserves True if already set by query_enhancement_node.
    """
    if state.get("needs_web_search"):
        return state  # Already flagged by enhancer

    query  = (state.get("enhanced_query") or state["question"]).lower()
    tokens = set(query.split())

    needs_web = bool(tokens & _RECENCY_SIGNALS or any(s in query for s in _RECENCY_SIGNALS))
    return {**state, "needs_web_search": needs_web}
