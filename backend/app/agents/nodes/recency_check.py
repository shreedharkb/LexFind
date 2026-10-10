"""
Node 6 — RecencyCheck

Keyword-only scan. No LLM call. Zero cost.
Detects if the query requires real-time or recent information.
Runs ONLY after a PASS from retrieval_grader_node.

Preserves needs_web_search=True if enhancer already flagged it.
"""
import re
from app.agents.state import LexFindState

_RECENCY_SIGNALS = {
    "recent", "latest", "current", "today", "news", "update",
    "amendment", "ordinance", "gazette", "notification", "this year",
    "this month", "2024", "2025", "2026", "recently",
    "upcoming", "pending", "introduced", "passed", "enacted",
}


def recency_check_node(state: LexFindState) -> LexFindState:
    """
    Set needs_web_search based on keyword scan of the enhanced query.
    Preserves True if already set by query_enhancement_node.
    """
    if state.get("needs_web_search"):
        return state  # Already flagged by enhancer

    query  = (state.get("enhanced_query") or state["question"]).lower()
    needs_web = bool(re.search(r'\b(' + '|'.join(re.escape(s) for s in _RECENCY_SIGNALS) + r')\b', query))
    return {**state, "needs_web_search": needs_web}
