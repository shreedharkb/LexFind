"""
LexFind LangGraph State Definition.

This TypedDict is the single source of truth that flows through every node
in the graph. It is ephemeral — one fresh instance is created per request
and discarded after the response is streamed.
"""
from typing import Any, Dict, List, Optional, TypedDict


class LexFindState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    session_id:    str
    user_id:       str
    question:      str
    history:       List[Dict[str, Any]]  # last N turns loaded from messages table
    explicit_mode: Optional[str]          # "auto" | "document" | "corpus"
    document_ids:  List[str]              # doc IDs attached to this session

    # ── Classifier output ──────────────────────────────────────────────────────
    is_legal: bool
    intent:   str   # "general" | "general_summary" | "corpus" | "document" | "blocked"

    # ── Query Enhancement ──────────────────────────────────────────────────────
    enhanced_query:   Optional[str]   # rewritten, standalone search query
    needs_web_search: Optional[bool]  # True if recency signals detected

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieved_chunks:   List[Dict[str, Any]]
    citations:          List[Dict[str, Any]]
    retrieval_attempts: int            # incremented by re-retrieval nodes
    search_mode:        Optional[str]  # "hybrid" | "dense" | "sparse"

    # ── Retrieval Grader ───────────────────────────────────────────────────────
    retrieval_passed: Optional[bool]
    grader_reason:    Optional[str]   # "too_specific" | "wrong_domain" | "no_coverage"

    # ── Web Augmentation ───────────────────────────────────────────────────────
    web_results: List[Dict[str, Any]]

    # ── Response Generation ────────────────────────────────────────────────────
    answer:         str
    system_note:    Optional[str]   # injected disclaimer from graceful_degradation
    confidence_tier: Optional[str]  # "high" | "medium" | "low" | "degraded"
    error:          Optional[str]

    # ── Streaming: nodes populate these; sessions.py streams via AsyncGroq ─────
    prompt_messages: Optional[List[Dict[str, Any]]]
    llm_temperature: Optional[float]
    llm_max_tokens:  Optional[int]
