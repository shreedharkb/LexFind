"""
LexFind LangGraph State Machine — Agentic RAG Pipeline.

Graph topology:
    classifier
        ├── blocked          → blocked_response                          → END
        ├── general          → general_chat                              → END
        ├── general_summary  → general_summary                           → END
        ├── document_chat    → query_enhancement → document_retrieval ──┐
        └── corpus_search    → query_enhancement → qdrant_retrieval    ──┤
                                                                          ↓
                                                                 retrieval_grader
                                                                /         |         \\
                                                     (pass)   /    (fail, attempts<2)  (exhausted)
                                                             /      re-retrieval         \\
                                                     recency_check  (corpus/doc)     graceful_degradation
                                                    /           \\       ↓                   |
                                              web_augment  response_gen ← ← ← ← ← ← response_gen
                                                   \\              /
                                                  response_generation → END
"""
import logging

from langgraph.graph import END, StateGraph

from app.agents.nodes.classifier import classifier_node, route_after_classifier
from app.agents.nodes.corpus_re_retrieval import corpus_re_retrieval_node
from app.agents.nodes.corpus_search import corpus_search_node
from app.agents.nodes.document_chat import document_chat_node
from app.agents.nodes.document_re_retrieval import document_re_retrieval_node
from app.agents.nodes.general_chat import general_chat_node
from app.agents.nodes.general_summary import general_summary_node
from app.agents.nodes.graceful_degradation import graceful_degradation_node
from app.agents.nodes.query_enhancement import query_enhancement_node
from app.agents.nodes.recency_check import recency_check_node
from app.agents.nodes.response_generation import response_generation_node
from app.agents.nodes.retrieval_grader import retrieval_grader_node
from app.agents.nodes.web_augment import web_augment_node
from app.agents.state import LexFindState

logger = logging.getLogger(__name__)


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_enhancement(state: LexFindState) -> str:
    """After query enhancement, route to the correct retrieval node."""
    intent = state.get("intent", "corpus_search")
    return "document_retrieval" if intent == "document_chat" else "qdrant_retrieval"


def route_after_grader(state: LexFindState) -> str:
    """
    After retrieval grading:
      - passed  → recency_check
      - failed, attempts < 2, corpus → corpus_re_retrieval
      - failed, attempts < 2, document → document_re_retrieval
      - failed, attempts >= 2 → graceful_degradation
    """
    passed   = state.get("retrieval_passed", True)
    attempts = state.get("retrieval_attempts", 1)
    intent   = state.get("intent", "corpus_search")

    if passed:
        return "recency_check"
    if attempts < 2:
        return "corpus_re_retrieval" if intent != "document_chat" else "document_re_retrieval"
    return "graceful_degradation"


def route_after_recency(state: LexFindState) -> str:
    """After recency check: web augment if needed, else straight to response."""
    if state.get("needs_web_search"):
        return "web_augment"
    return "response_generation"


def _blocked_node(state: LexFindState) -> LexFindState:
    return {
        **state,
        "answer": (
            "I am a specialized legal AI focused on Indian law and the judiciary. "
            "Please ask questions related to law, legal procedures, court systems, "
            "or legal concepts."
        ),
        "citations":        [],
        "retrieved_chunks": [],
        "prompt_messages":  None,
    }


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(LexFindState)

    # Nodes
    graph.add_node("classifier",            classifier_node)
    graph.add_node("general_chat",          general_chat_node)
    graph.add_node("general_summary",       general_summary_node)
    graph.add_node("blocked",               _blocked_node)
    graph.add_node("query_enhancement",     query_enhancement_node)
    graph.add_node("qdrant_retrieval",      corpus_search_node)
    graph.add_node("document_retrieval",    document_chat_node)
    graph.add_node("retrieval_grader",      retrieval_grader_node)
    graph.add_node("corpus_re_retrieval",   corpus_re_retrieval_node)
    graph.add_node("document_re_retrieval", document_re_retrieval_node)
    graph.add_node("recency_check",         recency_check_node)
    graph.add_node("web_augment",           web_augment_node)
    graph.add_node("graceful_degradation",  graceful_degradation_node)
    graph.add_node("response_generation",   response_generation_node)

    # Entry point
    graph.set_entry_point("classifier")

    # Classifier → route
    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "general":         "general_chat",
            "general_summary": "general_summary",
            "document_chat":   "query_enhancement",
            "corpus_search":   "query_enhancement",
            "blocked":         "blocked",
        },
    )

    # Query enhancement → retrieval
    graph.add_conditional_edges(
        "query_enhancement",
        route_after_enhancement,
        {
            "qdrant_retrieval":   "qdrant_retrieval",
            "document_retrieval": "document_retrieval",
        },
    )

    # Retrieval → grader
    graph.add_edge("qdrant_retrieval",   "retrieval_grader")
    graph.add_edge("document_retrieval", "retrieval_grader")

    # Grader → pass/fail/exhausted
    graph.add_conditional_edges(
        "retrieval_grader",
        route_after_grader,
        {
            "recency_check":         "recency_check",
            "corpus_re_retrieval":   "corpus_re_retrieval",
            "document_re_retrieval": "document_re_retrieval",
            "graceful_degradation":  "graceful_degradation",
        },
    )

    # Re-retrieval → grader again
    graph.add_edge("corpus_re_retrieval",   "retrieval_grader")
    graph.add_edge("document_re_retrieval", "retrieval_grader")

    # Recency check → web augment or response generation
    graph.add_conditional_edges(
        "recency_check",
        route_after_recency,
        {
            "web_augment":        "web_augment",
            "response_generation": "response_generation",
        },
    )

    # Web augment → response generation
    graph.add_edge("web_augment",           "response_generation")
    graph.add_edge("graceful_degradation",  "response_generation")

    # Terminal nodes
    graph.add_edge("general_chat",       END)
    graph.add_edge("general_summary",    END)
    graph.add_edge("blocked",            END)
    graph.add_edge("response_generation", END)

    compiled = graph.compile()
    logger.info("LexFind LangGraph compiled successfully (full 13-node pipeline).")
    return compiled


lex_graph = build_graph()
