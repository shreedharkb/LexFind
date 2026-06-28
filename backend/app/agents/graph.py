"""
LexFind LangGraph State Machine.
Defines the state graph and nodes for processing legal queries.
"""
import logging

from langgraph.graph import END, StateGraph

from app.agents.nodes.classifier import classifier_node, route_after_classifier
from app.agents.nodes.corpus_search import corpus_search_node
from app.agents.nodes.document_chat import document_chat_node
from app.agents.nodes.general_chat import general_chat_node
from app.agents.state import LexFindState

logger = logging.getLogger(__name__)


def _blocked_node(state: LexFindState) -> LexFindState:
    return {
        **state,
        "answer": (
            "I am a specialized legal AI focused on Indian law and the judiciary. "
            "Please ask questions related to law, legal procedures, court systems, "
            "or legal concepts."
        ),
        "citations": [],
        "retrieved_chunks": [],
    }


def build_graph() -> StateGraph:
    graph = StateGraph(LexFindState)

    graph.add_node("classifier", classifier_node)
    graph.add_node("general_chat", general_chat_node)
    graph.add_node("document_chat", document_chat_node)
    graph.add_node("corpus_search", corpus_search_node)
    graph.add_node("blocked", _blocked_node)

    graph.set_entry_point("classifier")

    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "general": "general_chat",
            "document_chat": "document_chat",
            "corpus_search": "corpus_search",
            "blocked": "blocked",
        },
    )

    graph.add_edge("general_chat", END)
    graph.add_edge("document_chat", END)
    graph.add_edge("corpus_search", END)
    graph.add_edge("blocked", END)

    compiled = graph.compile()
    logger.info("LexFind LangGraph compiled successfully.")
    return compiled


lex_graph = build_graph()
