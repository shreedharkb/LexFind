"""
Node — GeneralSummary

Handles requests for case summaries, statute summaries, and
legal concept overviews that don't require corpus retrieval.

Runs when classifier routes intent = "general_summary".
No vector search — pure LLM knowledge.
"""
import logging

from app.agents.state import LexFindState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert Indian legal research assistant. "
    "You provide detailed, structured summaries of legal cases, statutes, and concepts "
    "based on your training knowledge. "
    "Structure your response with clear headings. "
    "Always note that your answer is based on training data and users should "
    "consult primary sources for authoritative information."
)


def general_summary_node(state: LexFindState) -> LexFindState:
    """Streaming summary — no retrieval needed."""
    question = state["question"]
    history  = state.get("history", [])

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": question})

    return {
        **state,
        "prompt_messages":  messages,
        "llm_temperature":  0.2,
        "llm_max_tokens":   2000,
        "citations":        [],
        "retrieved_chunks": [],
    }
