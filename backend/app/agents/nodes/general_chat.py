"""
GeneralAnswer Node.
Handles pure legal knowledge queries using Groq LLM and conversation history.
"""
import logging
from pathlib import Path

from dotenv import load_dotenv

from app.agents.state import LexFindState

_dotenv_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=_dotenv_path, override=False)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert legal AI assistant specializing in the Indian judicial "
    "system and legal matters. Provide accurate, professional information about "
    "laws, legal processes, court systems, and legal concepts. "
    "Always note that your answers are for informational purposes only "
    "and do not constitute formal legal advice."
)


def general_chat_node(state: LexFindState) -> LexFindState:
    question = state["question"]
    history = state.get("history", [])

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    return {
        **state,
        "prompt_messages": messages,
        "llm_temperature": 0.3,
        "llm_max_tokens": 1024,
        "citations": [],
        "retrieved_chunks": [],
    }
