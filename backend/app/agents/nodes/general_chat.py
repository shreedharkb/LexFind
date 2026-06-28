"""
GeneralAnswer Node.
Handles pure legal knowledge queries using Groq LLM and conversation history.
"""
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

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


def _clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(System:|Assistant:|AI:|Response:)\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def general_chat_node(state: LexFindState) -> LexFindState:
    question = state["question"]
    history = state.get("history", [])

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    try:
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
            stream=False,
        )

        answer = _clean(response.choices[0].message.content or "")
        return {**state, "answer": answer, "citations": [], "retrieved_chunks": []}

    except Exception as exc:
        logger.error("GeneralChat error: %s", exc)
        return {
            **state,
            "answer": "I encountered an error while processing your request. Please try again.",
            "citations": [],
            "retrieved_chunks": [],
            "error": str(exc),
        }
