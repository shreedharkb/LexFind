"""
IntentClassifier Node.
Validates legal context and routes the query appropriately.
"""
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from app.agents.state import LexFindState

_dotenv_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=_dotenv_path, override=False)

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a legal AI classifier for a legal case search system.
Analyse the user question and return a JSON object — nothing else.

Rules for intent:
- "general": pure legal knowledge, no document search needed.
- "document_chat": question is about a specific attached document. Use only if has_documents=true.
- "corpus_search": needs searching across multiple cases.

is_legal must be false for: cooking, sports, weather, math, coding, or anything unrelated to law, legal procedures, courts, or legal concepts.\
"""

_CLASSIFIER_USER_TEMPLATE = """\
Question: {question}
Attached documents: {has_documents}

Return JSON:
{{
  "is_legal": true or false,
  "intent": "general" | "document_chat" | "corpus_search",
  "reasoning": "one line explanation"
}}\
"""

def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)


def classifier_node(state: LexFindState) -> LexFindState:
    explicit_mode = (state.get("explicit_mode") or "auto").lower()
    question = state["question"]
    has_documents = bool(state.get("document_ids"))

    if explicit_mode == "document":
        return {**state, "is_legal": True, "intent": "document_chat"}

    if explicit_mode == "corpus":
        return {**state, "is_legal": True, "intent": "corpus_search"}

    try:
        client = _get_groq_client()
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _CLASSIFIER_USER_TEMPLATE.format(
                        question=question,
                        has_documents=str(has_documents).lower(),
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=100,
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        is_legal = bool(parsed.get("is_legal", True))
        intent = str(parsed.get("intent", "general"))
        
        if intent not in {"general", "document_chat", "corpus_search"}:
            intent = "general"

        if intent == "document_chat" and not has_documents:
            intent = "corpus_search"

        return {**state, "is_legal": is_legal, "intent": intent}

    except Exception as exc:
        logger.error("Classifier error: %s", exc)
        return {**state, "is_legal": True, "intent": "general", "error": f"Classifier failed: {exc}"}


def route_after_classifier(state: LexFindState) -> str:
    if not state.get("is_legal", True):
        return "blocked"
    return state.get("intent", "general")
