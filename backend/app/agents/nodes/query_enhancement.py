"""
Node 3 — QueryEnhancement

Runs for both 'document' and 'corpus' intents BEFORE retrieval.
Single LLM call. Two outputs:
  - enhanced_query: standalone resolved search query
  - needs_web_search: bool, True if recency signals detected

Uses llama-3.1-8b-instant (fast small model) — never the main generation model.

Failure policy: On any exception, fall back to raw question and set
needs_web_search=False. Never crash the pipeline or pass empty string.
"""
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from app.agents.state import LexFindState

_dotenv_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=_dotenv_path, override=False)

logger = logging.getLogger(__name__)

# ── Differentiated instructions per intent ──────────────────────────────────

_DOCUMENT_INSTRUCTION = """\
You are a legal search query enhancer for a document chat system.
The user is asking about their specific uploaded legal document.
Using the chat history for context, rewrite their question as a PRECISE, FOCUSED, standalone search query.
Do NOT broaden the query. Do NOT add concepts not already in the conversation.
The query must retrieve a specific passage from this specific document."""

_CORPUS_INSTRUCTION = """\
You are a legal search query enhancer for a legal case database.
The user is searching a corpus of Indian legal cases and judgments.
Using the chat history for context, rewrite their question as a clear, standalone search query.
CRITICAL RULES:
1. If the user asks for "similar cases" to a previous case, DO NOT just copy the case title. Instead, extract the underlying material facts and legal concepts from the chat history to form the query.
2. For direct lookups, if the user explicitly wants to find a specific case name, judge, or citation, preserve it.
3. The backend uses Hybrid Search. Keep the query focused on the core legal issues and facts rather than conversational filler."""

_PROMPT_TEMPLATE = """\
{instruction}

CHAT HISTORY (last 5 turns):
{history_text}

CURRENT USER MESSAGE:
{question}

Respond with ONLY this JSON — no other text:
{{
  "enhanced_query": "<rewritten standalone search query>",
  "needs_web_search": <true if query requires recent/real-time info, else false>
}}"""

_RECENCY_KEYWORDS = {
    "recent", "latest", "new", "current", "today", "news", "update",
    "amendment", "ordinance", "gazette", "notification", "this year",
    "this month", "2024", "2025", "2026", "recently", "just", "now",
    "upcoming", "pending", "introduced", "passed", "enacted",
}


def query_enhancement_node(state: LexFindState) -> LexFindState:
    """Rewrite query for retrieval. Fast small LLM call."""
    question = state["question"]
    intent   = state.get("intent", "corpus")
    history  = state.get("history", [])

    instruction = _DOCUMENT_INSTRUCTION if intent == "document" else _CORPUS_INSTRUCTION

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in history[-10:]
    ) or "(no prior history)"

    prompt = _PROMPT_TEMPLATE.format(
        instruction=instruction,
        history_text=history_text,
        question=question,
    )

    try:
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        enhanced_query   = str(parsed.get("enhanced_query") or question).strip() or question
        needs_web_search = bool(parsed.get("needs_web_search", False))
    except Exception as exc:
        logger.warning("QueryEnhancement failed (%s) — falling back to raw question", exc)
        enhanced_query   = question
        needs_web_search = any(kw in question.lower() for kw in _RECENCY_KEYWORDS)

    logger.info("QueryEnhancement: '%s' → '%s' (web=%s)", question[:60], enhanced_query[:60], needs_web_search)

    return {
        **state,
        "enhanced_query":   enhanced_query,
        "needs_web_search": needs_web_search,
    }
