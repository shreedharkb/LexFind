"""
Node 10 — ResponseGeneration

Unified response generation node. Replaces the inline prompt_messages population
previously done inside corpus_search_node and document_chat_node.

Handles:
  - retrieved_chunks from Qdrant (corpus) or pgvector (document)
  - web_results from web_augment_node (optional)
  - system_note injection from graceful_degradation_node (optional)
  - confidence_tier for logging / telemetry

Uses the main Groq model (llama-3.3-70b-versatile) — quality matters here.
"""
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from app.agents.state import LexFindState

_dotenv_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=_dotenv_path, override=False)

logger = logging.getLogger(__name__)

_CORPUS_SYSTEM = (
    "You are a senior Indian legal expert. You are given excerpts from multiple "
    "Supreme Court judgments. Synthesise a comprehensive answer that references "
    "specific cases. When citing cases use [Case Name (Year)]. "
    "If cases have conflicting holdings, explain the evolution of law. "
    "Be precise and professional."
)

_DOCUMENT_SYSTEM = (
    "You are an expert legal assistant. You are given excerpts from a legal document "
    "the user has uploaded. Answer the user's question based strictly on the document content. "
    "Quote relevant passages where helpful. If the answer is not in the document, say so clearly."
)


def _build_context_block(chunks: list, web_results: list, system_note: str | None) -> str:
    """Build the full context section for the prompt."""
    parts = []

    if system_note:
        parts.append(f"IMPORTANT DISCLAIMER:\n{system_note}")

    db_chunks = [c for c in chunks if c.get("source") != "web"]
    if db_chunks:
        db_text = "\n\n---\n\n".join(
            f"[DATABASE RESULT {i+1}]\n"
            f"Case: {c.get('title', 'Unknown')} ({c.get('year', '')})\n"
            f"Court: {c.get('court', '')}\n"
            f"Citation: {c.get('citation', '')}\n"
            f"Excerpt:\n{c.get('chunk_text', '')}"
            for i, c in enumerate(db_chunks)
        )
        parts.append(f"DATABASE RESULTS:\n{db_text}")
    else:
        parts.append("DATABASE RESULTS:\nNo relevant database results found.")

    if web_results:
        web_text = "\n\n".join(
            f"[WEB {i+1}] {r.get('title', '')} ({r.get('url', '')})\n{r.get('content', '')}"
            for i, r in enumerate(web_results)
        )
        parts.append(f"RECENT WEB RESULTS:\n{web_text}")

    return "\n\n".join(parts)


def response_generation_node(state: LexFindState) -> LexFindState:
    """Build prompt_messages for streaming by sessions.py."""
    intent      = state.get("intent", "corpus")
    question    = state["question"]
    enhanced_q  = state.get("enhanced_query") or question
    history     = state.get("history", [])
    chunks      = state.get("retrieved_chunks", [])
    web_results = state.get("web_results", [])
    system_note = state.get("system_note")

    system_prompt = _DOCUMENT_SYSTEM if intent == "document" else _CORPUS_SYSTEM
    context_block = _build_context_block(chunks, web_results, system_note)

    user_prompt = (
        f"{context_block}\n\n"
        f"QUESTION: {enhanced_q}\n\n"
        f"Provide a comprehensive, well-structured answer:"
    )

    messages = [{"role": "system", "content": system_prompt}]
    # Include last 4 history turns for context
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_prompt})

    logger.info(
        "ResponseGeneration: intent=%s chunks=%d web=%d degraded=%s",
        intent, len(chunks), len(web_results), bool(system_note),
    )

    return {
        **state,
        "answer":          "",
        "prompt_messages": messages,
        "llm_temperature": 0.2,
        "llm_max_tokens":  1800,
        "confidence_tier": state.get("confidence_tier") or ("high" if chunks else "low"),
    }
