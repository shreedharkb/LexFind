"""
Node 5 — RetrievalGrader

LLM-based grader that evaluates whether retrieved chunks are actually
relevant to the enhanced query.

Uses openai/gpt-oss-20b (fast small model, binary output).

Failure policy: On grader failure default to retrieval_passed=True.
A false positive is recoverable (weak answer). A false negative wastes a
re-retrieval attempt and degrades UX unnecessarily.
"""
import json
import logging
from app.agents.state import LexFindState
from app.core.llm import get_groq_client

logger = logging.getLogger(__name__)

_GRADER_PROMPT = """\
You are a Retrieval Relevance Grader for a legal research system.

Search Query: {search_query}

Retrieved Documents:
{documents_text}

Your task: Determine if the retrieved documents contain information that is relevant
to answering the search query.

If relevant: the documents contain facts, cases, judgments, legal principles, or passages
that directly help answer the query.
If not relevant: the documents are about a completely different topic, too generic,
or contain no useful information for this query.

Also, if NOT relevant, classify the reason:
- "too_specific": the query was too narrow or contained specific names/dates that didn't match anything
- "wrong_domain": the documents returned are from a different area of law or topic entirely
- "no_coverage": the topic genuinely does not appear to exist in the database

Respond ONLY with this exact JSON — nothing else:
{{"relevant": true}}
OR
{{"relevant": false, "reason": "too_specific"}}"""


def retrieval_grader_node(state: LexFindState) -> LexFindState:
    """Grade retrieved chunks for relevance. Binary pass/fail."""
    chunks        = state.get("retrieved_chunks", [])
    enhanced_q    = state.get("enhanced_query") or state["question"]

    if not chunks:
        logger.warning("RetrievalGrader: no chunks to grade — fail")
        return {**state, "retrieval_passed": False, "grader_reason": "no_coverage"}

    # Build short text of retrieved docs for grader
    documents_text = "\n\n".join(
        f"[Doc {i+1}] {c.get('title', 'Unknown')} ({c.get('year', '')})\n{c.get('chunk_text', '')[:300]}"
        for i, c in enumerate(chunks[:5])
    )

    prompt = _GRADER_PROMPT.format(
        search_query=enhanced_q,
        documents_text=documents_text,
    )

    try:
        client = get_groq_client()
        if not client: raise ValueError("Groq client not configured")
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
            response_format={"type": "json_object"},
        )
        raw    = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        passed = bool(parsed.get("relevant", True))
        reason = parsed.get("reason", "") if not passed else None
    except Exception as exc:
        logger.warning("RetrievalGrader failed (%s) — defaulting to pass", exc)
        passed = True
        reason = None

    logger.info("RetrievalGrader: passed=%s reason=%s", passed, reason)
    return {**state, "retrieval_passed": passed, "grader_reason": reason}
