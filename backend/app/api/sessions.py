"""
Sessions API Router — LexFind.

Handles AssistantSessions, their Messages (including SSE streaming for AI responses),
and attaching/detaching Documents.

Streaming architecture:
  1. Run lex_graph.invoke() in a thread pool — handles classification + retrieval.
     Nodes set state["prompt_messages"] instead of calling the LLM directly.
  2. Stream the LLM response token-by-token via AsyncGroq (stream=True).
  3. Emit each token as an SSE chunk so the frontend renders progressively.
"""
import asyncio
import json
import logging
import os
import re
import uuid
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from sqlalchemy.orm import Session as DBSession
from starlette.concurrency import run_in_threadpool

from app.api.dependencies.auth import get_current_user
from app.db.session import DatabaseSession, get_db
from app.db.crud import session_repository as session_repo
from app.db.crud import message_repository as message_repo
from app.db.crud import document_repository as doc_repo
from app.db.crud import session_document_repository as sd_repo
from app.agents import lex_graph, LexFindState
from app.core.rate_limiter import check_chat_rate
from app.schemas.sessions import (
    SessionCreate,
    SessionListItem,
    SessionRenameRequest,
    SessionResponse,
)
from app.schemas.messages import MessageCreate, MessageResponse
from app.schemas.documents import AttachDocumentRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["Sessions"])


def _clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(System:|Assistant:|AI:|Response:)\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def _generate_session_title(question: str, answer: str) -> str:
    """Call Groq to generate a concise 4-6 word session title."""
    try:
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = AsyncGroq(api_key=api_key)
        prompt = (
            f"Create a concise 4-6 word title for a legal chat session. "
            f"User asked: \"{question[:200]}\"\n"
            f"Reply with ONLY the title — no quotes, no punctuation at the end."
        )
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=20,
        )
        title = resp.choices[0].message.content or ""
        title = title.strip().strip('"').strip("'").strip(".")
        return title[:60] if title else question[:40]
    except Exception as exc:
        logger.warning("Title generation failed: %s", exc)
        return question[:40]


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: SessionCreate,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    session = session_repo.create_session(db, uuid.UUID(user_id), request.title)
    return session


@router.get("", response_model=List[SessionListItem])
async def list_sessions(
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    return session_repo.list_sessions(db, uuid.UUID(user_id))


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    session = session_repo.get_session_for_user(db, session_id, uuid.UUID(user_id))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/{session_id}", response_model=SessionResponse)
async def rename_session(
    session_id: UUID,
    request: SessionRenameRequest,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    session = session_repo.get_session_for_user(db, session_id, uuid.UUID(user_id))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_repo.rename_session(db, session, request.title)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    session = session_repo.get_session_for_user(db, session_id, uuid.UUID(user_id))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session_repo.delete_session(db, session)


@router.post("/{session_id}/documents", status_code=status.HTTP_201_CREATED)
async def attach_document(
    session_id: UUID,
    request: AttachDocumentRequest,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    session = session_repo.get_session_for_user(db, session_id, uuid.UUID(user_id))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    doc = doc_repo.get_document(db, request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    sd_repo.attach_document(db, session.id, doc.id)
    return {"status": "attached"}


@router.delete("/{session_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def detach_document(
    session_id: UUID,
    document_id: UUID,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    session = session_repo.get_session_for_user(db, session_id, uuid.UUID(user_id))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not sd_repo.detach_document(db, session.id, document_id):
        raise HTTPException(status_code=404, detail="Document not attached to session")


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    session_id: UUID,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    session = session_repo.get_session_for_user(db, session_id, uuid.UUID(user_id))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return message_repo.list_messages(db, session.id)


@router.post("/{session_id}/messages")
async def send_message(
    session_id: UUID,
    request: MessageCreate,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    session = session_repo.get_session_for_user(db, session_id, uuid.UUID(user_id))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Rate limiting: 30 messages per minute per user
    if not check_chat_rate(user_id):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment before sending another message.")

    question = request.content.strip()
    is_first_message = session.title == "New Session"

    message_repo.append_message(db, session.id, "user", question)

    # Placeholder title while the LLM generates a real one
    if is_first_message:
        placeholder = (question[:40] + "...") if len(question) > 40 else question
        session_repo.rename_session(db, session, placeholder)
    else:
        session_repo.touch_session(db, session)

    recent_msgs = message_repo.get_recent_messages(db, session.id, n=6)
    history = [{"role": m.role, "content": m.content} for m in recent_msgs]

    doc_ids = [str(doc_id) for doc_id in sd_repo.get_attached_document_ids(db, session.id)]
    explicit_mode = getattr(request, "explicit_mode", None) or "auto"

    initial_state: LexFindState = {
        "session_id":    str(session.id),
        "user_id":       str(user_id),
        "question":      question,
        "history":       history,
        "explicit_mode": explicit_mode,
        "document_ids":  doc_ids,
        # Classifier
        "is_legal":      False,
        "intent":        "",
        # Query enhancement
        "enhanced_query":   None,
        "needs_web_search": False,
        # Retrieval
        "retrieved_chunks":   [],
        "citations":          [],
        "retrieval_attempts": 0,
        "search_mode":        "hybrid",
        # Grader
        "retrieval_passed": None,
        "grader_reason":    None,
        # Web augmentation
        "web_results": [],
        # Response
        "answer":          "",
        "system_note":     None,
        "confidence_tier": None,
        "error":           None,
        # Streaming
        "prompt_messages": None,
        "llm_temperature": None,
        "llm_max_tokens":  None,
    }

    async def generate():
        final_answer    = ""
        final_citations = []

        try:
            # Step 1: Run graph in threadpool (sync nodes do retrieval + classification).
            # Nodes populate state["prompt_messages"] instead of calling the LLM.
            final_state = await run_in_threadpool(lex_graph.invoke, initial_state)

            final_citations = final_state.get("citations", [])
            prompt_messages = final_state.get("prompt_messages")

            # Step 2a: Graph produced prompt_messages — stream via AsyncGroq token-by-token.
            if prompt_messages:
                api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
                model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                temperature = final_state.get("llm_temperature") or 0.3
                max_tokens  = final_state.get("llm_max_tokens")  or 1024

                groq_client = AsyncGroq(api_key=api_key)
                stream = await groq_client.chat.completions.create(
                    model=model,
                    messages=prompt_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )

                async for chunk in stream:
                    token = chunk.choices[0].delta.content if chunk.choices else None
                    if token:
                        final_answer += token
                        yield f"data: {json.dumps({'content': token})}\n\n"

            # Step 2b: Node returned a direct answer (blocked / error path).
            else:
                direct_answer = final_state.get("answer", "")
                if direct_answer:
                    final_answer = direct_answer
                    yield f"data: {json.dumps({'content': direct_answer})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': 'No response was generated. Please try again.'})}\n\n"

        except Exception as exc:
            logger.error("Streaming error: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'content': '[Server busy. Please try again in a moment.]'})}\n\n"

        yield "data: [DONE]\n\n"

        if final_citations:
            yield f"event: citations\ndata: {json.dumps(final_citations)}\n\n"

        # Persist assistant message
        if final_answer:
            final_answer = _clean(final_answer)
            with DatabaseSession() as write_db:
                message_repo.append_message(
                    write_db,
                    session.id,
                    "assistant",
                    final_answer,
                    citations=final_citations or None,
                )

                # Generate a smart LLM session title after the first exchange
                if is_first_message:
                    smart_title = await _generate_session_title(question, final_answer)
                    s = session_repo.get_session_for_user(write_db, session.id, uuid.UUID(user_id))
                    if s:
                        session_repo.rename_session(write_db, s, smart_title)

    return StreamingResponse(generate(), media_type="text/event-stream")
