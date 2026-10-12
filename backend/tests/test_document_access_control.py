import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.agents.nodes import document_chat
from app.api import sessions


def test_attach_document_rejects_another_users_upload(monkeypatch):
    owner_id = uuid.uuid4()
    other_user_id = uuid.uuid4()
    session = SimpleNamespace(id=uuid.uuid4())
    document = SimpleNamespace(id=uuid.uuid4(), owner_id=owner_id)

    monkeypatch.setattr(
        sessions.session_repo,
        "get_session_for_user",
        lambda *_: session,
    )
    monkeypatch.setattr(sessions.doc_repo, "get_document", lambda *_: document)
    attach = MagicMock()
    monkeypatch.setattr(sessions.sd_repo, "attach_document", attach)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sessions.attach_document(
                session.id,
                SimpleNamespace(document_id=document.id),
                db=MagicMock(),
                user_id=str(other_user_id),
            )
        )

    assert exc_info.value.status_code == 403
    attach.assert_not_called()


def test_document_chat_ignores_another_users_attached_upload(monkeypatch):
    owner_id = uuid.uuid4()
    other_user_id = uuid.uuid4()
    document = SimpleNamespace(
        id=uuid.uuid4(),
        owner_id=owner_id,
        source_type="uploaded",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = document
    db_context = MagicMock()
    db_context.__enter__.return_value = db

    monkeypatch.setattr(document_chat, "DatabaseSession", lambda: db_context)
    embed = MagicMock()
    monkeypatch.setattr(document_chat, "embed", embed)

    result = document_chat.document_chat_node({
        "question": "What does this document say?",
        "user_id": str(other_user_id),
        "document_ids": [str(document.id)],
        "retrieval_attempts": 0,
    })

    assert result["retrieved_chunks"] == []
    assert result["citations"] == []
    assert "could not find relevant content" in result["answer"]
    embed.assert_not_called()
