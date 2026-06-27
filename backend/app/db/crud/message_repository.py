"""Message CRUD operations."""

import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import Message


def append_message(
    db: Session,
    session_id: uuid.UUID,
    role: str,
    content: str,
    message_type: str = "text",
    citations: Optional[list] = None,
) -> Message:
    msg = Message(
        id=uuid.uuid4(),
        session_id=session_id,
        role=role,
        message_type=message_type,
        content=content,
        citations=citations,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def list_messages(
    db: Session, session_id: uuid.UUID, limit: int = 200, offset: int = 0,
) -> List[Message]:
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_recent_messages(db: Session, session_id: uuid.UUID, n: int = 6) -> List[Message]:
    """Last n messages in chronological order (for LLM conversation history)."""
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(n)
        .all()[::-1]
    )
