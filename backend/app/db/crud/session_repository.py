"""AssistantSession CRUD operations."""

import uuid
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.db.models import AssistantSession


def create_session(db: Session, user_id: uuid.UUID, title: str = "New Session") -> AssistantSession:
    s = AssistantSession(id=uuid.uuid4(), user_id=user_id, title=title)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def get_session(db: Session, session_id: uuid.UUID) -> Optional[AssistantSession]:
    return db.query(AssistantSession).filter(AssistantSession.id == session_id).first()


def get_session_for_user(
    db: Session, session_id: uuid.UUID, user_id: uuid.UUID,
) -> Optional[AssistantSession]:
    return (
        db.query(AssistantSession)
        .filter(AssistantSession.id == session_id, AssistantSession.user_id == user_id)
        .first()
    )


def list_sessions(db: Session, user_id: uuid.UUID, limit: int = 50) -> List[AssistantSession]:
    return (
        db.query(AssistantSession)
        .filter(AssistantSession.user_id == user_id)
        .order_by(AssistantSession.updated_at.desc())
        .limit(limit)
        .all()
    )


def rename_session(db: Session, session: AssistantSession, title: str) -> AssistantSession:
    session.title = title
    db.commit()
    db.refresh(session)
    return session


def touch_session(db: Session, session: AssistantSession) -> None:
    """Bump updated_at after a new message is added."""
    session.updated_at = func.now()
    db.commit()


def delete_session(db: Session, session: AssistantSession) -> None:
    db.delete(session)
    db.commit()
