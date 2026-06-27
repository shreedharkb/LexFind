"""
Database package for LexFind.

Re-exports models, engine, session helpers, and CRUD repositories
so consumers can do `from app.db import User, get_db, ...`.


"""

from .models import (
    Base, User, AssistantSession, Message,
    Document, SessionDocument, DocumentChunk, DocumentEmbedding,
)
from .config import engine, SessionLocal, get_engine, get_session_factory
from .session import get_db, DatabaseSession, create_tables, drop_tables

__all__ = [
    # ORM models
    "Base", "User", "AssistantSession", "Message",
    "Document", "SessionDocument", "DocumentChunk", "DocumentEmbedding",
    # Engine & sessions
    "engine", "SessionLocal", "get_engine", "get_session_factory",
    "get_db", "DatabaseSession", "create_tables", "drop_tables",
]
