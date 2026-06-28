"""
SQLAlchemy ORM models for LexFind.

Relationship graph:

  User (1) ──── (N) AssistantSession (1) ──── (N) Message
                          │
                          └──── (M:N via SessionDocument) ──── Document
                                                                  │
                                                              (N) DocumentChunk
                                                                  │
                                                              (1) DocumentEmbedding
"""

import uuid
from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


# ── User ────────────────────────────────────────────────────────────────────

class User(Base):
    """Authenticated platform user."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="user", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    sessions = relationship("AssistantSession", back_populates="user", cascade="all, delete-orphan")
    documents = relationship(
        "Document", back_populates="owner", cascade="all, delete-orphan",
        foreign_keys="Document.owner_id",
    )

    def __repr__(self):
        return f"<User id={self.id} email={self.email}>"


# ── AssistantSession ────────────────────────────────────────────────────────

class AssistantSession(Base):
    """
    A ChatGPT-style conversation. Contains Messages and has Documents
    attached via the SessionDocument join table.
    """

    __tablename__ = "assistant_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    title = Column(String(500), nullable=False, default="New Session")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )

    user = relationship("User", back_populates="sessions")
    messages = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    session_documents = relationship(
        "SessionDocument", back_populates="session", cascade="all, delete-orphan",
    )

    @property
    def documents(self):
        """Convenience accessor — returns attached Document objects."""
        return [sd.document for sd in self.session_documents]

    __table_args__ = (
        Index("ix_sessions_user_updated", "user_id", "updated_at"),
    )

    def __repr__(self):
        return f"<AssistantSession id={self.id} title={self.title!r}>"


# ── Message ─────────────────────────────────────────────────────────────────

class Message(Base):
    """
    Single turn in a conversation.

    role:         'user' | 'assistant' | 'system'
    message_type: 'text' | 'summary_card' | 'event_card' | 'legal_notice_card'
    citations:    JSON array of {doc_name, page_number, excerpt} for RAG answers
    """

    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("assistant_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    role = Column(
        Enum("user", "assistant", "system", name="message_role_enum_v2"),
        nullable=False,
    )
    message_type = Column(
        Enum("text", "summary_card", "event_card", "legal_notice_card", name="message_type_enum"),
        nullable=False, default="text",
    )
    content = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session = relationship("AssistantSession", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
    )

    def __repr__(self):
        return f"<Message id={self.id} role={self.role} type={self.message_type}>"


# ── Document ────────────────────────────────────────────────────────────────

class Document(Base):
    """
    A reusable PDF resource.

    - owner_id=NULL, source_type='legal_case'  → pre-indexed corpus case
    - owner_id=UUID, source_type='uploaded'    → user-uploaded PDF

    file_hash (SHA-256) is used for upload deduplication.
    """

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    source_type = Column(
        Enum("uploaded", "legal_case", name="doc_source_type_enum"), nullable=False,
    )
    title = Column(String(500), nullable=False)
    blob_path = Column(String(1000), nullable=False)
    file_hash = Column(String(64), nullable=True, unique=True)
    file_size_bytes = Column(Integer, nullable=True)
    status = Column(
        Enum("uploaded", "processing", "ready", "failed", name="doc_status_enum"),
        nullable=False, default="uploaded",
    )
    summary = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )

    owner = relationship("User", back_populates="documents", foreign_keys=[owner_id])
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    session_documents = relationship(
        "SessionDocument", back_populates="document", cascade="all, delete-orphan",
    )

    __table_args__ = (Index("ix_documents_status", "status"),)

    def __repr__(self):
        return f"<Document id={self.id} title={self.title!r} status={self.status}>"


# ── SessionDocument (M:N join) ──────────────────────────────────────────────

class SessionDocument(Base):
    """
    Join table: session ←→ document (many-to-many).
    Deleting a session cascades to these links but NOT to the Document itself.
    """

    __tablename__ = "session_documents"

    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("assistant_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    attached_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session = relationship("AssistantSession", back_populates="session_documents")
    document = relationship("Document", back_populates="session_documents")

    def __repr__(self):
        return f"<SessionDocument session={self.session_id} doc={self.document_id}>"


# ── DocumentChunk ───────────────────────────────────────────────────────────

class DocumentChunk(Base):
    """
    Sequential text segment from a Document page.
    chunk_index is zero-based ordering within the document.
    """

    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    page_number = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document = relationship("Document", back_populates="chunks")
    embedding = relationship(
        "DocumentEmbedding", back_populates="chunk",
        uselist=False, cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_chunks_doc_page", "document_id", "page_number"),
    )

    def __repr__(self):
        return f"<DocumentChunk id={self.id} doc={self.document_id} page={self.page_number}>"


# ── DocumentEmbedding ───────────────────────────────────────────────────────

class DocumentEmbedding(Base):
    """
    768-dim pgvector embedding for a DocumentChunk (1:1).

    document_id is denormalised for efficient filtered vector search:
        WHERE document_id = :id ORDER BY embedding <=> :vec
    """

    __tablename__ = "document_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    embedding = Column(Vector(768), nullable=False)

    chunk = relationship("DocumentChunk", back_populates="embedding")

    def __repr__(self):
        return f"<DocumentEmbedding id={self.id} chunk={self.chunk_id}>"
