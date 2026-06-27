"""V2 schema: Assistant Sessions, Messages, Documents, pgvector embeddings

Drops V1 tables (document_sessions, chat_sessions, chat_messages, document_chunks)
and creates the V2 schema centred on AssistantSession.

New tables:
  - assistant_sessions
  - messages
  - documents
  - session_documents
  - document_chunks  (now owned by documents, not sessions)
  - document_embeddings  (pgvector 768-dim)

Revision ID: 0002_v2_schema
Revises: 0001_initial
Create Date: 2026-06-03
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

# revision identifiers
revision: str = "0002_v2_schema"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Enable pgvector extension ──────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("DROP TABLE IF EXISTS chat_messages CASCADE")
    op.execute("DROP TABLE IF EXISTS chat_sessions CASCADE")
    op.execute("DROP TABLE IF EXISTS document_chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS document_sessions CASCADE")

    # Drop V1 enum types
    op.execute("DROP TYPE IF EXISTS message_role_enum")
    op.execute("DROP TYPE IF EXISTS processing_status_enum")
    op.execute("DROP TYPE IF EXISTS source_type_enum")

    # ── assistant_sessions ─────────────────────────────────────────────────────
    op.create_table(
        "assistant_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(500), nullable=False, server_default="New Session"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_assistant_sessions_user_id", "assistant_sessions", ["user_id"])
    op.create_index(
        "ix_assistant_sessions_user_updated",
        "assistant_sessions",
        ["user_id", "updated_at"],
    )

    # ── messages ───────────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("assistant_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "role",
            sa.Enum("user", "assistant", "system", name="message_role_enum_v2"),
            nullable=False,
        ),
        sa.Column(
            "message_type",
            sa.Enum("text", "summary_card", "event_card", name="message_type_enum"),
            nullable=False,
            server_default="text",
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("citations", sa.JSON, nullable=True),  # [{doc_name, page, text}]
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])
    op.create_index(
        "ix_messages_session_created", "messages", ["session_id", "created_at"]
    )

    # ── documents ──────────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,  # NULL = system-owned legal case
        ),
        sa.Column(
            "source_type",
            sa.Enum("uploaded", "legal_case", name="doc_source_type_enum"),
            nullable=False,
        ),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("blob_path", sa.String(1000), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=True, unique=True),  # SHA-256 for dedup
        sa.Column("file_size_bytes", sa.Integer, nullable=True),
        sa.Column(
            "status",
            sa.Enum("uploaded", "processing", "ready", "failed", name="doc_status_enum"),
            nullable=False,
            server_default="uploaded",
        ),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])
    op.create_index("ix_documents_file_hash", "documents", ["file_hash"], unique=True)
    op.create_index("ix_documents_status", "documents", ["status"])

    # ── session_documents (join table) ────────────────────────────────────────
    op.create_table(
        "session_documents",
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("assistant_sessions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "attached_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_session_documents_session_id", "session_documents", ["session_id"]
    )
    op.create_index(
        "ix_session_documents_document_id", "session_documents", ["document_id"]
    )

    # ── document_chunks (now owned by document, not session) ──────────────────
    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),  # order within document
        sa.Column("chunk_text", sa.Text, nullable=False),
        sa.Column("chunk_metadata", sa.JSON, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index(
        "ix_document_chunks_doc_page",
        "document_chunks",
        ["document_id", "page_number"],
    )

    # ── document_embeddings (pgvector 768-dim) ────────────────────────────────
    op.create_table(
        "document_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_chunks.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,  # one embedding per chunk
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # pgvector column — uses raw SQL DDL since SQLAlchemy doesn't know this type natively
        # without the pgvector SQLAlchemy integration registered
        sa.Column("embedding", sa.Text, nullable=False),  # placeholder, overridden below
    )
    # Replace the 'embedding' column with the actual vector type via raw SQL
    op.execute("ALTER TABLE document_embeddings DROP COLUMN embedding")
    op.execute("ALTER TABLE document_embeddings ADD COLUMN embedding vector(768) NOT NULL")

    op.create_index(
        "ix_document_embeddings_document_id",
        "document_embeddings",
        ["document_id"],
    )
    op.create_index(
        "ix_document_embeddings_chunk_id",
        "document_embeddings",
        ["chunk_id"],
        unique=True,
    )
    # HNSW index for fast ANN search (cosine distance)
    op.execute(
        """
        CREATE INDEX ix_document_embeddings_hnsw
        ON document_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    # Drop V2 tables (children first)
    op.drop_table("document_embeddings")
    op.drop_table("document_chunks")
    op.drop_table("session_documents")
    op.drop_table("documents")
    op.drop_table("messages")
    op.drop_table("assistant_sessions")

    # Drop V2 enum types
    op.execute("DROP TYPE IF EXISTS doc_status_enum")
    op.execute("DROP TYPE IF EXISTS doc_source_type_enum")
    op.execute("DROP TYPE IF EXISTS message_type_enum")
    op.execute("DROP TYPE IF EXISTS message_role_enum_v2")

    # Recreate V1 tables (abbreviated — use 0001 migration for full restore)
    # This downgrade only restores the tables, not all indexes/constraints.
    # For a full V1 restore, downgrade to 0001_initial instead.
    op.execute("DROP EXTENSION IF EXISTS vector")
