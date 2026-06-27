"""Initial migration for LexFind V2 schema.

Revision ID: 0001_initial
Revises: (none)
Create Date: 2026-06-26


"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '0001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Enable pgvector extension ──────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # ── users ──────────────────────────────────────────────────────────────────
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False, server_default='user'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True)
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # ── assistant_sessions ──────────────────────────────────────────────────────
    op.create_table(
        'assistant_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False, server_default='New Session'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    op.create_index(op.f('ix_assistant_sessions_user_id'), 'assistant_sessions', ['user_id'], unique=False)
    op.create_index('ix_assistant_sessions_user_updated', 'assistant_sessions', ['user_id', 'updated_at'], unique=False)

    # ── messages ────────────────────────────────────────────────────────────────
    op.create_table(
        'messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('assistant_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.Enum('user', 'assistant', 'system', name='message_role_enum_v2'), nullable=False),
        sa.Column('message_type', sa.Enum('text', 'summary_card', 'event_card', 'legal_notice_card', name='message_type_enum'), nullable=False, server_default='text'),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('citations', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    op.create_index(op.f('ix_messages_session_id'), 'messages', ['session_id'], unique=False)
    op.create_index('ix_messages_session_created', 'messages', ['session_id', 'created_at'], unique=False)

    # ── documents ───────────────────────────────────────────────────────────────
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('source_type', sa.Enum('uploaded', 'legal_case', name='doc_source_type_enum'), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('blob_path', sa.String(length=1000), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=True),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('status', sa.Enum('uploaded', 'processing', 'ready', 'failed', name='doc_status_enum'), nullable=False, server_default='uploaded'),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    op.create_index(op.f('ix_documents_owner_id'), 'documents', ['owner_id'], unique=False)
    op.create_index('ix_documents_status', 'documents', ['status'], unique=False)
    op.create_index(op.f('ix_documents_file_hash'), 'documents', ['file_hash'], unique=True)

    # ── session_documents ───────────────────────────────────────────────────────
    op.create_table(
        'session_documents',
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('assistant_sessions.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('attached_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # ── document_chunks ─────────────────────────────────────────────────────────
    op.create_table(
        'document_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('chunk_text', sa.Text(), nullable=False),
        sa.Column('chunk_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    op.create_index(op.f('ix_document_chunks_document_id'), 'document_chunks', ['document_id'], unique=False)
    op.create_index('ix_document_chunks_doc_page', 'document_chunks', ['document_id', 'page_number'], unique=False)

    # ── document_embeddings ─────────────────────────────────────────────────────
    op.create_table(
        'document_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('chunk_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('document_chunks.id', ondelete='CASCADE'), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('embedding', Vector(dim=768), nullable=False)
    )
    op.create_index(op.f('ix_document_embeddings_chunk_id'), 'document_embeddings', ['chunk_id'], unique=True)
    op.create_index(op.f('ix_document_embeddings_document_id'), 'document_embeddings', ['document_id'], unique=False)


def downgrade() -> None:
    op.drop_table('document_embeddings')
    op.drop_table('document_chunks')
    op.drop_table('session_documents')
    op.drop_table('documents')
    op.drop_table('messages')
    op.drop_table('assistant_sessions')
    op.drop_table('users')

    op.execute("DROP TYPE IF EXISTS message_role_enum_v2")
    op.execute("DROP TYPE IF EXISTS message_type_enum")
    op.execute("DROP TYPE IF EXISTS doc_source_type_enum")
    op.execute("DROP TYPE IF EXISTS doc_status_enum")
