"""Add legal corpus tables (legal_documents + legal_chunks) for Qdrant ingestion pipeline.

These tables are ISOLATED from the existing user/session schema.
They store metadata and text chunks for the 46k JUDIS corpus.
Vectors live in Qdrant; this table is the authoritative metadata store.

Revision ID: 0003_add_legal_corpus_tables
Revises: 0002_v2_schema
Create Date: 2026-06-09
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

# revision identifiers
revision: str = "0003_add_legal_corpus_tables"
down_revision: Union[str, None] = "0002_v2_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── legal_documents ───────────────────────────────────────────────────────
    op.create_table(
        "legal_documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("petitioner", sa.Text(), nullable=True),
        sa.Column("respondent", sa.Text(), nullable=True),
        sa.Column("court", sa.Text(), nullable=True),
        sa.Column("state", sa.Text(), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("date_of_judgment", sa.Date(), nullable=True),
        sa.Column("judges", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("citation", sa.Text(), nullable=True),
        sa.Column("acts_referred", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("bench_strength", sa.Integer(), nullable=True),
        sa.Column("case_type", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("full_text", sa.Text(), nullable=True),
        sa.Column("language", sa.Text(), server_default="en", nullable=True),
        sa.Column("quality_flag", sa.Text(), server_default="clean", nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=True,
        ),
        sa.Column("qdrant_synced", sa.Boolean(), server_default="false", nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("filename", name="uq_legal_documents_filename"),
    )

    # Indexes on legal_documents
    op.create_index("idx_legal_documents_court", "legal_documents", ["court"])
    op.create_index("idx_legal_documents_year", "legal_documents", ["year"])
    op.create_index("idx_legal_documents_quality", "legal_documents", ["quality_flag"])
    op.create_index("idx_legal_documents_qdrant_synced", "legal_documents", ["qdrant_synced"])
    op.create_index("idx_legal_documents_filename", "legal_documents", ["filename"])

    # ── legal_chunks ──────────────────────────────────────────────────────────
    op.create_table(
        "legal_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("qdrant_point_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["legal_documents.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("qdrant_point_id", name="uq_legal_chunks_qdrant_point_id"),
    )

    # Indexes on legal_chunks
    op.create_index("idx_legal_chunks_document_id", "legal_chunks", ["document_id"])
    op.create_index("idx_legal_chunks_qdrant_point_id", "legal_chunks", ["qdrant_point_id"])


def downgrade() -> None:
    # Drop in reverse order (children first)
    op.drop_index("idx_legal_chunks_qdrant_point_id", table_name="legal_chunks")
    op.drop_index("idx_legal_chunks_document_id", table_name="legal_chunks")
    op.drop_table("legal_chunks")

    op.drop_index("idx_legal_documents_filename", table_name="legal_documents")
    op.drop_index("idx_legal_documents_qdrant_synced", table_name="legal_documents")
    op.drop_index("idx_legal_documents_quality", table_name="legal_documents")
    op.drop_index("idx_legal_documents_year", table_name="legal_documents")
    op.drop_index("idx_legal_documents_court", table_name="legal_documents")
    op.drop_table("legal_documents")
