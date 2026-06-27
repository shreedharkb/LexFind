"""Add section_type and chunk_strategy columns to legal_chunks and legal_documents.

Adds the new legal-aware chunking metadata columns to the existing tables
created in migration 0003. Uses ALTER TABLE — existing rows are preserved
and back-filled with the default values ('unknown' / 'sentence_aware').

Revision ID: 0004_add_chunking_strategy_columns
Revises: 0003_add_legal_corpus_tables
Create Date: 2026-06-10
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_add_chunk_cols"
down_revision: Union[str, None] = "0003_add_legal_corpus_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── legal_chunks: add section_type and chunk_strategy ─────────────────────
    # Existing rows (from naive word-chunker) are back-filled with:
    #   section_type  = 'unknown'         (they weren't split by section)
    #   chunk_strategy = 'sentence_aware' (closest analogue to naive splitting)
    op.add_column(
        "legal_chunks",
        sa.Column(
            "section_type",
            sa.Text(),
            server_default="unknown",
            nullable=False,
        ),
    )
    op.add_column(
        "legal_chunks",
        sa.Column(
            "chunk_strategy",
            sa.Text(),
            server_default="sentence_aware",
            nullable=False,
        ),
    )
    op.create_index(
        "idx_legal_chunks_section_type",
        "legal_chunks",
        ["section_type"],
    )

    # ── legal_documents: add chunk_strategy summary column ────────────────────
    op.add_column(
        "legal_documents",
        sa.Column("chunk_strategy", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("legal_documents", "chunk_strategy")
    op.drop_index("idx_legal_chunks_section_type", table_name="legal_chunks")
    op.drop_column("legal_chunks", "chunk_strategy")
    op.drop_column("legal_chunks", "section_type")
