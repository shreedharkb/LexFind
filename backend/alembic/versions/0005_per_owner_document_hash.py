"""Scope uploaded-document hash deduplication to each owner.

Revision ID: 0005_per_owner_document_hash
Revises: 0004_add_chunk_cols
Create Date: 2026-10-12
"""

from typing import Sequence, Union

from alembic import op


revision: str = "0005_per_owner_document_hash"
down_revision: Union[str, None] = "0004_add_chunk_cols"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_documents_file_hash", table_name="documents")
    op.create_unique_constraint(
        "uq_documents_owner_file_hash",
        "documents",
        ["owner_id", "file_hash"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_documents_owner_file_hash",
        "documents",
        type_="unique",
    )
    op.create_index("ix_documents_file_hash", "documents", ["file_hash"], unique=True)
