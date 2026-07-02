"""V2 schema: Assistant Sessions, Messages, Documents, pgvector embeddings

All tables were consolidated into 0001_initial.
This migration is intentionally a no-op to avoid duplicate table creation.

Revision ID: 0002_v2_schema
Revises: 0001_initial
Create Date: 2026-06-03
"""
from typing import Sequence, Union
from alembic import op

# revision identifiers
revision: str = "0002_v2_schema"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # All tables were consolidated into 0001_initial.
    # This migration is intentionally a no-op to avoid duplicate table creation.
    pass


def downgrade() -> None:
    pass
