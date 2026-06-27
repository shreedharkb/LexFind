"""
Alembic migration environment for LexFind.

Loads DATABASE_URL from .env and registers all SQLAlchemy models
so that `alembic revision --autogenerate` can detect schema diffs.


"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# Put backend/ on sys.path so `from app.db.models import ...` works
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_DIR)

# Load .env from backend/.env
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

# Import Base to register all models for autogenerate
from app.db.models import Base  # noqa: F401

# ── Alembic config ──────────────────────────────────────────────────────────
config = context.config

# Use DATABASE_URL from env; fall back to default for local dev
db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/lexfind")
config.set_main_option("sqlalchemy.url", db_url)

# Wire up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ── Offline mode (SQL script generation) ────────────────────────────────────
def run_migrations_offline() -> None:
    """Generate SQL without a live database connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode (live database connection) ──────────────────────────────────
def run_migrations_online() -> None:
    """Connect to the database and run migrations inside a transaction."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
