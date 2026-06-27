"""
Database connection and engine configuration.

Sets up a connection-pooled SQLAlchemy engine pointed at the PostgreSQL
instance defined by DATABASE_URL.


"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/lexfind",
)

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,       # detect stale connections before use
    pool_recycle=3600,        # recycle after 1 hour
    echo=False,               # set True for SQL debug logging
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_engine():
    """Return the module-level SQLAlchemy engine."""
    return engine


def get_session_factory():
    """Return the session factory bound to the engine."""
    return SessionLocal
