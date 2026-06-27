"""
SQLAlchemy session management.

Provides:
  - get_db()          — FastAPI dependency (yield-based)
  - DatabaseSession   — Context manager for background tasks / scripts
  - create_tables()   — Dev-only table creation
  - drop_tables()     — Dev-only table teardown


"""

from typing import Generator
from sqlalchemy.orm import Session

from .config import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a DB session per request.

    Usage:
        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseSession:
    """
    Context manager for DB access outside of FastAPI routes
    (e.g. Celery tasks, CLI scripts, tests).

    Usage:
        with DatabaseSession() as db:
            user = db.query(User).first()
    """

    def __init__(self):
        self.db: Session = None

    def __enter__(self) -> Session:
        self.db = SessionLocal()
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.db.rollback()
        else:
            self.db.commit()
        self.db.close()


def create_tables():
    """Create all tables (dev/testing only — use Alembic in production)."""
    from .models import Base
    from .config import engine
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all tables (dev/testing only — DESTRUCTIVE)."""
    from .models import Base
    from .config import engine
    Base.metadata.drop_all(bind=engine)
