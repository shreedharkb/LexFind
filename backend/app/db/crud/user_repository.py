"""User CRUD operations."""

from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import User


class UserRepository:
    """Query helpers for the User model."""

    @staticmethod
    def get_by_id(db: Session, user_id: str) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def create(db: Session, user: User) -> User:
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def exists_by_email(db: Session, email: str) -> bool:
        return db.query(User.id).filter(User.email == email).first() is not None

    @staticmethod
    def delete(db: Session, user: User) -> None:
        db.delete(user)
        db.commit()
