"""
Authentication service — password hashing (bcrypt) and JWT management.

Usage:
    from app.services.auth_service import auth_service
    user = auth_service.register_user(db, email, password)
    token = auth_service.create_access_token(str(user.id))
    uid = auth_service.verify_token(token)


"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.db.models import User

# ── JWT config ──────────────────────────────────────────────────────────────
SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "changeme-in-production")
ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))


class AuthenticationError(Exception):
    """Raised on invalid credentials or token failures."""


class DuplicateEmailError(Exception):
    """Raised when an email is already registered."""


class AuthenticationService:
    """Encapsulates user registration, login, and JWT token lifecycle."""

    # ── Password helpers ────────────────────────────────────────────────
    @staticmethod
    def hash_password(password: str) -> str:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False

    # ── User management ─────────────────────────────────────────────────
    def register_user(self, db: Session, email: str, password: str) -> User:
        """Create a new user. Raises DuplicateEmailError if email exists."""
        if db.query(User).filter(User.email == email).first():
            raise DuplicateEmailError(f"Email '{email}' is already registered.")
        user = User(
            id=uuid.uuid4(),
            email=email,
            hashed_password=self.hash_password(password),
            role="user",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def authenticate_user(self, db: Session, email: str, password: str) -> User:
        """Verify credentials. Raises AuthenticationError on mismatch."""
        user = db.query(User).filter(User.email == email).first()
        if not user or not self.verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid email or password.")
        return user

    # ── JWT helpers ──────────────────────────────────────────────────────
    @staticmethod
    def create_access_token(user_id: str) -> str:
        """Issue a signed JWT with a 24-hour expiry."""
        now = datetime.now(timezone.utc)
        payload = {"sub": user_id, "iat": now, "exp": now + timedelta(hours=EXPIRY_HOURS)}
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> str:
        """Decode a JWT and return the user_id. Raises AuthenticationError on failure."""
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            uid: Optional[str] = data.get("sub")
            if uid is None:
                raise AuthenticationError("Token missing subject claim.")
            return uid
        except JWTError as exc:
            raise AuthenticationError(f"Token validation failed: {exc}") from exc


# Module-level singleton
auth_service = AuthenticationService()
