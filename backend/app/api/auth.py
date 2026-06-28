"""
Authentication routes.

POST /api/auth/register  → create account
POST /api/auth/login     → get JWT


"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.auth import UserRegisterRequest, UserLoginRequest, TokenResponse
from app.services.auth_service import auth_service, DuplicateEmailError, AuthenticationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(body: UserRegisterRequest, db: Session = Depends(get_db)):
    """Create a new account. Returns user_id on success."""
    try:
        user = auth_service.register_user(db, body.email, body.password)
        logger.info("User registered: %s", body.email)
        return {"user_id": str(user.id), "email": user.email, "message": "Account created."}
    except DuplicateEmailError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("Registration error: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Registration failed.")


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get JWT",
)
async def login(body: UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticate with email + password. Returns a 24-hour JWT."""
    try:
        user = auth_service.authenticate_user(db, body.email, body.password)
        token = auth_service.create_access_token(str(user.id))
        logger.info("User logged in: %s", body.email)
        return TokenResponse(access_token=token)
    except AuthenticationError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        logger.exception("Login error: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed.")
