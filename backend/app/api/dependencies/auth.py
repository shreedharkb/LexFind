"""
JWT authentication dependency for FastAPI route handlers.


"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services.auth_service import auth_service, AuthenticationError

_bearer = HTTPBearer(
    scheme_name="Bearer",
    description="JWT from POST /api/auth/login",
    auto_error=True,
)

_bearer_optional = HTTPBearer(
    scheme_name="Bearer",
    description="JWT from POST /api/auth/login",
    auto_error=False,
)


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """Extract and validate the JWT; return user_id string."""
    try:
        return auth_service.verify_token(creds.credentials)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_optional),
) -> Optional[str]:
    """Return user_id if a valid token is present, else None."""
    if not creds:
        return None
    try:
        return auth_service.verify_token(creds.credentials)
    except AuthenticationError:
        return None
