"""Pydantic schemas for authentication endpoints."""

from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    """POST /api/auth/register body."""
    email: EmailStr = Field(..., description="Valid email address", examples=["user@example.com"])
    password: str = Field(..., min_length=8, description="Min 8 characters", examples=["Secret123!"])


class UserLoginRequest(BaseModel):
    """POST /api/auth/login body."""
    email: EmailStr = Field(..., description="Registered email", examples=["user@example.com"])
    password: str = Field(..., description="Account password", examples=["Secret123!"])


class TokenResponse(BaseModel):
    """JWT token returned after successful login."""
    access_token: str = Field(..., description="Signed JWT")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(default=86400, description="TTL in seconds (24 h)")
