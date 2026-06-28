"""Pydantic schemas for chat sessions."""

from datetime import datetime
from typing import List
from uuid import UUID

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str = Field(default="New Session", max_length=500)


class SessionRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)


class SessionListItem(BaseModel):
    id: UUID
    title: str
    updated_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentReference(BaseModel):
    id: UUID
    title: str
    status: str
    source_type: str

    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    id: UUID
    title: str
    documents: List[DocumentReference] = []
    updated_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True
