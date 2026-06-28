"""Pydantic schemas for messages."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CitationModel(BaseModel):
    document_id: str
    document_title: str
    document_filename: Optional[str] = None
    page_number: Optional[int] = None
    excerpt: str
    court: Optional[str] = None
    year: Optional[int] = None
    citation: Optional[str] = None
    score: Optional[float] = None
    chunk_id: Optional[str] = None


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    id: UUID
    role: str
    message_type: str
    content: str
    citations: Optional[List[CitationModel]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    items: List[MessageResponse]
    total: int
