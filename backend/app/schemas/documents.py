"""Pydantic schemas for document endpoints."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    id: UUID
    title: str
    status: str
    message: str


class DocumentStatusResponse(BaseModel):
    id: UUID
    title: str
    status: str
    summary: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AttachDocumentRequest(BaseModel):
    document_id: UUID
