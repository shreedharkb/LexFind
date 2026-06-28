"""
Documents API Router — LexFind.

Handles PDF uploads, database saving, status polling, and triggering Celery workers.
"""
import logging
import io
import uuid
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.db.session import get_db
from app.db.crud import document_repository as doc_repo
from app.services.blob_storage_service import blob_storage_service
from app.workers.document_worker import process_document_task
from app.schemas.documents import DocumentStatusResponse, DocumentUploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    
    file_hash = doc_repo.sha256_of_bytes(content)
    existing_doc = doc_repo.get_document_by_hash(db, file_hash, uuid.UUID(user_id))
    
    if existing_doc:
        logger.info("Duplicate document detected: %s", file_hash)
        return {
            "id": existing_doc.id,
            "title": existing_doc.title,
            "status": existing_doc.status,
            "message": "Document already exists"
        }

    blob_path = blob_storage_service.upload_pdf(content, file.filename)
    
    doc = doc_repo.create_document(
        db=db,
        title=file.filename,
        blob_path=blob_path,
        source_type="uploaded",
        owner_id=uuid.UUID(user_id),
        file_hash=file_hash,
        file_size_bytes=len(content)
    )
    
    process_document_task.delay(document_id=str(doc.id), blob_path=blob_path)
    
    return {
        "id": doc.id,
        "title": doc.title,
        "status": doc.status,
        "message": "Document uploaded and processing started"
    }


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(
    document_id: UUID,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    doc = doc_repo.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if doc.owner_id is not None and doc.owner_id != uuid.UUID(user_id):
        raise HTTPException(status_code=403, detail="Not authorized to access this document")
        
    return doc


@router.get("/{document_id}/pdf", summary="Serve a document PDF inline")
def serve_document_pdf(
    document_id: UUID,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    doc = doc_repo.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if doc.owner_id is not None and doc.owner_id != uuid.UUID(user_id):
        raise HTTPException(status_code=403, detail="Not authorized to access this document.")

    try:
        pdf_bytes = blob_storage_service.download_pdf(doc.blob_path)
    except Exception as exc:
        logger.error("Failed to serve PDF for document %s: %s", document_id, exc)
        raise HTTPException(status_code=404, detail="PDF file not found.")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )
