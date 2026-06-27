"""Document CRUD operations."""

import hashlib
import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import Document


def create_document(
    db: Session,
    title: str,
    blob_path: str,
    source_type: str,
    owner_id: Optional[uuid.UUID] = None,
    file_hash: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    doc_id: Optional[uuid.UUID] = None,
) -> Document:
    doc = Document(
        id=doc_id or uuid.uuid4(),
        owner_id=owner_id,
        source_type=source_type,
        title=title,
        blob_path=blob_path,
        file_hash=file_hash,
        file_size_bytes=file_size_bytes,
        status="uploaded",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document(db: Session, document_id: uuid.UUID) -> Optional[Document]:
    return db.query(Document).filter(Document.id == document_id).first()


def get_document_by_hash(db: Session, file_hash: str, owner_id: uuid.UUID) -> Optional[Document]:
    """Deduplication: find existing doc with same SHA-256 hash for this owner."""
    return db.query(Document).filter(
        Document.file_hash == file_hash, Document.owner_id == owner_id,
    ).first()


def update_status(
    db: Session,
    document: Document,
    status: str,
    summary: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Document:
    document.status = status
    if summary is not None:
        document.summary = summary
    if error_message is not None:
        document.error_message = error_message
    db.commit()
    db.refresh(document)
    return document


def list_user_documents(db: Session, owner_id: uuid.UUID, limit: int = 50) -> List[Document]:
    return (
        db.query(Document)
        .filter(Document.owner_id == owner_id)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .all()
    )


def delete_document(db: Session, document: Document) -> None:
    db.delete(document)
    db.commit()


def sha256_of_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest for upload deduplication."""
    return hashlib.sha256(data).hexdigest()
