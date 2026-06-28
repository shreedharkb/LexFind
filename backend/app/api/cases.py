"""
Case search & retrieval routes — v1

POST /api/v1/cases/search      – semantic search over indexed legal cases
GET  /api/v1/cases/{case_id}   – get case metadata + PDF access URL
GET  /api/v1/health            – system health check
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.search_service import get_searcher
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.dependencies.auth import get_current_user
from app.db.models import User, Document
from app.db.crud import document_repository as doc_repo
from app.db.crud import session_repository as session_repo
from app.db.crud import session_document_repository as sd_repo
from app.workers.document_worker import process_document_task

logger = logging.getLogger(__name__)

router = APIRouter(tags=["v1 · Cases"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class CaseSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


class CaseSearchResult(BaseModel):
    case_id: str          # filename used as stable case identifier
    title: str
    score: float
    similarity_percentage: float


class CaseSearchResponse(BaseModel):
    success: bool
    query: str
    results: list[CaseSearchResult]
    total_results: int


class CaseDetailResponse(BaseModel):
    case_id: str
    title: str
    pdf_url: str


class HealthResponse(BaseModel):
    status: str
    message: str
    total_cases: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filename_to_title(filename: str) -> str:
    """Convert raw filename like 'abc__court__2019.pdf' into a readable title."""
    if not filename:
        return "Legal Case Document"
    return (
        filename
        .replace(".pdf", "")
        .replace("__", " — ")
        .replace("_", " ")
        .strip()
    )



# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check():
    """
    Check the health of the legal case search service.

    Returns searcher status and total number of indexed cases.
    """
    try:
        searcher = get_searcher()
        total = searcher.index.ntotal if searcher.index else 0
        return HealthResponse(
            status="healthy",
            message="Legal case search service is running.",
            total_cases=total,
        )
    except Exception as exc:
        logger.warning("Health check degraded: %s", exc)
        return HealthResponse(
            status="degraded",
            message=str(exc),
            total_cases=0,
        )


@router.post(
    "/cases/search",
    response_model=CaseSearchResponse,
    summary="Semantic search over legal cases",
)
async def search_cases(request: CaseSearchRequest):
    """
    Perform semantic search over 46,000+ indexed legal cases.

    - **query**: Natural language query or case description
    - **top_k**: Maximum number of results (1–50, default 10)
    """
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    top_k = max(1, min(request.top_k or 10, 50))

    try:
        searcher = get_searcher()
        raw_results = searcher.search(query_text, top_k=top_k)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    results = [
        CaseSearchResult(
            case_id=r["filename"],
            title=_filename_to_title(r["filename"]),
            score=r["score"],
            similarity_percentage=r.get("similarity_percentage", round(r["score"] * 100, 1)),
        )
        for r in raw_results
    ]

    return CaseSearchResponse(
        success=True,
        query=query_text,
        results=results,
        total_results=len(results),
    )


@router.get(
    "/cases/search",
    response_model=CaseSearchResponse,
    summary="Semantic search over legal cases (GET)",
)
async def search_cases_get(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
):
    """
    GET variant of case search — same semantics as POST but uses query params.
    Useful for shareable links and browser-level caching.
    """
    query_text = q.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        searcher = get_searcher()
        raw_results = searcher.search(query_text, top_k=top_k)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    results = [
        CaseSearchResult(
            case_id=r["filename"],
            title=_filename_to_title(r["filename"]),
            score=r["score"],
            similarity_percentage=r.get("similarity_percentage", round(r["score"] * 100, 1)),
        )
        for r in raw_results
    ]

    return CaseSearchResponse(
        success=True,
        query=query_text,
        results=results,
        total_results=len(results),
    )


@router.get(
    "/cases/{case_id:path}",
    response_model=CaseDetailResponse,
    summary="Get case metadata and PDF access URL",
)
async def get_case(case_id: str):
    """
    Retrieve metadata and the PDF access URL for a specific legal case.

    - **case_id**: The stable case identifier (typically the PDF filename)
    """
    base_url = os.getenv("VITE_API_BASE_URL", "")
    pdf_url = f"{base_url}/api/pdf/{case_id}"

    return CaseDetailResponse(
        case_id=case_id,
        title=_filename_to_title(case_id),
        pdf_url=pdf_url,
    )


@router.post(
    "/cases/{case_id:path}/analyze",
    summary="Create an Assistant Session from a Legal Case",
    status_code=201
)
async def analyze_case(
    case_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """
    Take a case from the global search corpus, ensure it has a Document record,
    create a new AssistantSession, attach the Document, and return the Session ID.

    case_id can be either:
      - A UUID (from Qdrant/new search) → we look up the real filename from legal_documents
      - A filename (legacy FAISS flow)   → used directly as the blob filename
    """
    import uuid as _uuid
    from sqlalchemy import text

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ── Resolve filename: UUID lookup first, fall back to treating case_id as filename ──
    actual_filename = None
    try:
        # If case_id is a valid UUID, look up the real PDF filename from legal_documents
        _uuid.UUID(case_id)  # raises ValueError if not a UUID
        row = db.execute(
            text("SELECT filename, title FROM legal_documents WHERE id = CAST(:cid AS uuid)"),
            {"cid": case_id}
        ).fetchone()
        if row:
            actual_filename = row[0]
            corpus_title = row[1] or _filename_to_title(actual_filename)
        else:
            raise HTTPException(status_code=404, detail=f"No case found with id={case_id}")
    except ValueError:
        # case_id is not a UUID — treat it as a raw filename (legacy FAISS path)
        actual_filename = case_id
        corpus_title = _filename_to_title(case_id)

    abs_pdf_path = os.path.join(base_dir, "data", "pdfs", actual_filename)

    # Verify the PDF actually exists before creating a document record
    if not os.path.exists(abs_pdf_path):
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found on disk: {actual_filename}"
        )

    # ── Ensure Document record exists ──────────────────────────────────────────
    doc = db.query(Document).filter(
        Document.source_type == "legal_case",
        Document.blob_path == abs_pdf_path
    ).first()

    if not doc:
        doc_uuid = None
        try:
            doc_uuid = _uuid.UUID(case_id)
        except ValueError:
            pass

        doc = doc_repo.create_document(
            db=db,
            title=corpus_title,
            blob_path=abs_pdf_path,
            source_type="legal_case",
            owner_id=None,
            doc_id=doc_uuid
        )
        
        # Corpus cases are already indexed in Qdrant! 
        # Mark as ready immediately and DO NOT trigger Celery.
        doc.status = "ready"
        db.commit()
    elif doc.status != "ready":
        doc.status = "ready"
        db.commit()

    # ── Create Session + attach Document ───────────────────────────────────────
    session = session_repo.create_session(db, _uuid.UUID(user_id), title=f"Analysis: {corpus_title}")
    sd_repo.attach_document(db, session.id, doc.id)

    return {"session_id": session.id, "document_id": doc.id}


@router.get(
    "/pdf/{filename}",
    summary="Serve a legal case PDF",
)
async def serve_pdf(filename: str):
    """
    Serve a PDF file from the data/pdfs directory.
    """
    safe_filename = os.path.basename(filename)
    local_pdf_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "pdfs",
    )
    pdf_path = os.path.join(local_pdf_dir, safe_filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found.")
    
    # content_disposition_type="inline" tells the browser to render in-viewer.
    # Omitting `filename` prevents Chrome from treating it as an attachment download.
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        content_disposition_type="inline",
    )

