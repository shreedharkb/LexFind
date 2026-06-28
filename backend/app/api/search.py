"""
LexFind — Corpus Search Router
==================================
FastAPI routes for the Qdrant-powered legal corpus search layer.

Endpoints:
  POST /api/search                      – full semantic search with filters
  POST /api/search/by-name              – search by party / case name
  GET  /api/search/case/{document_id}   – full case metadata + all chunks
  GET  /api/search/similar/{document_id}– cases similar to a given document
  POST /api/search/ask                  – RAG context fetch for a document

All routes require authentication (Bearer JWT).
All routes use dependency injection for DB session and search service.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
import os

from app.api.dependencies.auth import get_current_user
from app.db.session import get_db
from app.schemas.search_schemas import (
    AskRequest,
    AskResponse,
    CaseDetailResponse,
    SearchRequest,
    SearchResponse,
    SimilarCasesResponse,
)
from app.services.qdrant_search_service import QdrantSearchService, get_search_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/search",
    tags=["Corpus Search"],
)


# ── POST /api/search ───────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=SearchResponse,
    summary="Semantic search over 1.1 M legal chunks",
    status_code=status.HTTP_200_OK,
)
async def search(
    request: SearchRequest,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
    service: QdrantSearchService = Depends(get_search_service),
):
    """
    Semantic search over the full LexFind legal corpus (46k cases, 1.1M chunks).

    Qdrant returns chunk-level hits that are **grouped by document** before being
    returned — you receive `top_k` unique *cases*, not raw chunks.

    Optional filters:
    - **court**: e.g. `"Supreme Court"`, `"High Court of Bombay"`
    - **year_min / year_max**: judgment year range
    - **state**: e.g. `"Maharashtra"`, `"Delhi"`
    - **case_type**: e.g. `"Writ Petition"`, `"Criminal Appeal"`, `"SLP"`
    - **section_type**: restrict to a specific legal section — `"held"`, `"facts"`,
      `"judgment"`, `"headnote"`, `"issues"`, `"order"`, etc.
    """
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be empty.",
        )

    top_k = max(1, min(request.top_k, 50))

    try:
        return service.search(
            db,
            query_text,
            court=request.court,
            year_min=request.year_min,
            year_max=request.year_max,
            state=request.state,
            case_type=request.case_type,
            section_type=request.section_type,
            top_k=top_k,
            search_mode=request.search_mode,
        )
    except Exception as exc:
        logger.exception("Search failed for query=%r: %s", query_text, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        )


# ── POST /api/search/by-name ───────────────────────────────────────────────────

@router.post(
    "/by-name",
    response_model=SearchResponse,
    summary="Search by party or case name",
    status_code=status.HTTP_200_OK,
)
async def search_by_name(
    request: SearchRequest,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
    service: QdrantSearchService = Depends(get_search_service),
):
    """
    Search for cases by petitioner/respondent name or partial case title.

    Uses Qdrant's `MatchText` payload filter combined with semantic similarity,
    so partial names and abbreviations work well (e.g. *"Puttaswamy"*).
    """
    case_name = request.query.strip()
    if not case_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="case_name cannot be empty.",
        )

    top_k = max(1, min(request.top_k, 50))

    try:
        return service.search_by_case_name(db, case_name, top_k=top_k)
    except Exception as exc:
        logger.exception("Name search failed for name=%r: %s", case_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Case name search failed: {exc}",
        )


# ── GET /api/search/case/{document_id} ────────────────────────────────────────

@router.get(
    "/case/{document_id}",
    response_model=CaseDetailResponse,
    summary="Full case metadata and all chunks",
    status_code=status.HTTP_200_OK,
)
async def get_case(
    document_id: str,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
    service: QdrantSearchService = Depends(get_search_service),
):
    """
    Retrieve complete metadata and every text chunk for a single case.

    `document_id` is the UUID from the `legal_documents` table.
    This is a direct PostgreSQL lookup — no Qdrant call is made.
    """
    try:
        detail = service.get_case_detail(db, document_id)
    except Exception as exc:
        logger.exception("get_case_detail failed for doc=%s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve case: {exc}",
        )

    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No case found with document_id={document_id}",
        )

    return detail


# ── GET /api/search/similar/{document_id} ─────────────────────────────────────

@router.get(
    "/similar/{document_id}",
    response_model=SimilarCasesResponse,
    summary="Find cases similar to a given document",
    status_code=status.HTTP_200_OK,
)
async def get_similar(
    document_id: str,
    top_k: int = Query(10, ge=1, le=50, description="Number of similar cases to return"),
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
    service: QdrantSearchService = Depends(get_search_service),
):
    """
    Return the `top_k` cases most similar to the given document.

    Similarity is computed by averaging all chunk embeddings of the source
    document into a single centroid vector, then searching Qdrant for the
    nearest neighbours (excluding the source document itself).
    """
    try:
        return service.get_similar_cases(db, document_id, top_k=top_k)
    except Exception as exc:
        logger.exception("get_similar_cases failed for doc=%s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar-case search failed: {exc}",
        )


# ── POST /api/search/ask ───────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=AskResponse,
    summary="RAG context fetch — relevant chunks for a document-scoped question",
    status_code=status.HTTP_200_OK,
)
async def ask(
    request: AskRequest,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
    service: QdrantSearchService = Depends(get_search_service),
):
    """
    Fetch the most relevant text chunks from a *specific* document for a question.

    This is the **RAG retrieval step** — it returns the context chunks that
    should be passed to an LLM (e.g. via the legal agent) to generate an answer.
    The LLM call itself is handled by the agents layer, not here.

    `document_id` must be a UUID from the `legal_documents` table.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    try:
        return service.ask(
            db,
            request.document_id,
            question,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.exception(
            "ask() failed for doc=%s question=%r: %s",
            request.document_id, question, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context retrieval failed: {exc}",
        )


# ── GET /api/search/pdf/{document_id} ─────────────────────────────────────────

@router.get(
    "/pdf/{document_id}",
    summary="Serve a legal case PDF by document UUID",
    # Intentionally no auth dependency so iframes/blob URLs can fetch it directly
)
async def serve_pdf(
    document_id: str,
    db: Session = Depends(get_db),
):
    """
    Serve a PDF file from the data/pdfs directory using its document_id (UUID).
    """
    row = db.execute(
        text("SELECT filename FROM legal_documents WHERE id = CAST(:doc_id AS uuid)"),
        {"doc_id": document_id}
    ).fetchone()

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="PDF file not found in database.")

    filename = row[0]
    safe_filename = os.path.basename(filename)
    local_pdf_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "pdfs",
    )
    pdf_path = os.path.join(local_pdf_dir, safe_filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        content_disposition_type="inline",
    )
