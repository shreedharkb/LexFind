"""Search response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    court: Optional[str] = None
    year_min: Optional[int] = Field(None, ge=1950, le=2030)
    year_max: Optional[int] = Field(None, ge=1950, le=2030)
    state: Optional[str] = None
    case_type: Optional[str] = None
    section_type: Optional[str] = None
    search_mode: str = Field(default="hybrid")
    top_k: int = Field(10, ge=1, le=50)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_id: str
    top_k: int = Field(5, ge=1, le=20)


class ChunkResult(BaseModel):
    chunk_id: str
    chunk_text: str
    chunk_index: int
    section_type: str
    score: float


class CaseResult(BaseModel):
    document_id: str
    title: Optional[str] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    court: Optional[str] = None
    year: Optional[int] = None
    citation: Optional[str] = None
    judges: List[str] = Field(default_factory=list)
    case_type: Optional[str] = None
    state: Optional[str] = None
    score: float
    top_chunk: ChunkResult
    all_chunks: List[ChunkResult] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[CaseResult]
    search_time_ms: float


class CaseDetailResponse(BaseModel):
    document_id: str
    title: Optional[str] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    court: Optional[str] = None
    state: Optional[str] = None
    year: Optional[int] = None
    citation: Optional[str] = None
    judges: List[str] = Field(default_factory=list)
    case_type: Optional[str] = None
    page_count: Optional[int] = None
    chunk_strategy: Optional[str] = None
    chunks: List[ChunkResult] = Field(default_factory=list)


class SimilarCasesResponse(BaseModel):
    source_document_id: str
    total_results: int
    results: List[CaseResult]


class AskResponse(BaseModel):
    document_id: str
    question: str
    context_chunks: List[ChunkResult]
    total_chunks: int
