"""
LexFind — Qdrant Search Service
==================================
Core search layer using Qdrant (Hybrid Search with Sparse/Dense Vectors).

Responsibilities:
  - Embed queries using sentence-transformers/all-mpnet-base-v2 (loaded once)
  - Query Qdrant collection "legal_corpus" with optional payload filters
  - Group chunk-level Qdrant results by document_id
  - Hydrate results with full metadata from PostgreSQL (legal_documents + legal_chunks)
  - Provide case-name search, similar-case discovery, and RAG context fetch

This service touches ONLY the legal_documents and legal_chunks tables.
It does NOT touch users, assistant_sessions, messages, documents, document_chunks,
or document_embeddings.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchText,
    MatchValue,
    Prefetch,
    Range,
    ScoredPoint,
    SparseVector,
)
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.search_schemas import (
    AskResponse,
    CaseDetailResponse,
    CaseResult,
    ChunkResult,
    SearchResponse,
    SimilarCasesResponse,
)

logger = logging.getLogger(__name__)

# ── Config pulled from env (set in .env or docker-compose) ─────────────────────
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "legal_corpus")
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

# Fetch more raw chunks than top_k documents so we have enough to group
_CHUNK_FETCH_MULTIPLIER: int = 10


# ── Module-level singletons (loaded once at first use) ─────────────────────────

_embedding_model: Optional[SentenceTransformer] = None
_bm25_model: Optional[SparseTextEmbedding] = None
_qdrant_client: Optional[QdrantClient] = None


def _get_embedding_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer, loading it on first call."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading dense embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Dense embedding model loaded (dim=%d)", _embedding_model.get_sentence_embedding_dimension())
    return _embedding_model


def _get_bm25_model() -> SparseTextEmbedding:
    """Return the cached FastEmbed BM25 model, loading it on first call."""
    global _bm25_model
    if _bm25_model is None:
        logger.info("Loading BM25 sparse model (fastembed) ...")
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("BM25 model loaded.")
    return _bm25_model


def _get_qdrant_client() -> QdrantClient:
    """Return the cached QdrantClient, creating it on first call."""
    global _qdrant_client
    if _qdrant_client is None:
        logger.info("Connecting to Qdrant at %s:%d", QDRANT_HOST, QDRANT_PORT)
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    return _qdrant_client


# ── Internal helpers ───────────────────────────────────────────────────────────

def _embed(text_input: str) -> List[float]:
    """Embed a single string into a dense vector (unit-normalised float list)."""
    model = _get_embedding_model()
    vec = model.encode([text_input], normalize_embeddings=True)
    return vec[0].tolist()


def _embed_sparse(text_input: str) -> SparseVector:
    """Embed a single string into a BM25 sparse vector for Qdrant."""
    model = _get_bm25_model()
    result = list(model.query_embed(text_input))
    sparse = result[0]
    return SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )


def _build_filter(
    court: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    state: Optional[str] = None,
    case_type: Optional[str] = None,
    section_type: Optional[str] = None,
) -> Optional[Filter]:
    """Build a Qdrant Filter from optional field constraints."""
    conditions = []

    if court:
        conditions.append(FieldCondition(key="court", match=MatchValue(value=court)))
    if state:
        conditions.append(FieldCondition(key="state", match=MatchValue(value=state)))
    if case_type:
        conditions.append(FieldCondition(key="case_type", match=MatchValue(value=case_type)))
    if section_type:
        conditions.append(FieldCondition(key="section_type", match=MatchValue(value=section_type)))

    if year_min is not None or year_max is not None:
        conditions.append(
            FieldCondition(
                key="year",
                range=Range(
                    gte=year_min,
                    lte=year_max,
                ),
            )
        )

    if not conditions:
        return None
    return Filter(must=conditions)


def _fetch_metadata_batch(db: Session, document_ids: List[str]) -> Dict[str, dict]:
    """
    Fetch legal_documents metadata for a list of document UUIDs in one query.
    Returns a dict keyed by document_id (str).
    """
    if not document_ids:
        return {}

    rows = db.execute(
        text("""
            SELECT
                id::text,
                title,
                petitioner,
                respondent,
                court,
                state,
                year,
                citation,
                judges,
                case_type,
                page_count,
                chunk_strategy
            FROM legal_documents
            WHERE id = ANY(CAST(:ids AS uuid[]))
        """),
        {"ids": document_ids},
    ).fetchall()

    return {
        row[0]: {
            "title": row[1],
            "petitioner": row[2],
            "respondent": row[3],
            "court": row[4],
            "state": row[5],
            "year": row[6],
            "citation": row[7],
            "judges": row[8] or [],
            "case_type": row[9],
            "page_count": row[10],
            "chunk_strategy": row[11],
        }
        for row in rows
    }


def _fetch_chunk_texts_batch(db: Session, chunk_ids: List[str]) -> Dict[str, str]:
    """Fetch chunk texts for a list of chunk UUIDs in one query."""
    if not chunk_ids:
        return {}
    rows = db.execute(
        text("SELECT id::text, chunk_text FROM legal_chunks WHERE id = ANY(CAST(:ids AS uuid[]))"),
        {"ids": chunk_ids},
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _qdrant_hits_to_case_results(
    hits: List[ScoredPoint],
    db: Session,
    exclude_document_id: Optional[str] = None,
    top_k: int = 10,
) -> List[CaseResult]:
    """
    Group chunk-level Qdrant hits by document_id, hydrate with metadata,
    and return the top_k unique cases sorted by best chunk score.
    """
    doc_chunks: Dict[str, List[ScoredPoint]] = defaultdict(list)
    for hit in hits:
        doc_id = (hit.payload or {}).get("document_id")
        if doc_id and (not exclude_document_id or doc_id != exclude_document_id):
            doc_chunks[doc_id].append(hit)

    sorted_docs = sorted(
        doc_chunks.items(),
        key=lambda kv: max(h.score for h in kv[1]),
        reverse=True,
    )[:top_k]
    if not sorted_docs:
        return []

    doc_ids = [did for did, _ in sorted_docs]
    chunk_ids = [
        hit.payload.get("chunk_id")
        for _, hlist in sorted_docs
        for hit in hlist
        if hit.payload.get("chunk_id")
    ]

    meta_by_doc = _fetch_metadata_batch(db, doc_ids)
    text_by_chunk = _fetch_chunk_texts_batch(db, chunk_ids)

    results = []
    for doc_id, hlist in sorted_docs:
        meta = meta_by_doc.get(doc_id, {})
        chunk_results = [
            ChunkResult(
                chunk_id=(cid := (h.payload or {}).get("chunk_id", str(h.id))),
                chunk_text=text_by_chunk.get(cid, (h.payload or {}).get("chunk_text", "")),
                chunk_index=(h.payload or {}).get("chunk_index", 0),
                section_type=(h.payload or {}).get("section_type", "unknown"),
                score=round(h.score, 6),
            )
            for h in sorted(hlist, key=lambda x: x.score, reverse=True)
        ]
        results.append(
            CaseResult(
                document_id=doc_id,
                title=meta.get("title"),
                petitioner=meta.get("petitioner"),
                respondent=meta.get("respondent"),
                court=meta.get("court"),
                year=meta.get("year"),
                citation=meta.get("citation"),
                judges=meta.get("judges") or [],
                case_type=meta.get("case_type"),
                state=meta.get("state"),
                score=round(max(h.score for h in hlist), 6),
                top_chunk=chunk_results[0],
                all_chunks=chunk_results,
            )
        )
    return results


# ── Service class ──────────────────────────────────────────────────────────────


class QdrantSearchService:
    """Qdrant-backed search service with hybrid dense+sparse RRF search."""

    def __init__(self) -> None:
        try:
            _get_embedding_model()
            _get_bm25_model()
            _get_qdrant_client()
        except Exception as exc:
            logger.warning("Could not pre-warm search service singletons: %s", exc)

    # ── POST /api/search ───────────────────────────────────────────────────

    def search(
        self,
        db: Session,
        query: str,
        *,
        court: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        state: Optional[str] = None,
        case_type: Optional[str] = None,
        section_type: Optional[str] = None,
        top_k: int = 10,
        search_mode: str = "hybrid",
    ) -> SearchResponse:
        t0 = time.perf_counter()
        if not (query := query.strip()):
            return SearchResponse(query=query, total_results=0, results=[], search_time_ms=0.0)

        qdrant_filter = _build_filter(court, year_min, year_max, state, case_type, section_type)
        client = _get_qdrant_client()
        limit = top_k * _CHUNK_FETCH_MULTIPLIER

        try:
            dense_vec = _embed(query)
            sparse_vec = _embed_sparse(query)

            raw = client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", filter=qdrant_filter, limit=limit),
                    Prefetch(query=sparse_vec, using="sparse", filter=qdrant_filter, limit=limit),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.exception("Qdrant hybrid search failed: %s", exc)
            return SearchResponse(
                query=query, total_results=0, results=[],
                search_time_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        results = _qdrant_hits_to_case_results(raw, db, top_k=top_k)
        return SearchResponse(
            query=query,
            total_results=len(results),
            results=results,
            search_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ── POST /api/search/by-name ───────────────────────────────────────────

    def search_by_case_name(self, db: Session, case_name: str, *, top_k: int = 10) -> SearchResponse:
        t0 = time.perf_counter()
        if not (case_name := case_name.strip()):
            return SearchResponse(query=case_name, total_results=0, results=[], search_time_ms=0.0)

        try:
            dense_vec = _embed(case_name)
            sparse_vec = _embed_sparse(case_name)
            name_filter = Filter(must=[FieldCondition(key="title", match=MatchText(text=case_name))])
            limit = top_k * _CHUNK_FETCH_MULTIPLIER

            raw = _get_qdrant_client().query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", filter=name_filter, limit=limit),
                    Prefetch(query=sparse_vec, using="sparse", filter=name_filter, limit=limit),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.exception("Qdrant search_by_case_name failed: %s", exc)
            raw = []

        results = _qdrant_hits_to_case_results(raw, db, top_k=top_k)
        return SearchResponse(
            query=case_name,
            total_results=len(results),
            results=results,
            search_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ── GET /api/search/similar/{document_id} ──────────────────────────────

    def get_similar_cases(self, db: Session, document_id: str, *, top_k: int = 10) -> SimilarCasesResponse:
        client = _get_qdrant_client()
        doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        all_vectors, next_offset = [], None

        try:
            while True:
                result, next_offset = client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter=doc_filter,
                    limit=200,
                    offset=next_offset,
                    with_vectors=["dense"],
                    with_payload=False,
                )
                for p in result:
                    vec = p.vector
                    # When requesting named vectors, p.vector is a dict
                    if isinstance(vec, dict):
                        vec = vec.get("dense")
                    if vec is not None:
                        all_vectors.append(vec)
                if next_offset is None:
                    break
        except Exception as exc:
            logger.warning("Qdrant scroll similar cases failed: %s", exc)

        if not all_vectors:
            return SimilarCasesResponse(source_document_id=document_id, total_results=0, results=[])

        centroid = np.mean(all_vectors, axis=0)
        if (norm := np.linalg.norm(centroid)) > 0:
            centroid /= norm

        try:
            raw = client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    Prefetch(query=centroid.tolist(), using="dense", limit=(top_k + 1) * _CHUNK_FETCH_MULTIPLIER),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=(top_k + 1) * _CHUNK_FETCH_MULTIPLIER,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.exception("Qdrant query similar cases failed: %s", exc)
            raw = []

        results = _qdrant_hits_to_case_results(raw, db, exclude_document_id=document_id, top_k=top_k)
        return SimilarCasesResponse(source_document_id=document_id, total_results=len(results), results=results)

    # ── GET /api/search/case/{document_id} ─────────────────────────────────

    def get_case_detail(self, db: Session, document_id: str) -> CaseDetailResponse:
        doc = db.execute(
            text(
                "SELECT id::text, title, petitioner, respondent, court, state, year, citation, judges, case_type, page_count, chunk_strategy "
                "FROM legal_documents WHERE id = CAST(:doc_id AS uuid)"
            ),
            {"doc_id": document_id},
        ).fetchone()
        if not doc:
            return None

        chunk_rows = db.execute(
            text(
                "SELECT id::text, chunk_text, chunk_index, section_type "
                "FROM legal_chunks WHERE document_id = CAST(:doc_id AS uuid) ORDER BY chunk_index"
            ),
            {"doc_id": document_id},
        ).fetchall()
        chunks = [
            ChunkResult(
                chunk_id=r[0], chunk_text=r[1], chunk_index=r[2],
                section_type=r[3] or "unknown", score=1.0,
            )
            for r in chunk_rows
        ]
        return CaseDetailResponse(
            document_id=doc[0], title=doc[1], petitioner=doc[2], respondent=doc[3],
            court=doc[4], state=doc[5], year=doc[6], citation=doc[7],
            judges=doc[8] or [], case_type=doc[9], page_count=doc[10],
            chunk_strategy=doc[11], chunks=chunks,
        )

    # ── POST /api/search/ask ───────────────────────────────────────────────

    def ask(self, db: Session, document_id: str, question: str, *, top_k: int = 5) -> AskResponse:
        question = question.strip()
        doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])

        try:
            dense_vec = _embed(question)
            sparse_vec = _embed_sparse(question)

            raw = _get_qdrant_client().query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", filter=doc_filter, limit=top_k * 3),
                    Prefetch(query=sparse_vec, using="sparse", filter=doc_filter, limit=top_k * 3),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.warning("Qdrant ask failed: %s", exc)
            raw = []

        chunk_ids = [(h.payload or {}).get("chunk_id", str(h.id)) for h in raw]
        text_by_chunk = _fetch_chunk_texts_batch(db, chunk_ids)

        context_chunks = [
            ChunkResult(
                chunk_id=cid,
                chunk_text=text_by_chunk.get(cid, (h.payload or {}).get("chunk_text", "")),
                chunk_index=(h.payload or {}).get("chunk_index", 0),
                section_type=(h.payload or {}).get("section_type", "unknown"),
                score=round(h.score, 6),
            )
            for cid, h in zip(chunk_ids, raw)
        ]
        return AskResponse(
            document_id=document_id,
            question=question,
            context_chunks=context_chunks,
            total_chunks=len(context_chunks),
        )


# ── Singleton accessor ─────────────────────────────────────────────────────────

_service_instance: Optional[QdrantSearchService] = None


def get_search_service() -> QdrantSearchService:
    global _service_instance
    if _service_instance is None:
        _service_instance = QdrantSearchService()
    return _service_instance
