"""
LexFind — Qdrant Search Service
==================================
Core search layer using Qdrant hybrid RRF (Dense + BM25 sparse).

Responsibilities:
  - Embed queries using sentence-transformers/all-mpnet-base-v2 (loaded once)
  - Query Qdrant collection "legal_corpus" with optional payload filters
  - Group chunk-level Qdrant results by document_id
  - Hydrate results with full metadata from PostgreSQL (legal_documents + legal_chunks)
  - Provide case-name search, similar-case discovery, and RAG context fetch

This service touches ONLY the legal_documents and legal_chunks tables.
It does NOT touch users, assistant_sessions, messages, documents, document_chunks,
or document_embeddings.

NOTE: The collection MUST have named "dense" and "sparse" vector slots (hybrid format).
      This is how the Qdrant ingestor set it up. The old auto-detect code was the bug —
      it was silently falling back to unnamed-vector mode which returned 0 results.
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

# ── Config pulled from env ─────────────────────────────────────────────────────
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
    """
    Fetch chunk_text for a list of chunk UUIDs from legal_chunks in one query.
    Returns a dict keyed by chunk_id (str).
    """
    if not chunk_ids:
        return {}

    rows = db.execute(
        text("""
            SELECT id::text, chunk_text
            FROM legal_chunks
            WHERE id = ANY(CAST(:ids AS uuid[]))
        """),
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
    Group raw Qdrant chunk hits by document_id, hydrate with PG metadata,
    and return up to top_k CaseResult objects.

    Grouping logic:
      - document score = max(chunk scores)
      - top_chunk = highest-scoring chunk
      - all_chunks = all chunks from that document in the hits list
    """
    # 1. Group hits by document_id
    doc_chunks: Dict[str, List[ScoredPoint]] = defaultdict(list)
    for hit in hits:
        payload = hit.payload or {}
        doc_id = payload.get("document_id")
        if not doc_id:
            continue
        if exclude_document_id and doc_id == exclude_document_id:
            continue
        doc_chunks[doc_id].append(hit)

    # 2. Sort documents by their best chunk score, take top_k
    sorted_docs = sorted(
        doc_chunks.items(),
        key=lambda kv: max(h.score for h in kv[1]),
        reverse=True,
    )[:top_k]

    if not sorted_docs:
        return []

    doc_ids = [doc_id for doc_id, _ in sorted_docs]
    chunk_ids_needed = [
        hit.payload.get("chunk_id")
        for _, hits_list in sorted_docs
        for hit in hits_list
        if hit.payload.get("chunk_id")
    ]

    # 3. Batch-fetch PG metadata
    meta_by_doc = _fetch_metadata_batch(db, doc_ids)
    text_by_chunk = _fetch_chunk_texts_batch(db, chunk_ids_needed)

    # 4. Assemble CaseResult objects
    results: List[CaseResult] = []
    for doc_id, hits_list in sorted_docs:
        meta = meta_by_doc.get(doc_id, {})

        chunk_results: List[ChunkResult] = []
        for hit in sorted(hits_list, key=lambda h: h.score, reverse=True):
            p = hit.payload or {}
            c_id = p.get("chunk_id", str(hit.id))
            chunk_results.append(
                ChunkResult(
                    chunk_id=c_id,
                    chunk_text=text_by_chunk.get(c_id, p.get("chunk_text", "")),
                    chunk_index=p.get("chunk_index", 0),
                    section_type=p.get("section_type", "unknown"),
                    score=round(hit.score, 6),
                )
            )

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
                score=round(max(h.score for h in hits_list), 6),
                top_chunk=chunk_results[0],
                all_chunks=chunk_results,
            )
        )

    return results


# ── Public service class ──────────────────────────────────────────────────────

class QdrantSearchService:
    """
    Stateless search service. Each method accepts a SQLAlchemy Session so
    it can be used as a FastAPI dependency without holding a long-lived session.
    """

    def __init__(self) -> None:
        # Pre-warm the dense embedding model only — ~500MB, eliminates first-request latency.
        # BM25 is lazy-loaded on first keyword search (~1s load, ~50MB) to avoid RAM
        # pressure with the Celery document-processing worker which also loads this model.
        try:
            _get_embedding_model()
            _get_qdrant_client()
        except Exception as exc:
            logger.warning("Could not pre-warm search service singletons: %s", exc)

    # ── search ────────────────────────────────────────────────────────────────

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
        """
        Corpus search with two strategies controlled by `search_mode`:

        - **hybrid** (default): Dense (semantic) + BM25 (keyword) combined via
          Reciprocal Rank Fusion. Best for concept / topic queries.
        - **keyword**: BM25 sparse-only. Best for exact case names, citations,
          act sections (e.g. "Puttaswamy vs Union of India", "AIR 1973 SC 461").

        Steps (both modes):
          1. Build optional Qdrant payload filter.
          2. Embed query into dense and/or sparse vectors.
          3. Fetch top_k × MULTIPLIER chunk hits from Qdrant.
          4. Group by document, keeping top top_k unique cases.
          5. Hydrate with PostgreSQL metadata.
        """
        t0 = time.perf_counter()

        query = query.strip()
        if not query:
            return SearchResponse(query=query, total_results=0, results=[], search_time_ms=0.0)

        qdrant_filter = _build_filter(
            court=court,
            year_min=year_min,
            year_max=year_max,
            state=state,
            case_type=case_type,
            section_type=section_type,
        )

        client = _get_qdrant_client()
        limit = top_k * _CHUNK_FETCH_MULTIPLIER

        # Embed once for both paths
        dense_vec = _embed(query)

        try:
            if search_mode == "keyword":
                # ── BM25 sparse-only path ─────────────────────────────────────
                sparse_vec = _embed_sparse(query)
                raw = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=sparse_vec,
                    using="sparse",
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                ).points
            else:
                # ── Hybrid RRF path (default) ─────────────────────────────────
                sparse_vec = _embed_sparse(query)
                raw = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    prefetch=[
                        Prefetch(
                            query=dense_vec,
                            using="dense",
                            filter=qdrant_filter,
                            limit=limit,
                        ),
                        Prefetch(
                            query=sparse_vec,
                            using="sparse",
                            filter=qdrant_filter,
                            limit=limit,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=limit,
                    with_payload=True,
                ).points
        except Exception as exc:
            logger.error("Qdrant search failed (mode=%s): %s", search_mode, exc)
            raise

        results = _qdrant_hits_to_case_results(raw, db, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return SearchResponse(
            query=query,
            total_results=len(results),
            results=results,
            search_time_ms=round(elapsed_ms, 2),
        )

    # ── search_by_case_name ──────────────────────────────────────────────────

    def search_by_case_name(
        self,
        db: Session,
        case_name: str,
        *,
        top_k: int = 10,
    ) -> SearchResponse:
        """
        Payload-based title search using Qdrant's MatchText filter.
        Does a semantic search *and* pre-filters to documents whose title
        contains the case_name substring.
        """
        t0 = time.perf_counter()

        case_name = case_name.strip()
        if not case_name:
            return SearchResponse(query=case_name, total_results=0, results=[], search_time_ms=0.0)

        # Semantic vector for the name (works even for partial names)
        vec = _embed(case_name)

        # Filter: title payload must contain the query text
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="title",
                    match=MatchText(text=case_name),
                )
            ]
        )

        client = _get_qdrant_client()
        try:
            raw = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vec,
                using="dense",
                query_filter=qdrant_filter,
                limit=top_k * _CHUNK_FETCH_MULTIPLIER,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.error("Qdrant case-name search failed: %s", exc)
            raise

        results = _qdrant_hits_to_case_results(raw, db, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return SearchResponse(
            query=case_name,
            total_results=len(results),
            results=results,
            search_time_ms=round(elapsed_ms, 2),
        )

    # ── get_similar_cases ────────────────────────────────────────────────────

    def get_similar_cases(
        self,
        db: Session,
        document_id: str,
        *,
        top_k: int = 10,
    ) -> SimilarCasesResponse:
        """
        Find cases similar to a given document.

        Strategy:
          1. Retrieve *all* Qdrant points whose payload.document_id == document_id.
          2. Average their vectors (centroid).
          3. Search Qdrant for the nearest neighbours to that centroid.
          4. Exclude the source document from results.
        """
        client = _get_qdrant_client()

        # Step 1: Scroll all chunk point vectors for this document
        doc_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

        all_vectors: List[List[float]] = []
        next_offset = None
        page_size = 200

        while True:
            try:
                result, next_offset = client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter=doc_filter,
                    limit=page_size,
                    offset=next_offset,
                    with_vectors=["dense"],
                    with_payload=False,
                )
            except Exception as exc:
                logger.error("Qdrant scroll for document %s failed: %s", document_id, exc)
                raise

            for point in result:
                vec = point.vector
                if isinstance(vec, dict):
                    vec = vec.get("dense")
                if vec is not None:
                    all_vectors.append(vec)

            if next_offset is None:
                break

        if not all_vectors:
            logger.warning("No vectors found in Qdrant for document_id=%s", document_id)
            return SimilarCasesResponse(
                source_document_id=document_id,
                total_results=0,
                results=[],
            )

        # Step 2: Compute centroid
        centroid = np.mean(all_vectors, axis=0)
        # Re-normalise for cosine similarity
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Step 3: Search Qdrant for similar cases
        try:
            raw = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=centroid.tolist(),
                using="dense",
                limit=(top_k + 1) * _CHUNK_FETCH_MULTIPLIER,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.error("Qdrant similar-cases search failed: %s", exc)
            raise

        # Step 4: Group, exclude source, hydrate
        results = _qdrant_hits_to_case_results(
            raw, db, exclude_document_id=document_id, top_k=top_k
        )

        return SimilarCasesResponse(
            source_document_id=document_id,
            total_results=len(results),
            results=results,
        )

    # ── get_case_detail ──────────────────────────────────────────────────────

    def get_case_detail(self, db: Session, document_id: str) -> CaseDetailResponse:
        """
        Return full metadata + all chunks for a specific document from PostgreSQL.
        No Qdrant call needed — this is a direct DB lookup.
        """
        # Fetch document metadata
        doc_row = db.execute(
            text("""
                SELECT
                    id::text, title, petitioner, respondent, court, state,
                    year, citation, judges, case_type, page_count, chunk_strategy
                FROM legal_documents
                WHERE id = CAST(:doc_id AS uuid)
            """),
            {"doc_id": document_id},
        ).fetchone()

        if not doc_row:
            return None  # Caller raises 404

        # Fetch all chunks ordered by index
        chunk_rows = db.execute(
            text("""
                SELECT id::text, chunk_text, chunk_index, section_type
                FROM legal_chunks
                WHERE document_id = CAST(:doc_id AS uuid)
                ORDER BY chunk_index
            """),
            {"doc_id": document_id},
        ).fetchall()

        chunks = [
            ChunkResult(
                chunk_id=row[0],
                chunk_text=row[1],
                chunk_index=row[2],
                section_type=row[3] or "unknown",
                score=1.0,  # Not a search result — full doc fetch
            )
            for row in chunk_rows
        ]

        return CaseDetailResponse(
            document_id=doc_row[0],
            title=doc_row[1],
            petitioner=doc_row[2],
            respondent=doc_row[3],
            court=doc_row[4],
            state=doc_row[5],
            year=doc_row[6],
            citation=doc_row[7],
            judges=doc_row[8] or [],
            case_type=doc_row[9],
            page_count=doc_row[10],
            chunk_strategy=doc_row[11],
            chunks=chunks,
        )

    # ── ask (RAG context fetch) ───────────────────────────────────────────────

    def ask(
        self,
        db: Session,
        document_id: str,
        question: str,
        *,
        top_k: int = 5,
    ) -> AskResponse:
        """
        Fetch the most relevant chunks from a *specific* document for a question.
        This is the RAG context-retrieval step — it does NOT call the LLM.

        Uses Hybrid RRF (Dense + BM25) scoped to the document_id payload filter,
        so the LLM receives the best possible context — covering both semantic
        relevance and exact legal term matches.

        Workflow:
          1. Embed the question (dense + sparse).
          2. Prefetch from both vector slots filtered by document_id.
          3. Fuse with RRF and return top_k chunks as LLM context.
        """
        question = question.strip()

        doc_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

        dense_vec = _embed(question)
        sparse_vec = _embed_sparse(question)

        client = _get_qdrant_client()
        try:
            raw = client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    Prefetch(
                        query=dense_vec,
                        using="dense",
                        filter=doc_filter,
                        limit=top_k * 3,
                    ),
                    Prefetch(
                        query=sparse_vec,
                        using="sparse",
                        filter=doc_filter,
                        limit=top_k * 3,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.error("Qdrant ask() hybrid search failed: %s", exc)
            raise

        chunk_ids = [
            (hit.payload or {}).get("chunk_id", str(hit.id))
            for hit in raw
        ]
        text_by_chunk = _fetch_chunk_texts_batch(db, chunk_ids)

        context_chunks = [
            ChunkResult(
                chunk_id=cid,
                chunk_text=text_by_chunk.get(cid, (hit.payload or {}).get("chunk_text", "")),
                chunk_index=(hit.payload or {}).get("chunk_index", 0),
                section_type=(hit.payload or {}).get("section_type", "unknown"),
                score=round(hit.score, 6),
            )
            for cid, hit in zip(chunk_ids, raw)
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
    """
    FastAPI dependency / module-level accessor for the QdrantSearchService.
    Returns a singleton that re-uses the pre-loaded model and Qdrant client.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = QdrantSearchService()
    return _service_instance
