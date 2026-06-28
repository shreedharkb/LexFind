"""
LexFind — Qdrant Search Service
==================================
Core search layer using Qdrant (Hybrid Search with Sparse/Dense Vectors).

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
    FieldCondition, Filter, Fusion, FusionQuery, MatchText,
    MatchValue, Prefetch, Range, ScoredPoint, SparseVector,
)
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.search_schemas import (
    AskResponse, CaseDetailResponse, CaseResult,
    ChunkResult, SearchResponse, SimilarCasesResponse,
)

logger = logging.getLogger(__name__)

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "legal_corpus")
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

_CHUNK_FETCH_MULTIPLIER: int = 10

_embedding_model: Optional[SentenceTransformer] = None
_bm25_model: Optional[SparseTextEmbedding] = None
_qdrant_client: Optional[QdrantClient] = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading dense embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_bm25_model() -> SparseTextEmbedding:
    global _bm25_model
    if _bm25_model is None:
        logger.info("Loading BM25 sparse model...")
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25_model


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    return _qdrant_client


def _embed(text_input: str) -> List[float]:
    model = _get_embedding_model()
    vec = model.encode([text_input], normalize_embeddings=True)
    return vec[0].tolist()


def _embed_sparse(text_input: str) -> SparseVector:
    model = _get_bm25_model()
    result = list(model.query_embed(text_input))[0]
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())


def _build_filter(
    court: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    state: Optional[str] = None,
    case_type: Optional[str] = None,
    section_type: Optional[str] = None,
) -> Optional[Filter]:
    conditions = []
    if court: conditions.append(FieldCondition(key="court", match=MatchValue(value=court)))
    if state: conditions.append(FieldCondition(key="state", match=MatchValue(value=state)))
    if case_type: conditions.append(FieldCondition(key="case_type", match=MatchValue(value=case_type)))
    if section_type: conditions.append(FieldCondition(key="section_type", match=MatchValue(value=section_type)))
    
    if year_min is not None or year_max is not None:
        conditions.append(FieldCondition(key="year", range=Range(gte=year_min, lte=year_max)))
        
    return Filter(must=conditions) if conditions else None


def _fetch_metadata_batch(db: Session, document_ids: List[str]) -> Dict[str, dict]:
    if not document_ids:
        return {}
    rows = db.execute(
        text("""
            SELECT id::text, title, petitioner, respondent, court, state, year, citation, judges, case_type, page_count, chunk_strategy
            FROM legal_documents WHERE id = ANY(CAST(:ids AS uuid[]))
        """), {"ids": document_ids}
    ).fetchall()
    return {
        row[0]: {
            "title": row[1], "petitioner": row[2], "respondent": row[3], "court": row[4],
            "state": row[5], "year": row[6], "citation": row[7], "judges": row[8] or [],
            "case_type": row[9], "page_count": row[10], "chunk_strategy": row[11],
        } for row in rows
    }


def _fetch_chunk_texts_batch(db: Session, chunk_ids: List[str]) -> Dict[str, str]:
    if not chunk_ids: return {}
    rows = db.execute(text("SELECT id::text, chunk_text FROM legal_chunks WHERE id = ANY(CAST(:ids AS uuid[]))"), {"ids": chunk_ids}).fetchall()
    return {row[0]: row[1] for row in rows}


def _qdrant_hits_to_case_results(hits: List[ScoredPoint], db: Session, exclude_document_id: Optional[str] = None, top_k: int = 10) -> List[CaseResult]:
    doc_chunks: Dict[str, List[ScoredPoint]] = defaultdict(list)
    for hit in hits:
        doc_id = (hit.payload or {}).get("document_id")
        if doc_id and (not exclude_document_id or doc_id != exclude_document_id):
            doc_chunks[doc_id].append(hit)

    sorted_docs = sorted(doc_chunks.items(), key=lambda kv: max(h.score for h in kv[1]), reverse=True)[:top_k]
    if not sorted_docs:
        return []

    doc_ids = [did for did, _ in sorted_docs]
    chunk_ids = [hit.payload.get("chunk_id") for _, hlist in sorted_docs for hit in hlist if hit.payload.get("chunk_id")]

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
        results.append(CaseResult(
            document_id=doc_id, title=meta.get("title"), petitioner=meta.get("petitioner"),
            respondent=meta.get("respondent"), court=meta.get("court"), year=meta.get("year"),
            citation=meta.get("citation"), judges=meta.get("judges") or [], case_type=meta.get("case_type"),
            state=meta.get("state"), score=round(max(h.score for h in hlist), 6), top_chunk=chunk_results[0],
            all_chunks=chunk_results,
        ))
    return results


class QdrantSearchService:
    def __init__(self) -> None:
        try:
            _get_embedding_model()
            _get_qdrant_client()
        except Exception as exc:
            logger.warning("Could not pre-warm search service singletons: %s", exc)

    def search(self, db: Session, query: str, *, court: Optional[str] = None, year_min: Optional[int] = None, year_max: Optional[int] = None, state: Optional[str] = None, case_type: Optional[str] = None, section_type: Optional[str] = None, top_k: int = 10, search_mode: str = "hybrid") -> SearchResponse:
        t0 = time.perf_counter()
        if not (query := query.strip()):
            return SearchResponse(query=query, total_results=0, results=[], search_time_ms=0.0)

        qdrant_filter = _build_filter(court, year_min, year_max, state, case_type, section_type)
        client, limit = _get_qdrant_client(), top_k * _CHUNK_FETCH_MULTIPLIER

        if search_mode == "keyword":
            raw = client.query_points(
                collection_name=QDRANT_COLLECTION, query=_embed_sparse(query), using="sparse",
                query_filter=qdrant_filter, limit=limit, with_payload=True,
            ).points
        else:
            raw = client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    Prefetch(query=_embed(query), using="dense", filter=qdrant_filter, limit=limit),
                    Prefetch(query=_embed_sparse(query), using="sparse", filter=qdrant_filter, limit=limit),
                ],
                query=FusionQuery(fusion=Fusion.RRF), limit=limit, with_payload=True,
            ).points

        results = _qdrant_hits_to_case_results(raw, db, top_k=top_k)
        return SearchResponse(query=query, total_results=len(results), results=results, search_time_ms=round((time.perf_counter() - t0) * 1000, 2))

    def search_by_case_name(self, db: Session, case_name: str, *, top_k: int = 10) -> SearchResponse:
        t0 = time.perf_counter()
        if not (case_name := case_name.strip()):
            return SearchResponse(query=case_name, total_results=0, results=[], search_time_ms=0.0)

        raw = _get_qdrant_client().query_points(
            collection_name=QDRANT_COLLECTION, query=_embed(case_name),
            query_filter=Filter(must=[FieldCondition(key="title", match=MatchText(text=case_name))]),
            limit=top_k * _CHUNK_FETCH_MULTIPLIER, with_payload=True,
        ).points

        results = _qdrant_hits_to_case_results(raw, db, top_k=top_k)
        return SearchResponse(query=case_name, total_results=len(results), results=results, search_time_ms=round((time.perf_counter() - t0) * 1000, 2))

    def get_similar_cases(self, db: Session, document_id: str, *, top_k: int = 10) -> SimilarCasesResponse:
        client, doc_filter = _get_qdrant_client(), Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        all_vectors, next_offset = [], None

        while True:
            result, next_offset = client.scroll(collection_name=QDRANT_COLLECTION, scroll_filter=doc_filter, limit=200, offset=next_offset, with_vectors=True, with_payload=False)
            all_vectors.extend([p.vector for p in result if p.vector is not None])
            if next_offset is None: break

        if not all_vectors:
            return SimilarCasesResponse(source_document_id=document_id, total_results=0, results=[])

        centroid = np.mean(all_vectors, axis=0)
        if (norm := np.linalg.norm(centroid)) > 0:
            centroid /= norm

        raw = client.query_points(collection_name=QDRANT_COLLECTION, query=centroid.tolist(), limit=(top_k + 1) * _CHUNK_FETCH_MULTIPLIER, with_payload=True).points
        results = _qdrant_hits_to_case_results(raw, db, exclude_document_id=document_id, top_k=top_k)
        return SimilarCasesResponse(source_document_id=document_id, total_results=len(results), results=results)

    def get_case_detail(self, db: Session, document_id: str) -> CaseDetailResponse:
        doc = db.execute(text("SELECT id::text, title, petitioner, respondent, court, state, year, citation, judges, case_type, page_count, chunk_strategy FROM legal_documents WHERE id = CAST(:doc_id AS uuid)"), {"doc_id": document_id}).fetchone()
        if not doc: return None
        
        chunk_rows = db.execute(text("SELECT id::text, chunk_text, chunk_index, section_type FROM legal_chunks WHERE document_id = CAST(:doc_id AS uuid) ORDER BY chunk_index"), {"doc_id": document_id}).fetchall()
        chunks = [ChunkResult(chunk_id=r[0], chunk_text=r[1], chunk_index=r[2], section_type=r[3] or "unknown", score=1.0) for r in chunk_rows]
        return CaseDetailResponse(document_id=doc[0], title=doc[1], petitioner=doc[2], respondent=doc[3], court=doc[4], state=doc[5], year=doc[6], citation=doc[7], judges=doc[8] or [], case_type=doc[9], page_count=doc[10], chunk_strategy=doc[11], chunks=chunks)

    def ask(self, db: Session, document_id: str, question: str, *, top_k: int = 5) -> AskResponse:
        question = question.strip()
        doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        
        raw = _get_qdrant_client().query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=[
                Prefetch(query=_embed(question), using="dense", filter=doc_filter, limit=top_k * 3),
                Prefetch(query=_embed_sparse(question), using="sparse", filter=doc_filter, limit=top_k * 3),
            ],
            query=FusionQuery(fusion=Fusion.RRF), limit=top_k, with_payload=True,
        ).points

        chunk_ids = [(h.payload or {}).get("chunk_id", str(h.id)) for h in raw]
        text_by_chunk = _fetch_chunk_texts_batch(db, chunk_ids)

        context_chunks = [
            ChunkResult(chunk_id=cid, chunk_text=text_by_chunk.get(cid, (h.payload or {}).get("chunk_text", "")), chunk_index=(h.payload or {}).get("chunk_index", 0), section_type=(h.payload or {}).get("section_type", "unknown"), score=round(h.score, 6))
            for cid, h in zip(chunk_ids, raw)
        ]
        return AskResponse(document_id=document_id, question=question, context_chunks=context_chunks, total_chunks=len(context_chunks))


_service_instance: Optional[QdrantSearchService] = None

def get_search_service() -> QdrantSearchService:
    global _service_instance
    if _service_instance is None:
        _service_instance = QdrantSearchService()
    return _service_instance
