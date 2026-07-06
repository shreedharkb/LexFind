"""
LexFind — Qdrant Search Service
==================================
Core search layer using Qdrant.

Supports both collection formats:
  - Legacy: single unnamed dense vector (768-dim COSINE)
  - Hybrid: named "dense" + "sparse" vectors with RRF fusion

Auto-detects the collection format at startup and routes queries accordingly.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    Range,
    ScoredPoint,
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

_CHUNK_FETCH_MULTIPLIER: int = 10

# ── Module-level singletons ───────────────────────────────────────────────────

_embedding_model: Optional[SentenceTransformer] = None
_qdrant_client: Optional[QdrantClient] = None
_collection_has_named_vectors: Optional[bool] = None  # Auto-detected at startup
_bm25_model = None  # Lazy-loaded only if hybrid collection detected


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading dense embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Dense embedding model loaded (dim=%d)", _embedding_model.get_sentence_embedding_dimension())
    return _embedding_model


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        logger.info("Connecting to Qdrant at %s:%d", QDRANT_HOST, QDRANT_PORT)
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    return _qdrant_client


def _detect_collection_format() -> bool:
    """
    Auto-detect whether the Qdrant collection uses named vectors ("dense"/"sparse")
    or a single unnamed vector. Returns True if named vectors exist.
    """
    global _collection_has_named_vectors
    if _collection_has_named_vectors is not None:
        return _collection_has_named_vectors

    try:
        client = _get_qdrant_client()
        info = client.get_collection(QDRANT_COLLECTION)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict) and "dense" in vectors_config:
            _collection_has_named_vectors = True
            logger.info("Qdrant collection '%s': HYBRID mode (named dense+sparse vectors)", QDRANT_COLLECTION)
        else:
            _collection_has_named_vectors = False
            logger.info("Qdrant collection '%s': SIMPLE mode (single unnamed vector)", QDRANT_COLLECTION)
    except Exception as exc:
        logger.warning("Could not detect collection format: %s. Defaulting to simple mode.", exc)
        _collection_has_named_vectors = False

    return _collection_has_named_vectors


def _get_bm25_model():
    """Lazy-load the BM25 sparse model. Only needed for hybrid collections."""
    global _bm25_model
    if _bm25_model is None:
        try:
            from fastembed import SparseTextEmbedding
            logger.info("Loading BM25 sparse model (fastembed) ...")
            _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
            logger.info("BM25 model loaded.")
        except ImportError:
            logger.warning("fastembed not installed. Sparse search disabled.")
            _bm25_model = False  # Sentinel to avoid re-trying
        except Exception as exc:
            logger.warning("Failed to load BM25 model: %s", exc)
            _bm25_model = False
    return _bm25_model if _bm25_model is not False else None


# ── Internal helpers ───────────────────────────────────────────────────────────

def _embed(text_input: str) -> List[float]:
    """Embed a single string into a dense vector (unit-normalised float list)."""
    model = _get_embedding_model()
    vec = model.encode([text_input], normalize_embeddings=True)
    return vec[0].tolist()


def _embed_sparse(text_input: str):
    """Embed a single string into a BM25 sparse vector for Qdrant."""
    model = _get_bm25_model()
    if model is None:
        return None
    from qdrant_client.models import SparseVector
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
        conditions.append(FieldCondition(key="year", range=Range(gte=year_min, lte=year_max)))
    return Filter(must=conditions) if conditions else None


def _fetch_metadata_batch(db: Session, document_ids: List[str]) -> Dict[str, dict]:
    """Fetch document metadata from PostgreSQL using a properly-bound IN clause."""
    if not document_ids:
        return {}
    # Build  id = ANY(ARRAY[...])  by expanding each UUID individually to avoid
    # the SQLAlchemy list-as-string binding bug that breaks CAST(:ids AS uuid[]).
    placeholders = ", ".join(f"CAST(:id_{i} AS uuid)" for i in range(len(document_ids)))
    params = {f"id_{i}": did for i, did in enumerate(document_ids)}
    try:
        rows = db.execute(
            text(f"""
                SELECT id::text, title, petitioner, respondent, court, state, year,
                       citation, judges, case_type, page_count, chunk_strategy
                FROM legal_documents WHERE id IN ({placeholders})
            """),
            params,
        ).fetchall()
        return {
            row[0]: {
                "title": row[1], "petitioner": row[2], "respondent": row[3],
                "court": row[4], "state": row[5], "year": row[6], "citation": row[7],
                "judges": row[8] or [], "case_type": row[9], "page_count": row[10],
                "chunk_strategy": row[11],
            }
            for row in rows
        }
    except Exception as exc:
        logger.warning("_fetch_metadata_batch failed: %s", exc)
        return {}


def _fetch_chunk_texts_batch(db: Session, chunk_ids: List[str]) -> Dict[str, str]:
    """Fetch chunk texts from PostgreSQL using a properly-bound IN clause."""
    if not chunk_ids:
        return {}
    placeholders = ", ".join(f"CAST(:id_{i} AS uuid)" for i in range(len(chunk_ids)))
    params = {f"id_{i}": cid for i, cid in enumerate(chunk_ids)}
    try:
        rows = db.execute(
            text(f"SELECT id::text, chunk_text FROM legal_chunks WHERE id IN ({placeholders})"),
            params,
        ).fetchall()
        return {row[0]: row[1] for row in rows}
    except Exception as exc:
        logger.warning("_fetch_chunk_texts_batch failed: %s", exc)
        return {}


def _meta_from_payload(payload: dict) -> dict:
    """Extract document-level metadata from a Qdrant point payload.

    The ingestor stores title/court/year/etc. directly in the payload,
    so we can avoid an expensive (and bug-prone) PostgreSQL round-trip.
    """
    return {
        "title":      payload.get("title"),
        "petitioner": payload.get("petitioner"),
        "respondent": payload.get("respondent"),
        "court":      payload.get("court"),
        "state":      payload.get("state"),
        "year":       payload.get("year"),
        "citation":   payload.get("citation"),
        "judges":     payload.get("judges") or [],
        "case_type":  payload.get("case_type"),
    }


def _qdrant_hits_to_case_results(
    hits: List[ScoredPoint], db: Session,
    exclude_document_id: Optional[str] = None, top_k: int = 10,
) -> List[CaseResult]:
    doc_chunks: Dict[str, List[ScoredPoint]] = defaultdict(list)
    for hit in hits:
        doc_id = (hit.payload or {}).get("document_id")
        if doc_id and (not exclude_document_id or doc_id != exclude_document_id):
            doc_chunks[doc_id].append(hit)

    sorted_docs = sorted(
        doc_chunks.items(), key=lambda kv: max(h.score for h in kv[1]), reverse=True,
    )[:top_k]
    if not sorted_docs:
        return []

    # Collect chunk_ids that need full text from PostgreSQL
    chunk_ids = [
        hit.payload.get("chunk_id")
        for _, hlist in sorted_docs for hit in hlist
        if hit.payload.get("chunk_id")
    ]
    text_by_chunk = _fetch_chunk_texts_batch(db, chunk_ids)

    # If the Qdrant payload doesn't carry a title (old ingestion), fall back to PG.
    doc_ids_missing_meta = [
        doc_id for doc_id, hlist in sorted_docs
        if not (hlist[0].payload or {}).get("title")
    ]
    pg_meta: Dict[str, dict] = {}
    if doc_ids_missing_meta:
        pg_meta = _fetch_metadata_batch(db, doc_ids_missing_meta)

    results = []
    for doc_id, hlist in sorted_docs:
        # Prefer metadata from Qdrant payload (always present, no SQL needed).
        # Fall back to the PG query result if the payload lacks a title.
        first_payload = hlist[0].payload or {}
        if first_payload.get("title"):
            meta = _meta_from_payload(first_payload)
        else:
            meta = pg_meta.get(doc_id, _meta_from_payload(first_payload))

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
            document_id=doc_id, title=meta.get("title"),
            petitioner=meta.get("petitioner"), respondent=meta.get("respondent"),
            court=meta.get("court"), year=meta.get("year"),
            citation=meta.get("citation"), judges=meta.get("judges") or [],
            case_type=meta.get("case_type"), state=meta.get("state"),
            score=round(max(h.score for h in hlist), 6),
            top_chunk=chunk_results[0], all_chunks=chunk_results,
        ))
    return results


# ── Search helpers (simple vs hybrid) ──────────────────────────────────────────

def _search_simple(client: QdrantClient, query_vector: List[float],
                   qdrant_filter: Optional[Filter], limit: int) -> List[ScoredPoint]:
    """
    Search using a single unnamed dense vector via direct HTTP POST to Qdrant REST API.
    Uses httpx instead of qdrant-client to bypass library timeout/version bugs.
    Proven working approach (commit d9b3252).
    """
    import httpx, json as _json
    payload: dict = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
    }
    if qdrant_filter is not None:
        payload["filter"] = _json.loads(qdrant_filter.model_dump_json(exclude_none=True))

    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION}/points/search"
    try:
        response = httpx.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        points_data = response.json().get("result", [])
        return [
            ScoredPoint(
                id=p["id"],
                version=p.get("version", 0),
                score=p["score"],
                payload=p.get("payload", {}),
                vector=None,
            )
            for p in points_data
        ]
    except Exception as exc:
        logger.error("_search_simple HTTP POST failed: %s", exc)
        raise


def _search_hybrid(client: QdrantClient, query: str, query_vector: List[float],
                   qdrant_filter: Optional[Filter], limit: int) -> List[ScoredPoint]:
    """Search using named dense+sparse vectors with RRF fusion (hybrid format)."""
    from qdrant_client.models import Fusion, FusionQuery, Prefetch

    sparse_vec = _embed_sparse(query)
    prefetch = [
        Prefetch(query=query_vector, using="dense", filter=qdrant_filter, limit=limit),
    ]
    if sparse_vec is not None:
        prefetch.append(
            Prefetch(query=sparse_vec, using="sparse", filter=qdrant_filter, limit=limit),
        )

    return client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points


# ── Service class ──────────────────────────────────────────────────────────────

class QdrantSearchService:
    """Qdrant-backed search service. Auto-detects collection format."""

    def __init__(self) -> None:
        try:
            _get_embedding_model()
            _get_qdrant_client()
            _detect_collection_format()
            if _collection_has_named_vectors:
                _get_bm25_model()
        except Exception as exc:
            logger.warning("Could not pre-warm search service singletons: %s", exc)

    def search(
        self, db: Session, query: str, *, court: Optional[str] = None,
        year_min: Optional[int] = None, year_max: Optional[int] = None,
        state: Optional[str] = None, case_type: Optional[str] = None,
        section_type: Optional[str] = None, top_k: int = 10,
        search_mode: str = "hybrid",
    ) -> SearchResponse:
        t0 = time.perf_counter()
        if not (query := query.strip()):
            return SearchResponse(query=query, total_results=0, results=[], search_time_ms=0.0)

        client = _get_qdrant_client()
        limit = top_k * _CHUNK_FETCH_MULTIPLIER

        # ── Keyword / exact-match mode ─────────────────────────────────────────
        # When the user selects "Keyword Search" we use Qdrant's MatchText filter
        # on the title field combined with a scroll (no vector needed).  We still
        # fall through to semantic if zero results are found.
        if search_mode == "keyword":
            try:
                kw_must = [FieldCondition(key="title", match=MatchText(text=query))]
                base_conditions = _build_filter(court, year_min, year_max, state, case_type, section_type)
                if base_conditions:
                    kw_must.extend(base_conditions.must or [])
                kw_filter = Filter(must=kw_must)
                # scroll() returns Record objects — convert them to ScoredPoint
                # with a synthetic score=1.0 so the shared result-builder works.
                scroll_records, _ = client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter=kw_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
                raw: List[ScoredPoint] = [
                    ScoredPoint(
                        id=rec.id,
                        version=0,
                        score=1.0,
                        payload=rec.payload,
                        vector=None,
                    )
                    for rec in scroll_records
                ]
                if raw:
                    results = _qdrant_hits_to_case_results(raw, db, top_k=top_k)
                    return SearchResponse(
                        query=query, total_results=len(results), results=results,
                        search_time_ms=round((time.perf_counter() - t0) * 1000, 2),
                    )
                # Fall through to semantic search if no keyword hits
            except Exception as exc:
                logger.warning("Keyword search failed, falling back to semantic: %s", exc)

        # ── Semantic / hybrid mode ─────────────────────────────────────────────
        qdrant_filter = _build_filter(court, year_min, year_max, state, case_type, section_type)
        query_vector = _embed(query)

        try:
            if _detect_collection_format():
                raw = _search_hybrid(client, query, query_vector, qdrant_filter, limit)
            else:
                raw = _search_simple(client, query_vector, qdrant_filter, limit)
        except Exception as exc:
            logger.exception("Qdrant search failed: %s", exc)
            return SearchResponse(
                query=query, total_results=0, results=[],
                search_time_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        results = _qdrant_hits_to_case_results(raw, db, top_k=top_k)
        return SearchResponse(
            query=query, total_results=len(results), results=results,
            search_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    def search_by_case_name(self, db: Session, case_name: str, *, top_k: int = 10) -> SearchResponse:
        t0 = time.perf_counter()
        if not (case_name := case_name.strip()):
            return SearchResponse(query=case_name, total_results=0, results=[], search_time_ms=0.0)

        try:
            query_vector = _embed(case_name)
            name_filter = Filter(must=[FieldCondition(key="title", match=MatchText(text=case_name))])
            limit = top_k * _CHUNK_FETCH_MULTIPLIER

            if _detect_collection_format():
                raw = _search_hybrid(_get_qdrant_client(), case_name, query_vector, name_filter, limit)
            else:
                raw = _search_simple(_get_qdrant_client(), query_vector, name_filter, limit)
        except Exception as exc:
            logger.exception("Qdrant search_by_case_name failed: %s", exc)
            raw = []

        results = _qdrant_hits_to_case_results(raw, db, top_k=top_k)
        return SearchResponse(
            query=case_name, total_results=len(results), results=results,
            search_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    def get_similar_cases(self, db: Session, document_id: str, *, top_k: int = 10) -> SimilarCasesResponse:
        client = _get_qdrant_client()
        doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        all_vectors, next_offset = [], None
        is_hybrid = _detect_collection_format()

        try:
            while True:
                result, next_offset = client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter=doc_filter, limit=200, offset=next_offset,
                    with_vectors=["dense"] if is_hybrid else True,
                    with_payload=False,
                )
                for p in result:
                    vec = p.vector
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
            limit = (top_k + 1) * _CHUNK_FETCH_MULTIPLIER
            if is_hybrid:
                from qdrant_client.models import Prefetch, Fusion, FusionQuery
                raw = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    prefetch=[Prefetch(query=centroid.tolist(), using="dense", limit=limit)],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=limit, with_payload=True,
                ).points
            else:
                raw = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=centroid.tolist(),
                    limit=limit, with_payload=True,
                ).points
        except Exception as exc:
            logger.exception("Qdrant query similar cases failed: %s", exc)
            raw = []

        results = _qdrant_hits_to_case_results(raw, db, exclude_document_id=document_id, top_k=top_k)
        return SimilarCasesResponse(source_document_id=document_id, total_results=len(results), results=results)

    def get_case_detail(self, db: Session, document_id: str) -> CaseDetailResponse:
        doc = db.execute(
            text(
                "SELECT id::text, title, petitioner, respondent, court, state, year, "
                "citation, judges, case_type, page_count, chunk_strategy "
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
            ChunkResult(chunk_id=r[0], chunk_text=r[1], chunk_index=r[2],
                        section_type=r[3] or "unknown", score=1.0)
            for r in chunk_rows
        ]
        return CaseDetailResponse(
            document_id=doc[0], title=doc[1], petitioner=doc[2], respondent=doc[3],
            court=doc[4], state=doc[5], year=doc[6], citation=doc[7],
            judges=doc[8] or [], case_type=doc[9], page_count=doc[10],
            chunk_strategy=doc[11], chunks=chunks,
        )

    def ask(self, db: Session, document_id: str, question: str, *, top_k: int = 5) -> AskResponse:
        question = question.strip()
        doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])

        try:
            query_vector = _embed(question)
            if _detect_collection_format():
                raw = _search_hybrid(_get_qdrant_client(), question, query_vector, doc_filter, top_k)
            else:
                raw = _search_simple(_get_qdrant_client(), query_vector, doc_filter, top_k)
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
            document_id=document_id, question=question,
            context_chunks=context_chunks, total_chunks=len(context_chunks),
        )


# ── Singleton accessor ─────────────────────────────────────────────────────────

_service_instance: Optional[QdrantSearchService] = None


def get_search_service() -> QdrantSearchService:
    global _service_instance
    if _service_instance is None:
        _service_instance = QdrantSearchService()
    return _service_instance
