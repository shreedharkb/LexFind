"""
LexFind — Qdrant Search Service
==================================
Core search layer using Qdrant.

Uses hybrid search with named vectors:
  - "dense" vector (768-dim, sentence-transformers/all-mpnet-base-v2)
  - "sparse" vector (BM25 via fastembed)
  - RRF fusion to combine both

This matches the collection format created by the LexFind ingestion
pipeline. All queries go through query_points() with Prefetch + FusionQuery.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
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
QDRANT_URL: str = os.getenv("QDRANT_URL", "").strip().rstrip("/")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "legal_corpus_hybrid")
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

_CHUNK_FETCH_MULTIPLIER: int = 10
_QDRANT_QUERY_TIMEOUT_SECONDS: float = float(
    os.getenv("QDRANT_QUERY_TIMEOUT_SECONDS", "12")
)

# ── Module-level singletons ───────────────────────────────────────────────────

_embedding_model: Optional[SentenceTransformer] = None
_qdrant_client: Optional[QdrantClient] = None
_bm25_model = None  # Lazy-loaded
_bm25_model_load_attempted = False


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
        if QDRANT_URL:
            logger.info("Connecting to Qdrant at %s", QDRANT_URL)
            _qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY or None,
                timeout=120,
            )
        else:
            logger.info("Connecting to Qdrant at %s:%d", QDRANT_HOST, QDRANT_PORT)
            _qdrant_client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY or None,
                timeout=120,
            )
    return _qdrant_client


def _qdrant_rest_url() -> str:
    """Return the Qdrant REST URL for raw hybrid-query requests."""
    base_url = QDRANT_URL or f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    return f"{base_url}/collections/{QDRANT_COLLECTION}/points/query"


def _qdrant_request_headers() -> Optional[Dict[str, str]]:
    """Supply the Qdrant Cloud API-key header only when configured."""
    return {"api-key": QDRANT_API_KEY} if QDRANT_API_KEY else None


def _get_bm25_model():
    """Lazy-load the BM25 sparse model."""
    global _bm25_model, _bm25_model_load_attempted
    if _bm25_model is None:
        if _bm25_model_load_attempted:
            return None
        _bm25_model_load_attempted = True
        try:
            from fastembed import SparseTextEmbedding
            logger.info("Loading BM25 sparse model (fastembed) ...")
            _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
            logger.info("BM25 model loaded.")
        except ImportError:
            logger.warning("fastembed not installed — sparse/BM25 search disabled. "
                           "Install with: pip install fastembed")
        except Exception as exc:
            logger.warning("Failed to load BM25 model: %s", exc)
    return _bm25_model


# ── Internal helpers ───────────────────────────────────────────────────────────

def _embed(text_input: str) -> List[float]:
    """Embed a single string into a dense vector (unit-normalised float list)."""
    model = _get_embedding_model()
    vec = model.encode([text_input], normalize_embeddings=True)
    return vec[0].tolist()


def _embed_sparse(text_input: str) -> Optional[SparseVector]:
    """Embed a single string into a BM25 sparse vector for Qdrant."""
    model = _get_bm25_model()
    if model is None:
        return None
    try:
        result = list(model.query_embed(text_input))
        sparse = result[0]
        return SparseVector(
            indices=sparse.indices.tolist(),
            values=sparse.values.tolist(),
        )
    except Exception as exc:
        logger.warning("BM25 sparse embedding failed: %s", exc)
        return None


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
    dense_query: Optional[List[float]] = None,
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

        # Hybrid RRF scores represent rank fusion, not cosine similarity.  If
        # the hybrid request included dense vectors, expose the best true dense
        # cosine score for the UI's percentage badge while retaining RRF for
        # ordering the results.
        semantic_scores = []
        if dense_query is not None:
            for hit in hlist:
                vector = getattr(hit, "vector", None)
                if isinstance(vector, dict):
                    vector = vector.get("dense")
                if vector is not None:
                    try:
                        semantic_scores.append(
                            float(np.dot(np.asarray(vector), np.asarray(dense_query)))
                        )
                    except (TypeError, ValueError):
                        pass

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
            semantic_score=round(max(semantic_scores), 6) if semantic_scores else None,
            top_chunk=chunk_results[0], all_chunks=chunk_results,
        ))
    return results


def _search_by_case_name_in_database(
    db: Session, case_name: str, top_k: int,
) -> List[CaseResult]:
    """Find cases by title or party name without requiring a Qdrant text index."""
    escape_char = chr(92)
    escaped_name = (
        case_name.replace(escape_char, escape_char * 2)
        .replace("%", f"{escape_char}%")
        .replace("_", f"{escape_char}_")
    )
    params = {
        "exact_name": case_name,
        "prefix": f"{escaped_name}%",
        "pattern": f"%{escaped_name}%",
        "top_k": top_k,
    }
    rows = db.execute(
        text("""
            SELECT
                ld.id::text, ld.title, ld.petitioner, ld.respondent, ld.court,
                ld.year, ld.citation, ld.judges, ld.case_type, ld.state,
                lc.id::text, lc.chunk_text, lc.chunk_index, lc.section_type
            FROM legal_documents ld
            LEFT JOIN LATERAL (
                SELECT id, chunk_text, chunk_index, section_type
                FROM legal_chunks
                WHERE document_id = ld.id
                ORDER BY chunk_index
                LIMIT 1
            ) lc ON TRUE
            WHERE COALESCE(ld.title, '') ILIKE :pattern ESCAPE CHR(92)
               OR COALESCE(ld.petitioner, '') ILIKE :pattern ESCAPE CHR(92)
               OR COALESCE(ld.respondent, '') ILIKE :pattern ESCAPE CHR(92)
            ORDER BY CASE
                WHEN LOWER(COALESCE(ld.title, '')) = LOWER(:exact_name) THEN 0
                WHEN LOWER(COALESCE(ld.petitioner, '')) = LOWER(:exact_name)
                  OR LOWER(COALESCE(ld.respondent, '')) = LOWER(:exact_name) THEN 1
                WHEN COALESCE(ld.title, '') ILIKE :prefix ESCAPE CHR(92) THEN 2
                ELSE 3
            END, ld.title NULLS LAST
            LIMIT :top_k
        """),
        params,
    ).fetchall()

    results = []
    for row in rows:
        chunk_id = row[10] or row[0]
        chunk = ChunkResult(
            chunk_id=chunk_id,
            chunk_text=row[11] or "",
            chunk_index=row[12] or 0,
            section_type=row[13] or "unknown",
            score=1.0,
        )
        results.append(CaseResult(
            document_id=row[0], title=row[1], petitioner=row[2], respondent=row[3],
            court=row[4], year=row[5], citation=row[6], judges=row[7] or [],
            case_type=row[8], state=row[9], score=1.0,
            top_chunk=chunk, all_chunks=[chunk],
        ))
    return results


# ── Vector search (named dense + sparse vector collection) ────────────────────

def _qdrant_filter_to_dict(qfilter: Optional[Filter]) -> Optional[dict]:
    """Convert the simple filters produced by _build_filter to REST JSON."""
    if not qfilter:
        return None

    # We only use simple must/match/range in _build_filter
    must_list = []
    for cond in (qfilter.must or []):
        match = getattr(cond, "match", None)
        if match and getattr(match, "value", None) is not None:
            must_list.append({"key": cond.key, "match": {"value": match.value}})
        elif match and getattr(match, "text", None):
            must_list.append({"key": cond.key, "match": {"text": match.text}})
        elif hasattr(cond, "range") and cond.range:
            must_list.append({
                "key": cond.key,
                "range": {"gte": cond.range.gte, "lte": cond.range.lte},
            })
    if not must_list:
        return None
    return {"must": must_list}

def qdrant_hybrid_search(
    query_text: str,
    qdrant_filter: Optional[Filter],
    limit: int,
    include_dense_vectors: bool = False,
) -> List[ScoredPoint]:
    """Run a dense+sparse RRF search through Qdrant's REST query endpoint."""
    dense_query = _embed(query_text)
    sparse_vec = _embed_sparse(query_text)

    body = {
        "prefetch": [{
            "query": dense_query,
            "using": "dense",
            "limit": limit,
        }],
        "query": {"fusion": "rrf"},
        "limit": limit,
        "with_payload": True,
    }
    if include_dense_vectors:
        body["with_vector"] = ["dense"]
    filter_data = _qdrant_filter_to_dict(qdrant_filter)
    if filter_data:
        body["prefetch"][0]["filter"] = filter_data

    if sparse_vec is not None:
        sparse_prefetch = {
            "query": {"indices": sparse_vec.indices, "values": sparse_vec.values},
            "using": "sparse",
            "limit": limit,
        }
        if filter_data:
            sparse_prefetch["filter"] = filter_data
        body["prefetch"].append(sparse_prefetch)

    url = _qdrant_rest_url()
    try:
        response = requests.post(
            url,
            json=body,
            headers=_qdrant_request_headers(),
            timeout=(3, _QDRANT_QUERY_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        points = response.json().get("result", {}).get("points", [])
    except requests.Timeout:
        # RRF can be expensive while Qdrant is indexing a large collection.
        # Return useful semantic results instead of leaving the search page
        # waiting indefinitely.
        logger.warning(
            "Hybrid Qdrant query timed out after %ss; falling back to dense search.",
            _QDRANT_QUERY_TIMEOUT_SECONDS,
        )
        dense_body = {
            "query": dense_query,
            "using": "dense",
            "limit": limit,
            "with_payload": True,
        }
        if include_dense_vectors:
            dense_body["with_vector"] = ["dense"]
        if filter_data:
            dense_body["filter"] = filter_data
        response = requests.post(
            url,
            json=dense_body,
            headers=_qdrant_request_headers(),
            timeout=(3, _QDRANT_QUERY_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        points = response.json().get("result", {}).get("points", [])
    return [
        ScoredPoint(
            id=point["id"],
            version=point.get("version", 0),
            score=point["score"],
            payload=point.get("payload", {}),
            vector=point.get("vector"),
        )
        for point in points
    ]

def qdrant_search_by_mode(query_text: str, qdrant_filter: Optional[Filter], limit: int, mode: str = "hybrid") -> List[ScoredPoint]:
    """Helper for dense, sparse, or hybrid routing (used by re-retriever)."""
    if mode == "hybrid":
        return qdrant_hybrid_search(query_text, qdrant_filter, limit)

    if mode == "sparse":
        query = _embed_sparse(query_text)
        if query is None:
            logger.warning("Sparse search requested but the BM25 model is unavailable; using dense search.")
            query = _embed(query_text)
            using = "dense"
        else:
            using = "sparse"
    else:  # dense
        query = _embed(query_text)
        using = "dense"

    return _get_qdrant_client().query_points(
        collection_name=QDRANT_COLLECTION,
        query=query,
        using=using,
        query_filter=qdrant_filter,
        limit=limit,
        with_payload=True,
    ).points



# ── Service class ──────────────────────────────────────────────────────────────

class QdrantSearchService:
    """Qdrant-backed search service using hybrid named-vector search."""

    def __init__(self) -> None:
        # Keep construction cheap. In particular, the party-name endpoint is a
        # PostgreSQL query and must not wait for embedding-model initialisation.
        pass

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

        limit = top_k * _CHUNK_FETCH_MULTIPLIER
        qdrant_filter = _build_filter(court, year_min, year_max, state, case_type, section_type)

        # Keyword search uses BM25 sparse vectors, which cover full case text
        # without requiring a Qdrant payload text index.
        if search_mode == "keyword":
            try:
                raw = qdrant_search_by_mode(query, qdrant_filter, limit, mode="sparse")
                if raw:
                    results = _qdrant_hits_to_case_results(raw, db, top_k=top_k)
                    return SearchResponse(
                        query=query, total_results=len(results), results=results,
                        search_time_ms=round((time.perf_counter() - t0) * 1000, 2),
                    )
                # Fall through to hybrid if no sparse hits are available.
            except Exception as exc:
                logger.warning("Keyword search failed, falling back to hybrid: %s", exc)

        # ── Semantic / hybrid mode ─────────────────────────────────────────────
        raw = qdrant_hybrid_search(
            query, qdrant_filter, limit, include_dense_vectors=True,
        )

        # The search UI displays dense cosine similarity as a percentage. Sort
        # by that same value so the visible percentage and top-to-bottom rank
        # never disagree. RRF is still used to generate the candidate set.
        # The embedding model is already warm for the RRF request, so this
        # small local encode does not create another network round trip.
        results = _qdrant_hits_to_case_results(
            raw, db, top_k=top_k, dense_query=_embed(query),
        )
        results.sort(
            key=lambda result: (
                result.semantic_score is not None,
                result.semantic_score if result.semantic_score is not None else result.score,
            ),
            reverse=True,
        )
        return SearchResponse(
            query=query, total_results=len(results), results=results,
            search_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    def search_by_case_name(self, db: Session, case_name: str, *, top_k: int = 10) -> SearchResponse:
        t0 = time.perf_counter()
        if not (case_name := case_name.strip()):
            return SearchResponse(query=case_name, total_results=0, results=[], search_time_ms=0.0)

        results = _search_by_case_name_in_database(db, case_name, top_k)
        return SearchResponse(
            query=case_name, total_results=len(results), results=results,
            search_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    def get_similar_cases(self, db: Session, document_id: str, *, top_k: int = 10) -> SimilarCasesResponse:
        client = _get_qdrant_client()
        doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        all_vectors, next_offset = [], None

        try:
            while True:
                result, next_offset = client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter=doc_filter, limit=200, offset=next_offset,
                    with_vectors=True,
                    with_payload=False,
                )
                for p in result:
                    vec = p.vector
                    if isinstance(vec, dict):
                        vec = vec.get("dense", list(vec.values())[0] if vec else None)
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
            raw = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=centroid.tolist(),
                using="dense",
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
            raw = qdrant_hybrid_search(question, doc_filter, top_k)
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
