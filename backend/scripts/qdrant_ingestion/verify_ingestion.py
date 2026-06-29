"""
LexFind — Ingestion Verification (Step 4 of Qdrant Ingestion Pipeline)
=========================================================================
Sanity-check queries against Qdrant + PostgreSQL after ingestion.

Checks:
  1. Collection info (vector count, status, config)
  2. Sample filtered searches (by court, year, case_type)
  3. Postgres sync status (what % is qdrant_synced)
  4. Spot-check: fetch a known chunk from Qdrant by point_id

Run from backend/:
  python scripts/verify_ingestion.py
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

DATABASE_URL   = os.getenv("DATABASE_URL", "")
QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION     = "legal_corpus"
MODEL_NAME     = "sentence-transformers/all-mpnet-base-v2"


def check_collection(client: QdrantClient):
    print("\n── 1. Collection Info ────────────────────────────────────")
    try:
        info = client.get_collection(COLLECTION)
        print(f"  Status       : {info.status}")
        print(f"  Vectors      : {info.points_count:,}")
        print(f"  Vector size  : {info.config.params.vectors.size}")
        print(f"  Distance     : {info.config.params.vectors.distance}")
        print(f"  HNSW m       : {info.config.hnsw_config.m}")
        print(f"  HNSW ef_c    : {info.config.hnsw_config.ef_construct}")
    except Exception as e:
        print(f"  ERROR: {e}")


def check_postgres_sync(engine):
    print("\n── 2. PostgreSQL Sync Status ─────────────────────────────")
    with engine.connect() as conn:
        total_docs = conn.execute(text("SELECT COUNT(*) FROM legal_documents")).fetchone()[0]
        synced_docs = conn.execute(text("SELECT COUNT(*) FROM legal_documents WHERE qdrant_synced = TRUE")).fetchone()[0]
        total_chunks = conn.execute(text("SELECT COUNT(*) FROM legal_chunks")).fetchone()[0]
        synced_chunks = conn.execute(text("SELECT COUNT(*) FROM legal_chunks WHERE qdrant_point_id IS NOT NULL")).fetchone()[0]

    def pct(n, d):
        return f"{n/d*100:.1f}%" if d > 0 else "0.0%"

    print(f"  Documents   : {synced_docs:,} / {total_docs:,} synced ({pct(synced_docs, total_docs)})")
    print(f"  Chunks      : {synced_chunks:,} / {total_chunks:,} synced ({pct(synced_chunks, total_chunks)})")


def check_semantic_search(client: QdrantClient, model: SentenceTransformer):
    print("\n── 3. Sample Semantic Searches ───────────────────────────")

    queries = [
        ("basic structure of constitution", None),
        ("right to privacy fundamental right", None),
        ("murder conviction appeal", {"court": "Supreme Court"}),
        ("land acquisition compensation", {"year_min": 2000, "year_max": 2015}),
        ("habeas corpus detention", {"case_type": "Writ Petition"}),
    ]

    for query_text, filters in queries:
        emb = model.encode([query_text], normalize_embeddings=True)
        vec = emb[0].tolist()

        qdrant_filter = None
        if filters:
            conditions = []
            if "court" in filters:
                conditions.append(
                    FieldCondition(key="court", match=MatchValue(value=filters["court"]))
                )
            if "case_type" in filters:
                conditions.append(
                    FieldCondition(key="case_type", match=MatchValue(value=filters["case_type"]))
                )
            if "year_min" in filters or "year_max" in filters:
                conditions.append(
                    FieldCondition(
                        key="year",
                        range=Range(
                            gte=filters.get("year_min"),
                            lte=filters.get("year_max"),
                        ),
                    )
                )
            if conditions:
                from qdrant_client.models import Filter as QFilter
                qdrant_filter = QFilter(must=conditions)

        try:
            results = client.query_points(
                collection_name=COLLECTION,
                query=vec,
                query_filter=qdrant_filter,
                limit=3,
                with_payload=["title", "court", "year", "case_type", "filename"],
            ).points
        except Exception as e:
            print(f"\n  Query : '{query_text}' → ERROR: {e}")
            continue

        filter_str = str(filters) if filters else "none"
        print(f"\n  Query  : '{query_text}'")
        print(f"  Filter : {filter_str}")
        for i, hit in enumerate(results, 1):
            p = hit.payload
            print(
                f"    [{i}] score={hit.score:.4f} | "
                f"{(p.get('title') or 'N/A')[:60]} | "
                f"{p.get('court', '?')} | {p.get('year', '?')}"
            )


def spot_check_chunk(client: QdrantClient, engine):
    print("\n── 4. Spot-Check: Random Synced Chunk ───────────────────")
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT lc.id::text, lc.qdrant_point_id::text,
                       lc.chunk_index, ld.title, ld.court, ld.year
                FROM legal_chunks lc
                JOIN legal_documents ld ON lc.document_id = ld.id
                WHERE lc.qdrant_point_id IS NOT NULL
                ORDER BY RANDOM()
                LIMIT 1
            """)
        ).fetchone()

    if not row:
        print("  No synced chunks found.")
        return

    chunk_id, point_id, chunk_index, title, court, year = row
    print(f"  Chunk ID    : {chunk_id}")
    print(f"  Point ID    : {point_id}")
    print(f"  Chunk Index : {chunk_index}")
    print(f"  Title       : {title}")
    print(f"  Court       : {court}")
    print(f"  Year        : {year}")

    # Retrieve from Qdrant by point_id
    try:
        fetched = client.retrieve(
            collection_name=COLLECTION,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if fetched:
            payload = fetched[0].payload
            print(f"  ✓ Found in Qdrant")
            print(f"    Payload title : {(payload.get('title') or 'N/A')[:60]}")
            print(f"    Payload court : {payload.get('court', 'N/A')}")
            print(f"    Payload year  : {payload.get('year', 'N/A')}")
        else:
            print(f"  ✗ NOT found in Qdrant! Sync mismatch.")
    except Exception as e:
        print(f"  ERROR retrieving from Qdrant: {e}")


def main():
    print("=" * 60)
    print("  LexFind — Ingestion Verification")
    print("=" * 60)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    check_collection(client)
    check_postgres_sync(engine)

    print("\n── Loading embedding model for semantic search checks ─────")
    model = SentenceTransformer(MODEL_NAME)
    print("   Model loaded.")

    check_semantic_search(client, model)
    spot_check_chunk(client, engine)

    print("\n" + "=" * 60)
    print("  Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
