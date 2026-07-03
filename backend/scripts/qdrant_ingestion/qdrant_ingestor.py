"""
LexFind — Qdrant Ingestor (Step 3 of Qdrant Ingestion Pipeline)
==================================================================
Loads embeddings.npy + chunk_ids.json downloaded from Kaggle,
fetches chunk metadata from PostgreSQL, upserts all points into
the Qdrant "legal_corpus" collection, and writes qdrant_point_id
back to legal_chunks.

Features:
  - Idempotent: skips chunks already in Qdrant (qdrant_point_id is set)
  - Batch upserts (256 points per request)
  - tqdm progress bar
  - Writes qdrant_point_id back to legal_chunks after each batch
  - Marks legal_documents.qdrant_synced = TRUE after all chunks done
  - Logs errors to scripts/qdrant_ingest_errors.log
  - Prints final summary with collection info

Run from backend/:
  python scripts/qdrant_ingestor.py
  python scripts/qdrant_ingestor.py --embeddings /path/to/embeddings.npy --chunk-ids /path/to/chunk_ids.json
  python scripts/qdrant_ingestor.py --recreate-collection   # DANGER: wipes existing collection
"""

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

import json
import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine, text

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL      = os.getenv("DATABASE_URL", "")
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME   = "legal_corpus"
VECTOR_SIZE       = 768
UPSERT_BATCH_SIZE = 128   # Smaller batches = less chance of timeout during HNSW indexing
METADATA_BATCH    = 2000   # how many chunk IDs to fetch from Postgres at once

DEFAULT_EMB_PATH  = BASE_DIR / "scripts" / "qdrant_ingestion" / "embeddings.npy"
DEFAULT_IDS_PATH  = BASE_DIR / "scripts" / "qdrant_ingestion" / "chunk_ids.json"
LOG_FILE          = BASE_DIR / "scripts" / "qdrant_ingestion" / "qdrant_ingest_errors.log"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# QDRANT COLLECTION SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_collection(client: QdrantClient, recreate: bool = False):
    """Create or verify the legal_corpus collection."""
    existing = [c.name for c in client.get_collections().collections]

    if recreate and COLLECTION_NAME in existing:
        logger.info(f"Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        existing = []

    if COLLECTION_NAME not in existing:
        logger.info(f"Creating Qdrant collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,           # HNSW connections per node (higher = better recall, more RAM)
                ef_construct=100,  # build quality vs speed tradeoff
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20_000,  # start HNSW indexing after 20k vectors
            ),
        )

        # Payload indexes for fast filtered search
        logger.info("Creating payload indexes...")
        client.create_payload_index(COLLECTION_NAME, "court",        PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION_NAME, "year",         PayloadSchemaType.INTEGER)
        client.create_payload_index(COLLECTION_NAME, "state",        PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION_NAME, "case_type",    PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION_NAME, "section_type", PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION_NAME, "document_id",  PayloadSchemaType.KEYWORD)
        logger.info("Collection and indexes ready.")
    else:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(
            f"Collection '{COLLECTION_NAME}' already exists — "
            f"{info.points_count:,} points. Resuming..."
        )


# ─────────────────────────────────────────────────────────────────────────────
# POSTGRES METADATA FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_chunk_metadata_batch(conn, chunk_ids: list[str]) -> dict:
    """
    Fetch metadata for a batch of chunk UUIDs.
    Returns dict: chunk_id → {document_id, chunk_index, court, year, state,
                               case_type, title, petitioner, respondent, citation}
    """
    if not chunk_ids:
        return {}

    rows = conn.execute(
        text("""
            SELECT
                lc.id::text            AS chunk_id,
                lc.chunk_index,
                lc.section_type,
                ld.id::text            AS document_id,
                ld.title,
                ld.petitioner,
                ld.respondent,
                ld.court,
                ld.state,
                ld.year,
                ld.citation,
                ld.case_type,
                ld.filename
            FROM legal_chunks lc
            JOIN legal_documents ld ON lc.document_id = ld.id
            WHERE lc.id = ANY(CAST(:ids AS uuid[]))
        """),
        {"ids": chunk_ids},
    ).fetchall()

    return {
        row.chunk_id: {
            "chunk_id":     row.chunk_id,
            "chunk_index":  row.chunk_index,
            "section_type": row.section_type,
            "document_id":  row.document_id,
            "title":        row.title,
            "petitioner":   row.petitioner,
            "respondent":   row.respondent,
            "court":        row.court,
            "state":        row.state,
            "year":         row.year,
            "citation":     row.citation,
            "case_type":    row.case_type,
            "filename":     row.filename,
        }
        for row in rows
    }


def get_already_ingested_chunk_ids(conn) -> set:
    """Return chunk_ids that already have qdrant_point_id set (resume support)."""
    rows = conn.execute(
        text("SELECT id::text FROM legal_chunks WHERE qdrant_point_id IS NOT NULL")
    ).fetchall()
    return {r[0] for r in rows}


def update_chunk_point_ids(conn, chunk_point_map: dict):
    """
    Write qdrant_point_id back to legal_chunks for each (chunk_id → point_id) pair.
    """
    if not chunk_point_map:
        return
    rows = [
        {"chunk_id": cid, "point_id": pid}
        for cid, pid in chunk_point_map.items()
    ]
    conn.execute(
        text("""
            UPDATE legal_chunks
            SET qdrant_point_id = CAST(:point_id AS uuid)
            WHERE id = CAST(:chunk_id AS uuid)
        """),
        rows,
    )


def mark_documents_synced(conn, document_ids: list[str]):
    """Mark legal_documents.qdrant_synced = TRUE for all documents whose chunks are done."""
    if not document_ids:
        return
    conn.execute(
        text("""
            UPDATE legal_documents
            SET qdrant_synced = TRUE
            WHERE id = ANY(CAST(:ids AS uuid[]))
              AND NOT EXISTS (
                  SELECT 1 FROM legal_chunks
                  WHERE document_id = legal_documents.id
                    AND qdrant_point_id IS NULL
              )
        """),
        {"ids": document_ids},
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INGESTOR
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LexFind Qdrant Ingestor")
    parser.add_argument("--embeddings",        default=str(DEFAULT_EMB_PATH), help="Path to embeddings.npy")
    parser.add_argument("--chunk-ids",         default=str(DEFAULT_IDS_PATH), help="Path to chunk_ids.json")
    parser.add_argument("--recreate-collection", action="store_true",
                        help="DELETE existing collection and recreate (WARNING: destructive)")
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    ids_path = Path(args.chunk_ids)

    if not emb_path.exists():
        logger.error(f"embeddings.npy not found: {emb_path}")
        sys.exit(1)
    if not ids_path.exists():
        logger.error(f"chunk_ids.json not found: {ids_path}")
        sys.exit(1)
    if not DATABASE_URL:
        logger.error("DATABASE_URL is not set in .env")
        sys.exit(1)

    # ── Load embeddings (memory-mapped — avoids loading 3.5 GB into RAM) ─────────
    logger.info(f"Loading embeddings from {emb_path} ...")
    embeddings = np.load(str(emb_path), mmap_mode="r")
    logger.info(f"Embeddings shape: {embeddings.shape} — memory-mapped, dtype={embeddings.dtype}")

    logger.info(f"Loading chunk IDs from {ids_path} ...")
    with open(ids_path, "r", encoding="utf-8") as f:
        chunk_ids: list[str] = json.load(f)

    assert len(chunk_ids) == embeddings.shape[0], (
        f"Mismatch: {len(chunk_ids)} chunk_ids vs {embeddings.shape[0]} embeddings"
    )
    assert embeddings.shape[1] == VECTOR_SIZE, (
        f"Expected {VECTOR_SIZE}-dim vectors, got {embeddings.shape[1]}"
    )
    logger.info(f"Loaded {len(chunk_ids):,} chunk IDs.")

    # ── Qdrant client ──────────────────────────────────────────────────────────
    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} ...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)  # 5min for HNSW indexing periods
    setup_collection(client, recreate=args.recreate_collection)

    # ── Postgres engine ────────────────────────────────────────────────────────
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    with engine.connect() as conn:
        already_done = get_already_ingested_chunk_ids(conn)

    logger.info(f"Already ingested: {len(already_done):,} chunks — will skip.")

    # Filter out already-done chunks
    pending_indices = [
        i for i, cid in enumerate(chunk_ids) if cid not in already_done
    ]
    logger.info(f"Chunks to ingest: {len(pending_indices):,}")

    if not pending_indices:
        logger.info("Nothing to ingest — all chunks already in Qdrant!")
        _print_final_stats(client, 0, 0, time.time())
        return

    # ── Ingest in batches ──────────────────────────────────────────────────────
    t_start = time.time()
    total_upserted = 0
    total_errors = 0
    affected_document_ids: set[str] = set()

    with tqdm(total=len(pending_indices), desc="Upserting to Qdrant", unit="chunk") as pbar:
        for batch_start in range(0, len(pending_indices), UPSERT_BATCH_SIZE):
            batch_idx = pending_indices[batch_start: batch_start + UPSERT_BATCH_SIZE]
            batch_chunk_ids  = [chunk_ids[i]  for i in batch_idx]
            # Read only this batch from disk (mmap) and cast to float32
            batch_embeddings = np.array([embeddings[i] for i in batch_idx], dtype="float32")

            # Fetch metadata from Postgres for this batch
            try:
                with engine.connect() as conn:
                    meta_map = fetch_chunk_metadata_batch(conn, batch_chunk_ids)
            except Exception as e:
                logger.error(f"Postgres metadata fetch failed for batch at {batch_start}: {e}")
                total_errors += len(batch_idx)
                pbar.update(len(batch_idx))
                continue

            # Build PointStructs
            points: list[PointStruct] = []
            chunk_point_map: dict[str, str] = {}  # chunk_id → qdrant_point_id

            for chunk_id, vector in zip(batch_chunk_ids, batch_embeddings):
                meta = meta_map.get(chunk_id)
                if not meta:
                    logger.warning(f"No metadata found for chunk {chunk_id} — skipping")
                    total_errors += 1
                    continue

                point_id = str(uuid.uuid4())
                chunk_point_map[chunk_id] = point_id

                if meta["document_id"]:
                    affected_document_ids.add(meta["document_id"])

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            "document_id":  meta["document_id"],
                            "chunk_id":     chunk_id,
                            "chunk_index":  meta["chunk_index"],
                            "section_type": meta["section_type"],
                            "filename":     meta["filename"],
                            "title":        meta["title"],
                            "petitioner":   meta["petitioner"],
                            "respondent":   meta["respondent"],
                            "court":        meta["court"],
                            "state":        meta["state"],
                            "year":         meta["year"],
                            "citation":     meta["citation"],
                            "case_type":    meta["case_type"],
                        },
                    )
                )

            if not points:
                pbar.update(len(batch_idx))
                continue

            # Upsert to Qdrant
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True,
                )
            except Exception as e:
                logger.error(f"Qdrant upsert failed for batch at {batch_start}: {e}")
                total_errors += len(points)
                pbar.update(len(batch_idx))
                continue

            # Write point IDs back to Postgres
            try:
                with engine.begin() as txn:
                    update_chunk_point_ids(txn, chunk_point_map)
            except Exception as e:
                logger.error(f"Postgres update failed for batch at {batch_start}: {e}")

            total_upserted += len(points)
            pbar.update(len(batch_idx))

    # ── Mark documents as synced ───────────────────────────────────────────────
    if affected_document_ids:
        logger.info(f"Marking {len(affected_document_ids):,} documents as qdrant_synced=TRUE ...")
        with engine.begin() as conn:
            mark_documents_synced(conn, list(affected_document_ids))

    _print_final_stats(client, total_upserted, total_errors, t_start)


def _print_final_stats(client: QdrantClient, total_upserted: int, total_errors: int, t_start: float):
    elapsed = time.time() - t_start
    try:
        info = client.get_collection(COLLECTION_NAME)
        vec_count = info.points_count
        status    = info.status
    except Exception:
        vec_count = "?"
        status    = "?"

    print("\n" + "=" * 60)
    print("  LexFind — Qdrant Ingestion Complete")
    print("=" * 60)
    print(f"  Points upserted this run : {total_upserted:,}")
    print(f"  Errors                   : {total_errors:,}")
    print(f"  Time taken               : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Collection status        : {status}")
    print(f"  Total vectors in Qdrant  : {vec_count:,}")
    print(f"  Error log                : {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
