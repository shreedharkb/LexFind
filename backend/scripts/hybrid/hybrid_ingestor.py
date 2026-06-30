"""
LexFind — Hybrid Ingestor (Step 2 of 3)
==========================================
Reads chunk_text from PostgreSQL in batches of 5,000.
For each batch:
  1. Fetches chunk_id + chunk_text + metadata from legal_chunks JOIN legal_documents
  2. Looks up dense vector from embeddings.npy using chunk_ids.json as the index map
  3. Generates BM25 sparse vector using FastEmbed
  4. Upserts batch to Qdrant with:
       - Point ID     = chunk_id (Postgres UUID) — 1:1, no translation needed
       - vector.dense  = 768-dim float vector from embeddings.npy
       - vector.sparse = BM25 sparse vector from FastEmbed
       - payload       = metadata (chunk_text is NEVER stored in Qdrant)
  5. Skips chunks already present in Qdrant (idempotent / crash-resumable)

CRITICAL ID GUARANTEE:
  Qdrant Point ID === PostgreSQL legal_chunks.id
  No random UUIDs. No translation table. Ever.

Prerequisites:
  pip install qdrant-client[fastembed] fastembed tqdm sqlalchemy psycopg2-binary

Run from backend/:
  python scripts/hybrid/hybrid_ingestor.py
  python scripts/hybrid/hybrid_ingestor.py --embeddings /custom/path/embeddings.npy
  python scripts/hybrid/hybrid_ingestor.py --chunk-ids /custom/path/chunk_ids.json
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

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
    PointStruct,
    SparseVector,
)

from fastembed import SparseTextEmbedding

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL    = os.getenv("DATABASE_URL", "")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "legal_corpus"
BATCH_SIZE      = 5_000   # rows fetched from Postgres + sparse vectors held in RAM at once
UPSERT_SIZE     = 256     # Qdrant upsert sub-batch (avoid HTTP payload limits)

# Default paths — override via CLI or env vars
DEFAULT_EMB_PATH = Path(
    os.getenv("EMBEDDINGS_PATH", str(BASE_DIR / "scripts" / "qdrant_ingestion" / "embeddings.npy"))
)
DEFAULT_IDS_PATH = Path(
    os.getenv("CHUNK_IDS_PATH", str(BASE_DIR / "scripts" / "qdrant_ingestion" / "chunk_ids.json"))
)
LOG_FILE = BASE_DIR / "scripts" / "hybrid" / "hybrid_ingest_errors.log"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── SQL ───────────────────────────────────────────────────────────────────────
# Only ingest quality-flagged documents. ORDER BY lc.id for deterministic paging.
BATCH_SQL = text("""
    SELECT
        lc.id::text          AS chunk_id,
        lc.chunk_text        AS chunk_text,
        lc.chunk_index       AS chunk_index,
        lc.section_type      AS section_type,
        ld.id::text          AS document_id,
        ld.filename          AS filename,
        ld.title             AS title,
        ld.petitioner        AS petitioner,
        ld.respondent        AS respondent,
        ld.court             AS court,
        ld.state             AS state,
        ld.year              AS year,
        ld.citation          AS citation,
        ld.case_type         AS case_type
    FROM legal_chunks lc
    JOIN legal_documents ld ON lc.document_id = ld.id
    WHERE ld.quality_flag IN ('clean', 'encoding_fixed')
    ORDER BY lc.id
    LIMIT :limit OFFSET :offset
""")

COUNT_SQL = text("""
    SELECT COUNT(*)
    FROM legal_chunks lc
    JOIN legal_documents ld ON lc.document_id = ld.id
    WHERE ld.quality_flag IN ('clean', 'encoding_fixed')
""")


# ── Dense vector helpers ──────────────────────────────────────────────────────

def load_dense_index(emb_path: Path, ids_path: Path):
    """
    Memory-map embeddings.npy and build a reverse-lookup dict.
    mmap_mode='r' means the 3.57 GB file is NOT loaded into RAM —
    only the rows we actually access are paged in.

    Returns:
        embeddings  — np.memmap array (shape: N × 768)
        id_to_idx   — dict mapping chunk_id (str) → row index in embeddings
    """
    logger.info(f"Memory-mapping dense embeddings: {emb_path}")
    embeddings = np.load(str(emb_path), mmap_mode="r")
    logger.info(f"  Shape: {embeddings.shape} — dtype: {embeddings.dtype}")

    logger.info(f"Loading chunk ID list: {ids_path}")
    with open(str(ids_path), "r", encoding="utf-8") as f:
        chunk_ids_list: list[str] = json.load(f)

    if len(chunk_ids_list) != embeddings.shape[0]:
        raise ValueError(
            f"ID/embedding count mismatch: "
            f"{len(chunk_ids_list)} IDs vs {embeddings.shape[0]} embeddings"
        )

    id_to_idx: dict[str, int] = {cid: idx for idx, cid in enumerate(chunk_ids_list)}
    logger.info(f"  Built reverse map for {len(id_to_idx):,} chunk IDs.")
    return embeddings, id_to_idx


# ── Sparse vector helpers ─────────────────────────────────────────────────────

def load_bm25_model() -> SparseTextEmbedding:
    logger.info("Loading BM25 sparse model (fastembed) ...")
    model = SparseTextEmbedding(model_name="Qdrant/bm25")
    logger.info("  BM25 model ready.")
    return model


def generate_sparse_vectors(texts: list[str], bm25_model: SparseTextEmbedding) -> list[dict]:
    """
    Generate BM25 sparse vectors for a list of texts.
    Returns list of dicts with keys 'indices' and 'values'.
    """
    raw = list(bm25_model.embed(texts, batch_size=512))
    result = []
    for emb in raw:
        result.append({
            "indices": emb.indices.tolist(),
            "values":  emb.values.tolist(),
        })
    return result


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def get_already_ingested_ids(client: QdrantClient, chunk_ids: list[str]) -> set[str]:
    """
    Ask Qdrant which of these chunk_ids already exist as Point IDs.
    Since Qdrant Point ID === Postgres chunk_id, no translation is needed.
    Returns a set of already-ingested chunk_id strings.
    """
    try:
        existing = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=chunk_ids,
            with_payload=False,
            with_vectors=False,
        )
        return {str(p.id) for p in existing}
    except Exception as e:
        logger.error(f"Failed to check existing IDs from Qdrant: {e}")
        return set()   # if check fails, assume nothing is done → re-ingest (safe, idempotent)


def build_points(
    rows: list,
    dense_vectors: list,
    sparse_vectors: list[dict],
) -> list[PointStruct]:
    """
    Construct Qdrant PointStruct objects from parallel lists.
    Point ID === Postgres chunk_id (UUID string) — always, non-negotiable.
    chunk_text is NEVER included in the payload.
    """
    points = []
    for row, dense_vec, sparse_vec in zip(rows, dense_vectors, sparse_vectors):
        chunk_id = row["chunk_id"]   # already a str from the SQL cast

        # Guard: skip if sparse or dense vectors are empty/degenerate
        if dense_vec is None:
            logger.warning(f"Skipping chunk {chunk_id}: missing dense vector")
            continue
        if not sparse_vec["indices"]:
            logger.warning(f"Chunk {chunk_id}: empty sparse vector (blank text?) — using zero-index fallback")
            sparse_vec = {"indices": [0], "values": [0.0]}

        points.append(
            PointStruct(
                id=chunk_id,          # ← Qdrant Point ID = Postgres chunk_id
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"],
                    ),
                },
                payload={
                    # IDs
                    "document_id":  row["document_id"],
                    "chunk_id":     chunk_id,
                    # Chunk metadata
                    "chunk_index":  row["chunk_index"],
                    "section_type": row["section_type"],
                    # Document metadata
                    "filename":     row["filename"],
                    "title":        row["title"],
                    "petitioner":   row["petitioner"],
                    "respondent":   row["respondent"],
                    "court":        row["court"],
                    "state":        row["state"],
                    "year":         row["year"],
                    "citation":     row["citation"],
                    "case_type":    row["case_type"],
                    # chunk_text is intentionally OMITTED — lives only in Postgres
                },
            )
        )
    return points


def upsert_batch(client: QdrantClient, points: list[PointStruct]) -> int:
    """
    Upsert points to Qdrant in sub-batches to avoid HTTP payload limits.
    Returns number of successfully upserted points.
    """
    upserted = 0
    for i in range(0, len(points), UPSERT_SIZE):
        sub = points[i: i + UPSERT_SIZE]
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=sub,
                wait=True,
            )
            upserted += len(sub)
        except Exception as e:
            logger.error(f"Qdrant upsert failed for sub-batch at index {i}: {e}")
            # Log individual chunk IDs for investigation
            for p in sub:
                logger.error(f"  Failed point ID: {p.id}")
    return upserted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LexFind Hybrid Ingestor")
    parser.add_argument(
        "--embeddings", default=str(DEFAULT_EMB_PATH),
        help="Path to embeddings.npy (default: scripts/qdrant_ingestion/embeddings.npy)"
    )
    parser.add_argument(
        "--chunk-ids", default=str(DEFAULT_IDS_PATH),
        help="Path to chunk_ids.json (default: scripts/qdrant_ingestion/chunk_ids.json)"
    )
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    ids_path = Path(args.chunk_ids)

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    if not emb_path.exists():
        logger.error(f"embeddings.npy not found: {emb_path}")
        sys.exit(1)
    if not ids_path.exists():
        logger.error(f"chunk_ids.json not found: {ids_path}")
        sys.exit(1)
    if not DATABASE_URL:
        logger.error("DATABASE_URL is not set in .env")
        sys.exit(1)

    print("=" * 60)
    print("  LexFind — Hybrid Ingestor")
    print("=" * 60)

    # ── Load dense index ───────────────────────────────────────────────────────
    print("\n[1/5] Loading dense embeddings index ...")
    embeddings, id_to_idx = load_dense_index(emb_path, ids_path)

    # ── Load BM25 model ────────────────────────────────────────────────────────
    print("\n[2/5] Loading BM25 sparse model ...")
    bm25_model = load_bm25_model()

    # ── Connect to Qdrant ──────────────────────────────────────────────────────
    print(f"\n[3/5] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} ...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)

    try:
        coll_info = client.get_collection(COLLECTION_NAME)
        existing_count = coll_info.points_count
        logger.info(f"Collection '{COLLECTION_NAME}' has {existing_count:,} existing points.")
    except Exception as e:
        logger.error(f"Cannot access collection '{COLLECTION_NAME}': {e}")
        logger.error("Did you run recreate_collection.py first?")
        sys.exit(1)

    # ── Connect to Postgres and count total chunks ─────────────────────────────
    print(f"\n[4/5] Connecting to PostgreSQL ...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    with engine.connect() as conn:
        total_chunks = conn.execute(COUNT_SQL).scalar()

    logger.info(f"Total eligible chunks in Postgres: {total_chunks:,}")

    # ── Ingest ────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Starting hybrid ingestion ({total_chunks:,} chunks, batch={BATCH_SIZE}) ...")
    print(f"      Estimated time: 2-4 hours on CPU. Ctrl+C safe — fully resumable.\n")

    t_start        = time.time()
    offset         = 0
    total_ingested = 0
    total_skipped  = 0
    total_failed   = 0
    batch_num      = 0

    with tqdm(total=total_chunks, desc="Hybrid ingestion", unit="chunk") as pbar:
        while True:
            # ── 1. Fetch batch from Postgres ───────────────────────────────────
            try:
                with engine.connect() as conn:
                    rows = conn.execute(
                        BATCH_SQL, {"limit": BATCH_SIZE, "offset": offset}
                    ).mappings().fetchall()
            except Exception as e:
                logger.error(f"Postgres fetch failed at offset {offset}: {e}")
                # Advance offset to avoid infinite loop on a broken page
                offset += BATCH_SIZE
                pbar.update(BATCH_SIZE)
                total_failed += BATCH_SIZE
                continue

            if not rows:
                break   # All chunks processed

            batch_chunk_ids = [row["chunk_id"] for row in rows]

            # ── 2. Resume: skip already-ingested chunks ────────────────────────
            already_done = get_already_ingested_ids(client, batch_chunk_ids)
            new_rows = [row for row in rows if row["chunk_id"] not in already_done]

            if not new_rows:
                total_skipped += len(rows)
                offset += BATCH_SIZE
                pbar.update(len(rows))
                continue

            # ── 3. Look up dense vectors ───────────────────────────────────────
            dense_vectors = []
            missing_dense_ids = []

            for row in new_rows:
                cid = row["chunk_id"]
                if cid in id_to_idx:
                    dense_vectors.append(embeddings[id_to_idx[cid]])
                else:
                    missing_dense_ids.append(cid)
                    dense_vectors.append(None)

            if missing_dense_ids:
                logger.warning(
                    f"Offset {offset}: {len(missing_dense_ids)} chunks have no dense vector — "
                    f"will be skipped. First few: {missing_dense_ids[:3]}"
                )

            # Filter to only rows that have a valid dense vector
            valid_pairs = [
                (row, dv) for row, dv in zip(new_rows, dense_vectors) if dv is not None
            ]
            failed_count = len(new_rows) - len(valid_pairs)
            total_failed += failed_count

            if not valid_pairs:
                total_skipped += len(already_done)
                offset += BATCH_SIZE
                pbar.update(len(rows))
                continue

            valid_rows  = [r for r, _ in valid_pairs]
            valid_dense = [dv for _, dv in valid_pairs]
            valid_texts = [row["chunk_text"] or "" for row in valid_rows]

            # ── 4. Generate BM25 sparse vectors ───────────────────────────────
            try:
                sparse_vectors = generate_sparse_vectors(valid_texts, bm25_model)
            except Exception as e:
                logger.error(f"BM25 generation failed at offset {offset}: {e}")
                total_failed += len(valid_rows)
                offset += BATCH_SIZE
                pbar.update(len(rows))
                continue

            # ── 5. Build PointStructs ──────────────────────────────────────────
            points = build_points(valid_rows, valid_dense, sparse_vectors)

            # ── 6. Upsert to Qdrant ────────────────────────────────────────────
            upserted = upsert_batch(client, points)
            failed_upsert = len(points) - upserted
            total_failed  += failed_upsert

            total_ingested += upserted
            total_skipped  += len(already_done)
            batch_num      += 1
            offset         += BATCH_SIZE
            pbar.update(len(rows))

            # Progress log every 10 batches
            if batch_num % 10 == 0:
                elapsed = time.time() - t_start
                rate    = total_ingested / elapsed if elapsed > 0 else 0
                eta_s   = (total_chunks - offset) / rate if rate > 0 else 0
                logger.info(
                    f"Batch {batch_num} | offset={offset:,} | "
                    f"ingested={total_ingested:,} | skipped={total_skipped:,} | "
                    f"failed={total_failed:,} | {rate:.0f} chunks/s | ETA {eta_s/60:.0f}m"
                )

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed  = time.time() - t_start
    try:
        final_count = client.get_collection(COLLECTION_NAME).points_count
    except Exception:
        final_count = "?"

    print("\n" + "=" * 60)
    print("  LexFind — Hybrid Ingestion Complete")
    print("=" * 60)
    print(f"  Chunks ingested this run  : {total_ingested:,}")
    print(f"  Chunks skipped (resume)   : {total_skipped:,}")
    print(f"  Chunks failed             : {total_failed:,}")
    print(f"  Time taken                : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Total points in Qdrant    : {final_count:,}")
    print(f"  Error log                 : {LOG_FILE}")
    print("=" * 60)
    print("\n  Next step: python scripts/hybrid/hybrid_eval.py\n")


if __name__ == "__main__":
    main()
