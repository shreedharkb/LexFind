"""
LexFind — Hybrid Collection Setup (Step 1 of 3)
==================================================
Deletes the existing "legal_corpus" collection (single dense vector) and
recreates it with TWO named vector slots:

    "dense"  → 768-dim COSINE float vector  (sentence-transformers/all-mpnet-base-v2)
    "sparse" → BM25 sparse vector           (fastembed BM25)

All payload indexes are also recreated.

Run from backend/:
    python scripts/hybrid/recreate_collection.py
"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL      = os.getenv("QDRANT_URL", "").strip().rstrip("/")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "").strip() or None
COLLECTION_NAME = "legal_corpus_hybrid"
DENSE_DIM       = 768

PAYLOAD_INDEXES = [
    ("court",        PayloadSchemaType.KEYWORD),
    ("year",         PayloadSchemaType.INTEGER),
    ("state",        PayloadSchemaType.KEYWORD),
    ("case_type",    PayloadSchemaType.KEYWORD),
    ("section_type", PayloadSchemaType.KEYWORD),
    ("filename",     PayloadSchemaType.KEYWORD),
    ("document_id",  PayloadSchemaType.KEYWORD),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def confirm_deletion(collection_name: str, assume_yes: bool) -> bool:
    """Require an explicit command-line opt-in before deleting a collection."""
    if assume_yes:
        return True
    print(
        f"\nRefusing to delete '{collection_name}' without --yes. "
        "No data has been changed."
    )
    return False


def print_collection_info(client: QdrantClient, collection_name: str):
    """Print a summary of the newly created collection."""
    try:
        info = client.get_collection(collection_name)
        print(f"\n{'='*60}")
        print(f"  Collection '{collection_name}' - VERIFIED")
        print(f"{'='*60}")
        print(f"  Status         : {info.status}")
        print(f"  Points count   : {info.points_count:,}")

        cfg = info.config
        if cfg and cfg.params:
            vp = cfg.params.vectors
            sp = cfg.params.sparse_vectors
            if isinstance(vp, dict):
                for name, v in vp.items():
                    print(f"  Dense vector   : '{name}' - {v.size}d {v.distance.value}")
            if sp:
                for name in sp:
                    print(f"  Sparse vector  : '{name}' - BM25")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"  Could not fetch collection info: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create the LexFind hybrid collection. Existing data is never deleted by default."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="allow deletion and recreation of an existing collection",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  LexFind — Hybrid Collection Setup")
    print("=" * 60)

    # ── Connect ───────────────────────────────────────────────────────────────
    location = QDRANT_URL or f"{QDRANT_HOST}:{QDRANT_PORT}"
    print(f"\n[1/3] Connecting to Qdrant at {location} ...")
    if QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    else:
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            timeout=120,
        )

    try:
        client.get_collections()
    except Exception as e:
        print(f"\nERROR: Cannot connect to Qdrant: {e}")
        sys.exit(1)

    print("      Connected")

    # ── Delete existing collection ─────────────────────────────────────────────
    print(f"\n[2/3] Checking for existing collection '{COLLECTION_NAME}' ...")

    if client.collection_exists(COLLECTION_NAME):
        existing_info = client.get_collection(COLLECTION_NAME)
        print(f"      Found: {existing_info.points_count:,} points")

        if not confirm_deletion(COLLECTION_NAME, args.yes):
            print("\n  Aborted. Collection unchanged.")
            sys.exit(0)

        print(f"\n  Deleting '{COLLECTION_NAME}' ...")
        client.delete_collection(COLLECTION_NAME)

        # Verify deletion
        if client.collection_exists(COLLECTION_NAME):
            print(f"\nERROR: Deletion failed - '{COLLECTION_NAME}' still exists.")
            sys.exit(1)
        print("  Deleted")
    else:
        print("      Collection does not exist - will create fresh.")

    # ── Create collection with named vectors ──────────────────────────────────
    print(f"\n[3/3] Creating '{COLLECTION_NAME}' with dense + sparse vectors ...")

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False  # keep sparse index in RAM for fast search
                    )
                )
            },
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20_000,  # start HNSW indexing after 20k vectors
            ),
        )
    except Exception as e:
        print(f"\nERROR: Collection creation failed: {e}")
        sys.exit(1)

    print("  Collection created")

    # ── Create payload indexes ─────────────────────────────────────────────────
    print(f"\n  Creating payload indexes ...")
    failed_indexes = []
    for field, schema in PAYLOAD_INDEXES:
        try:
            client.create_payload_index(COLLECTION_NAME, field, schema)
            print(f"    OK {field} ({schema.value})")
        except Exception as e:
            print(f"    FAILED {field} - {e}")
            failed_indexes.append(field)

    if failed_indexes:
        print(f"\nWARNING: Some indexes failed to create: {failed_indexes}")
        print("    You can create them manually later. Continuing ...")

    # ── Verify ────────────────────────────────────────────────────────────────
    print_collection_info(client, COLLECTION_NAME)
    print("  Ready for hybrid ingestion.")
    print("  Next step: python scripts/hybrid/hybrid_ingestor.py\n")


if __name__ == "__main__":
    main()
