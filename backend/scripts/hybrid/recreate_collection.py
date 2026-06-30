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
COLLECTION_NAME = "legal_corpus"
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

def confirm_deletion(collection_name: str) -> bool:
    """Prompt user to confirm destructive deletion."""
    print(f"\n⚠️  WARNING: This will permanently delete '{collection_name}' from Qdrant.")
    print("    All 1.1M+ vectors will be erased. This cannot be undone.")
    answer = input("\n    Are you sure? [y/N]: ").strip().lower()
    return answer == "y"


def print_collection_info(client: QdrantClient, collection_name: str):
    """Print a summary of the newly created collection."""
    try:
        info = client.get_collection(collection_name)
        print(f"\n{'='*60}")
        print(f"  Collection '{collection_name}' — VERIFIED")
        print(f"{'='*60}")
        print(f"  Status         : {info.status}")
        print(f"  Points count   : {info.points_count:,}")

        cfg = info.config
        if cfg and cfg.params:
            vp = cfg.params.vectors
            sp = cfg.params.sparse_vectors
            if isinstance(vp, dict):
                for name, v in vp.items():
                    print(f"  Dense vector   : '{name}' — {v.size}d {v.distance.value}")
            if sp:
                for name in sp:
                    print(f"  Sparse vector  : '{name}' — BM25")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"  Could not fetch collection info: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LexFind — Hybrid Collection Setup")
    print("=" * 60)

    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"\n[1/3] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} ...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)

    try:
        client.get_collections()
    except Exception as e:
        print(f"\n❌  Cannot connect to Qdrant: {e}")
        sys.exit(1)

    print("      Connected ✓")

    # ── Delete existing collection ─────────────────────────────────────────────
    print(f"\n[2/3] Checking for existing collection '{COLLECTION_NAME}' ...")

    if client.collection_exists(COLLECTION_NAME):
        existing_info = client.get_collection(COLLECTION_NAME)
        print(f"      Found: {existing_info.points_count:,} points")

        if not confirm_deletion(COLLECTION_NAME):
            print("\n  Aborted. Collection unchanged.")
            sys.exit(0)

        print(f"\n  Deleting '{COLLECTION_NAME}' ...")
        client.delete_collection(COLLECTION_NAME)

        # Verify deletion
        if client.collection_exists(COLLECTION_NAME):
            print(f"\n❌  Deletion failed — '{COLLECTION_NAME}' still exists.")
            sys.exit(1)
        print(f"  Deleted ✓")
    else:
        print(f"      Collection does not exist — will create fresh.")

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
        print(f"\n❌  Collection creation failed: {e}")
        sys.exit(1)

    print(f"  Collection created ✓")

    # ── Create payload indexes ─────────────────────────────────────────────────
    print(f"\n  Creating payload indexes ...")
    failed_indexes = []
    for field, schema in PAYLOAD_INDEXES:
        try:
            client.create_payload_index(COLLECTION_NAME, field, schema)
            print(f"    ✓ {field} ({schema.value})")
        except Exception as e:
            print(f"    ✗ {field} — {e}")
            failed_indexes.append(field)

    if failed_indexes:
        print(f"\n⚠️  Some indexes failed to create: {failed_indexes}")
        print("    You can create them manually later. Continuing ...")

    # ── Verify ────────────────────────────────────────────────────────────────
    print_collection_info(client, COLLECTION_NAME)
    print("  Ready for hybrid ingestion.")
    print("  Next step: python scripts/hybrid/hybrid_ingestor.py\n")


if __name__ == "__main__":
    main()
