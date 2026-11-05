"""
LexFind — Retrieval Latency Measurement
=========================================
Measures per-query latency for the three search modes on the same sample the
accuracy harness uses, so the reported figures are distributional rather than a
single mean contaminated by model warm-up.

Per-query time covers query encoding plus the Qdrant round trip plus document
de-duplication, measured client-side — the same basis as ``hybrid_eval.py``, with
the first ``WARMUP`` queries of each mode discarded.

Run from ``backend/`` (inside the API container):

    python scripts/hybrid/eval_latency.py
    python scripts/hybrid/eval_latency.py --docs 192 --repeats 1
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

import os
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from hybrid_eval import (
    COLLECTION_NAME,
    PDF_DIR,
    TOP_K,
    extract_pdf_info,
    search_dense_only,
    search_hybrid,
    search_sparse_only,
)

SAMPLE_FILE = BASE_DIR / "scripts" / "hybrid" / "eval_sample_192.json"
OUT_FILE    = BASE_DIR / "scripts" / "hybrid" / "latency_report.json"
WARMUP      = 12


def build_queries(sample_size: int):
    judgments = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))["judgments"][:sample_size]
    rows = []
    for j in judgments:
        info = extract_pdf_info(PDF_DIR / j["filename"])
        if not info["extractable"]:
            continue
        page = info["first_page_text"]
        sentences = [s.strip() for s in page.split(".") if len(s.strip()) > 40]
        rows.append({
            "a": page[:400].replace("\n", " ").strip(),
            "b": f"What was the judgment in {info['title']}?",
            "c": sentences[len(sentences) // 2] if sentences else page[:400],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="LexFind latency measurement")
    parser.add_argument("--docs", type=int, default=192, help="judgments to time")
    args = parser.parse_args()

    location = os.getenv("QDRANT_URL", "").strip().rstrip("/")
    if location:
        client = QdrantClient(url=location, api_key=os.getenv("QDRANT_API_KEY"), timeout=120)
    else:
        client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"),
                              port=int(os.getenv("QDRANT_PORT", "6333")), timeout=120)

    print(f"Collection {COLLECTION_NAME}: "
          f"{client.get_collection(COLLECTION_NAME).points_count:,} points")

    dense = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    bm25  = SparseTextEmbedding(model_name="Qdrant/bm25")
    rows  = build_queries(args.docs)
    print(f"Timed over {len(rows)} judgments x 3 query types")

    modes = {
        "dense":  lambda q: search_dense_only(q, client, dense, TOP_K),
        "sparse": lambda q: search_sparse_only(q, client, bm25, TOP_K),
        "hybrid": lambda q: search_hybrid(q, client, dense, bm25, TOP_K),
    }

    report: dict[str, dict[str, dict[str, float]]] = {}
    for mode, fn in modes.items():
        samples_ms = {key: [] for key in ("a", "b", "c")}
        timed = 0
        for row in rows:
            for key in ("a", "b", "c"):
                t0 = time.perf_counter()
                fn(row[key])
                elapsed = (time.perf_counter() - t0) * 1000
                if timed >= WARMUP:
                    samples_ms[key].append(elapsed)
                else:
                    timed += 1
        report[mode] = {
            key: {
                "n":     len(vals),
                "mean":  round(statistics.fmean(vals), 1),
                "p50":   round(float(np.percentile(vals, 50)), 1),
                "p95":   round(float(np.percentile(vals, 95)), 1),
                "max":   round(max(vals), 1),
            }
            for key, vals in samples_ms.items()
        }
        print(f"  {mode:<7} " + "  ".join(
            f"{key}: p50 {report[mode][key]['p50']:>6} ms  p95 {report[mode][key]['p95']:>7} ms"
            for key in ("a", "b", "c")))

    OUT_FILE.write_text(json.dumps({
        "basis": "client-side encode + Qdrant round trip + de-duplication",
        "warmup_queries_dropped_per_mode": WARMUP,
        "judgments": len(rows),
        "collection": COLLECTION_NAME,
        "points": client.get_collection(COLLECTION_NAME).points_count,
        "results": report,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
