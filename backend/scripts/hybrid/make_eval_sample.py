"""
LexFind — Evaluation Sample Builder
====================================
Draws the judgment sample that ``hybrid_eval.py`` evaluates, and writes the
``chunked_build_report.json`` file that the harness expects as its input.

Deterministic: candidate filenames are ordered by ``md5(filename || SEED)`` in
PostgreSQL, so the same seed always reproduces the same sample. A candidate is
kept only if (a) its PDF is present in ``backend/data/pdfs``, (b) its first page
yields at least 100 characters, and (c) a ``X v. Y`` case title can be parsed
from that page — condition (c) keeps Query B a genuine title query instead of
the court-and-year fallback.

Run from ``backend/`` (inside the API container, where the corpus and the
database are reachable):

    python scripts/hybrid/make_eval_sample.py
    python scripts/hybrid/make_eval_sample.py --seed 20261105 --size 192
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

import os
from sqlalchemy import create_engine, text

from hybrid_eval import PDF_DIR, extract_pdf_info

BUILD_REPORT = BASE_DIR / "scripts" / "faiss_ingestion" / "chunked_build_report.json"
SAMPLE_OUT   = BASE_DIR / "scripts" / "hybrid" / "eval_sample_192.json"

POOL_SIZE = 900


def candidates(engine, seed: str):
    sql = text(
        "SELECT filename, court, year FROM legal_documents "
        "WHERE filename IS NOT NULL ORDER BY md5(filename || :seed) LIMIT :pool"
    )
    with engine.connect() as conn:
        return conn.execute(sql, {"seed": seed, "pool": POOL_SIZE}).all()


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the LexFind evaluation sample")
    parser.add_argument("--seed", default="20261105", help="salt making the draw reproducible")
    parser.add_argument("--size", type=int, default=192, help="number of judgments to draw")
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        sys.exit("DATABASE_URL is not set")
    engine = create_engine(db_url)

    pool = candidates(engine, args.seed)
    print(f"Candidates from PostgreSQL : {len(pool)}")
    print(f"Corpus directory           : {PDF_DIR}")

    kept, skipped = [], {"no_pdf": 0, "no_text": 0, "no_title": 0}
    for filename, court, year in pool:
        path = PDF_DIR / filename
        if not path.exists():
            skipped["no_pdf"] += 1
            continue
        info = extract_pdf_info(path)
        if not info["extractable"]:
            skipped["no_text"] += 1
            continue
        if not info["title"]:
            skipped["no_title"] += 1
            continue
        kept.append({
            "filename": filename,
            "title":    info["title"],
            "court":    info.get("court") or court,
            "year":     info.get("year") or year,
        })
        if len(kept) >= args.size:
            break

    if len(kept) < args.size:
        sys.exit(f"Only {len(kept)} of {args.size} requested candidates were usable")

    payload = {
        "seed":           args.seed,
        "size":           len(kept),
        "pool_size":      POOL_SIZE,
        "draw":           "ORDER BY md5(filename || seed) from legal_documents",
        "filters":        "PDF present, first page >= 100 chars, X v. Y title parseable",
        "skipped":        skipped,
        "judgments":      kept,
    }

    SAMPLE_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    BUILD_REPORT.write_text(
        json.dumps(
            {
                "sampled_files": [j["filename"] for j in kept],
                "_note": "Written by scripts/hybrid/make_eval_sample.py "
                         f"(seed={args.seed}); the FAISS-era ingestion no longer produces this file.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Sample kept                : {len(kept)}")
    print(f"Skipped                    : {skipped}")
    print(f"Wrote {SAMPLE_OUT}")
    print(f"Wrote {BUILD_REPORT}")


if __name__ == "__main__":
    main()
