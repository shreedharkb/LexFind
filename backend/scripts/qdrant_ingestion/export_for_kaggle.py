"""
LexFind — Export Chunks for Kaggle (Step 2a of Qdrant Ingestion Pipeline)
============================================================================
Reads all chunks from the legal_chunks table in PostgreSQL and exports them
to chunks_for_embedding.jsonl in batches of 100k rows.

Each line: {"chunk_id": "<uuid>", "chunk_text": "..."}

Run from backend/:
  python scripts/qdrant_ingestion/export_for_kaggle.py
  python scripts/qdrant_ingestion/export_for_kaggle.py --output-dir /tmp/kaggle_data
  python scripts/qdrant_ingestion/export_for_kaggle.py --batch-size 100000
"""

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

from sqlalchemy import create_engine, text
from tqdm import tqdm

DATABASE_URL  = os.getenv("DATABASE_URL", "")
DEFAULT_OUT   = BASE_DIR / "scripts" / "qdrant_ingestion" / "kaggle_export"
BATCH_SIZE    = 100_000


def export_chunks(output_dir: Path, batch_size: int):
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set in .env")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    with engine.connect() as conn:
        print("Starting export... (Skipped exact counting to start immediately)")

        # Fast keyset pagination (avoids slow OFFSET scanning)
        last_id = "00000000-0000-0000-0000-000000000000"
        file_index = 0
        exported_total = 0

        with tqdm(desc="Exporting chunks", unit="chunk") as pbar:
            while True:
                rows = conn.execute(
                    text("""
                        SELECT id::text, chunk_text
                        FROM legal_chunks
                        WHERE qdrant_point_id IS NULL 
                          AND id::text > :last_id
                        ORDER BY id::text
                        LIMIT :lim
                    """),
                    {"lim": batch_size, "last_id": last_id},
                ).fetchall()

                if not rows:
                    break

                out_file = output_dir / f"chunks_for_embedding_{file_index:03d}.jsonl"
                with open(out_file, "w", encoding="utf-8") as f:
                    for chunk_id, chunk_text in rows:
                        line = json.dumps(
                            {"chunk_id": chunk_id, "chunk_text": chunk_text},
                            ensure_ascii=False,
                        )
                        f.write(line + "\n")

                n = len(rows)
                exported_total += n
                pbar.update(n)

                last_id = rows[-1][0]  # Update last_id for next batch
                file_index += 1

                print(
                    f"\n  Wrote {n:,} chunks -> {out_file.name} "
                    f"({out_file.stat().st_size / 1024 / 1024:.1f} MB)"
                )

    print(f"\n[OK] Export complete!")
    print(f"  Total chunks exported : {exported_total:,}")
    print(f"  Output files          : {file_index}")
    print(f"  Output directory      : {output_dir}")
    print()
    print("Next step: Upload the .jsonl file(s) to Kaggle and run kaggle_embedder.ipynb")
    print()


def main():
    parser = argparse.ArgumentParser(description="Export legal_chunks to JSONL for Kaggle embedding")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUT),
        help="Directory to write .jsonl files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Rows per output file (default 100000)",
    )
    args = parser.parse_args()
    export_chunks(Path(args.output_dir), args.batch_size)


if __name__ == "__main__":
    main()
