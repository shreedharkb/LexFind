"""
LexFind — Chunk Length vs the Encoder Window
=============================================
``all-mpnet-base-v2`` accepts 384 word-piece tokens. The chunker caps chunks at
``MAX_CHUNK_WORDS`` words, but that cap is soft: a single long sentence becomes one
oversized chunk. This script measures how many indexed chunks are actually longer
than the encoder window, which is the part of the dense path that is silently
dropped. Reads a random sample straight from PostgreSQL with ``tablesample``.

Run from ``backend/`` where the database is reachable and .env is set:

    python scripts/hybrid/chunk_token_probe.py
    python scripts/hybrid/chunk_token_probe.py --sample 0.06 --model sentence-transformers/all-mpnet-base-v2
"""

import argparse
import os
import statistics
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

from sqlalchemy import create_engine, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=0.06, help="tablesample percentage")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--limit", type=int, default=384, help="encoder token window")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    url = os.environ["DATABASE_URL"].replace("postgresql+psycopg2", "postgresql+psycopg")
    engine = create_engine(url)

    # Whitespace is collapsed so one chunk is one row regardless of PDF line breaks;
    # without this the sample loses exactly the long, newline-rich chunks we care about.
    sql = text(
        "select replace(replace(replace(chunk_text, chr(10), ' '), chr(9), ' '), chr(13), ' ') "
        "as chunk_text, chunk_strategy "
        f"from legal_chunks tablesample system ({args.sample})"
    )
    with engine.connect() as c:
        rows = [(r[0], r[1]) for r in c.execute(sql) if r[0] and r[0].strip()]

    toks = [len(tok(t, add_special_tokens=True).input_ids) for t, _ in rows]
    words = [len(t.split()) for t, _ in rows]
    over = sum(1 for k in toks if k > args.limit)
    n = len(toks)
    ordered = sorted(toks)

    print(f"chunks sampled        : {n}")
    print(f"mean words            : {statistics.mean(words):.1f}")
    print(f"mean tokens           : {statistics.mean(toks):.1f}  (ratio "
          f"{statistics.mean(toks) / statistics.mean(words):.3f} tokens per word)")
    for q in (50, 75, 90, 95, 99):
        print(f"p{q} tokens             : {ordered[min(n - 1, int(n * q / 100))]}")
    print(f"max tokens            : {max(toks)}")
    print(f"over {args.limit} tokens    : {over} / {n} = {100 * over / n:.1f}%")
    by: dict[str, list[int]] = {}
    for (_, strategy), k in zip(rows, toks):
        by.setdefault(strategy or "?", []).append(k)
    for strategy, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        o = sum(1 for x in v if x > args.limit)
        print(f"  {strategy:<20} n={len(v):<5} median={statistics.median(v):<5.0f} "
              f"over={100 * o / len(v):.1f}%")


if __name__ == "__main__":
    main()
