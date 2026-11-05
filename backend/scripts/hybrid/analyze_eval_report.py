"""
LexFind — Recompute the reported metrics from the committed hit log
====================================================================
Every accuracy figure in docs/rag_metrics.md and in the report comes out of this
script, from ``hybrid_eval_report.json`` alone, so no run is needed to audit them.
Prints Hit@1/3/5 counts, percentages and 95% Wilson intervals per query type and
mode, the mean over the three types, and the case-title split by whether the
``X v. Y`` extractor produced a party-style title or a fragment.

    python scripts/hybrid/analyze_eval_report.py
    python scripts/hybrid/analyze_eval_report.py --report path/to/hybrid_eval_report.json
"""

import argparse
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = {"dense": "gen_dense_{q}", "sparse": "gen_sparse_{q}", "hybrid": "gen_hybrid_{q}"}
TITLE_RE = re.compile(r"\bVs\b|\bv\.\s*$|\bV/s\b|\bversus\b", re.I)


def wilson(hits: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (centre - half), 100 * (centre + half))


def report(rows: list[dict], label: str) -> None:
    n = len(rows)
    print(f"\n=== {label}  (n={n}) ===")
    for q in ("a", "b", "c"):
        for mode, tmpl in MODES.items():
            key = tmpl.format(q=q)
            line = f"  query {q.upper()} {mode:<7}"
            for rank in ("hit@1", "hit@3", "hit@5"):
                h = sum(1 for r in rows if r.get(key, {}).get(rank))
                line += f"  {rank}={h} ({100 * h / n:.1f}%)"
            h5 = sum(1 for r in rows if r.get(key, {}).get("hit@5"))
            lo, hi = wilson(h5, n)
            mrr = sum(r.get(key, {}).get("rr", 0.0) for r in rows) / n
            print(f"{line}  CI=[{lo:.1f}, {hi:.1f}]  MRR={mrr:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(HERE / "hybrid_eval_report.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = data["results"]
    report(rows, "all sampled judgments")

    clean = [r for r in rows if TITLE_RE.search(r.get("title") or "")]
    noisy = [r for r in rows if not TITLE_RE.search(r.get("title") or "")]
    print("\n=== query B split by title extraction quality ===")
    for label, subset in (("party-style title", clean), ("fragment / jurisdiction line", noisy)):
        for mode, tmpl in MODES.items():
            key = tmpl.format(q="b")
            h = sum(1 for r in subset if r.get(key, {}).get("hit@5"))
            print(f"  {label:<30} {mode:<7} hit@5 = {h}/{len(subset)} = {100 * h / len(subset):.1f}%")

    print("\n=== mean Hit@5 over the three query types (of 3n) ===")
    for mode, tmpl in MODES.items():
        h = sum(1 for r in rows for q in ("a", "b", "c") if r.get(tmpl.format(q=q), {}).get("hit@5"))
        lo, hi = wilson(h, 3 * len(rows))
        print(f"  {mode:<7} {h}/{3 * len(rows)} = {100 * h / (3 * len(rows)):.1f}%  CI=[{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    main()
