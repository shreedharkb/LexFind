"""
LexFind — Hybrid Search Evaluation (Step 3 of 3)
===================================================
Evaluates three search modes side-by-side against the same 192 sampled PDFs
used in FAISS v3 and Qdrant v1 evaluations, enabling a clean historical comparison.

Search modes:
  1. Dense Only   — standard semantic vector search (baseline)
  2. Sparse Only  — BM25 keyword search
  3. Hybrid (RRF) — Reciprocal Rank Fusion of dense + sparse

Metrics: Hit@1, Hit@3, Hit@5, MRR, Avg Latency

Output:
  scripts/hybrid/hybrid_eval_report.json    ← machine-readable full results
  scripts/hybrid/hybrid_eval_summary.txt    ← human-readable comparison table

Run from backend/:
  python scripts/hybrid/hybrid_eval.py --full
  python scripts/hybrid/hybrid_eval.py --500
  python scripts/hybrid/hybrid_eval.py --full --500
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    MatchText,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_DIR      = BASE_DIR / "data" / "pdfs"
BUILD_REPORT = BASE_DIR / "scripts" / "faiss_ingestion" / "chunked_build_report.json"
OUT_DIR      = BASE_DIR / "scripts" / "hybrid"
OUT_JSON     = OUT_DIR / "hybrid_eval_report.json"
OUT_TXT      = OUT_DIR / "hybrid_eval_summary.txt"

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "legal_corpus"
DENSE_MODEL     = "sentence-transformers/all-mpnet-base-v2"
TOP_K           = 5
OVER_FETCH      = TOP_K * 10   # over-fetch to allow dedup by filename


# ── PDF helpers ───────────────────────────────────────────────────────────────

def load_sampled_files() -> set[str]:
    with open(BUILD_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)
    return set(report["sampled_files"])


def extract_pdf_info(pdf_path: Path) -> dict:
    result = {
        "filename":       pdf_path.name,
        "extractable":    False,
        "first_page_text": "",
        "title":          None,
        "court":          None,
        "year":           None,
    }
    try:
        doc = fitz.open(str(pdf_path))
        if len(doc) == 0:
            return result
        first_page = doc[0].get_text().strip()
        if len(first_page) < 100:
            return result

        result["extractable"]     = True
        result["first_page_text"] = first_page

        m = re.search(
            r"([A-Z][A-Za-z\s\.\,]+(?:Vs?\.?|VERSUS|vs\.?)[A-Za-z\s\.\,]+)",
            first_page, re.IGNORECASE
        )
        if m:
            result["title"] = m.group(0).strip()[:120]

        m = re.search(r"(Supreme Court|High Court of \w[\w\s]*)", first_page, re.IGNORECASE)
        if m:
            result["court"] = m.group(0).strip()

        m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", first_page)
        if m:
            result["year"] = m.group(0)

    except Exception as e:
        result["error"] = str(e)
    return result


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_by_filename(points, top_k: int) -> list[dict]:
    """
    Qdrant returns chunk-level hits. Collapse to document-level by filename.
    Keeps the highest-scoring chunk per document. Returns top_k unique docs.
    """
    seen: dict[str, dict] = {}
    for point in points:
        fn = point.payload.get("filename")
        if not fn:
            continue
        score = float(point.score)
        if fn not in seen or score > seen[fn]["score"]:
            seen[fn] = {"filename": fn, "score": round(score, 4)}

    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


# ── Search functions ──────────────────────────────────────────────────────────

def search_dense_only(
    query: str,
    client: QdrantClient,
    dense_model: SentenceTransformer,
    top_k: int = TOP_K,
    query_filter=None,
) -> list[dict]:
    """Standard semantic dense search (baseline — identical to Qdrant v1)."""
    vector = dense_model.encode([query], normalize_embeddings=True)[0].tolist()
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        using="dense",
        query_filter=query_filter,
        limit=OVER_FETCH,
        with_payload=True,
    )
    return deduplicate_by_filename(res.points, top_k)


def search_sparse_only(
    query: str,
    client: QdrantClient,
    bm25_model: SparseTextEmbedding,
    top_k: int = TOP_K,
    query_filter=None,
) -> list[dict]:
    """Pure BM25 keyword search."""
    sparse_emb = list(bm25_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vec,
        using="sparse",
        query_filter=query_filter,
        limit=OVER_FETCH,
        with_payload=True,
    )
    return deduplicate_by_filename(res.points, top_k)


def search_hybrid(
    query: str,
    client: QdrantClient,
    dense_model: SentenceTransformer,
    bm25_model: SparseTextEmbedding,
    top_k: int = TOP_K,
    query_filter=None,
) -> list[dict]:
    """Hybrid search: dense + sparse via Reciprocal Rank Fusion (RRF)."""
    dense_vec  = dense_model.encode([query], normalize_embeddings=True)[0].tolist()
    sparse_emb = list(bm25_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=OVER_FETCH,
                filter=query_filter,
            ),
            Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=OVER_FETCH,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=OVER_FETCH,
        with_payload=True,
    )
    return deduplicate_by_filename(res.points, top_k)


# ── Hit / MRR helpers ─────────────────────────────────────────────────────────

def check_hit(expected_filename: str, results: list[dict]) -> dict:
    filenames = [r["filename"] for r in results]
    hit_rank  = None
    for rank, fn in enumerate(filenames, 1):
        if fn == expected_filename:
            hit_rank = rank
            break

    rr = 1.0 / hit_rank if hit_rank else 0.0
    return {
        "hit@1":     hit_rank == 1,
        "hit@3":     hit_rank is not None and hit_rank <= 3,
        "hit@5":     hit_rank is not None and hit_rank <= 5,
        "hit_rank":  hit_rank,
        "rr":        rr,
    }


# ── Filter accuracy tests ─────────────────────────────────────────────────────

def run_filter_accuracy_tests(
    client: QdrantClient,
    dense_model: SentenceTransformer,
    bm25_model: SparseTextEmbedding,
) -> list[dict]:
    """
    Runs a small set of metadata filter tests to verify that payload filters
    still work correctly with hybrid search. Prints PASS/FAIL for each test.
    """
    test_query = "fundamental rights constitutional law India"

    filter_tests = [
        {
            "name":    "court = 'Supreme Court'",
            "filter":  Filter(must=[FieldCondition(key="court", match=MatchValue(value="Supreme Court"))]),
            "check":   lambda payload: payload.get("court") == "Supreme Court",
        },
        {
            "name":    "year in [2000, 2010]",
            "filter":  Filter(must=[
                FieldCondition(key="year", range={"gte": 2000, "lte": 2010})
            ]),
            "check":   lambda payload: 2000 <= (payload.get("year") or -1) <= 2010,
        },
        {
            "name":    "case_type = 'Writ Petition'",
            "filter":  Filter(must=[FieldCondition(key="case_type", match=MatchValue(value="Writ Petition"))]),
            "check":   lambda payload: payload.get("case_type") == "Writ Petition",
        },
    ]

    print("\n  Filter Accuracy Tests (Hybrid Search):")
    results = []
    for test in filter_tests:
        try:
            res = client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(
                        query=dense_model.encode([test_query], normalize_embeddings=True)[0].tolist(),
                        using="dense",
                        limit=20,
                        filter=test["filter"],
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=list(bm25_model.embed([test_query]))[0].indices.tolist(),
                            values=list(bm25_model.embed([test_query]))[0].values.tolist(),
                        ),
                        using="sparse",
                        limit=20,
                        filter=test["filter"],
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=10,
                with_payload=True,
            )

            points = res.points
            if not points:
                status  = "WARN (no results)"
                passed  = False
            else:
                violations = [p for p in points if not test["check"](p.payload)]
                passed  = len(violations) == 0
                status  = "PASS ✓" if passed else f"FAIL ✗ ({len(violations)}/{len(points)} violations)"

        except Exception as e:
            status = f"ERROR — {e}"
            passed = False

        print(f"    [{status:^12}] {test['name']}")
        results.append({"test": test["name"], "status": status, "passed": passed})

    return results


# ── Metrics helpers ───────────────────────────────────────────────────────────

def compute_metrics(
    results_log: list[dict],
    mode: str,          # "dense", "sparse", "hybrid"
    scope: str,         # "gen" or "app"
    lat_a: list[float],
    lat_b: list[float],
    lat_c: list[float],
) -> dict:
    total = len(results_log)
    if total == 0:
        return {}

    def hits(query, k):
        key = f"{scope}_{mode}_{query}"
        return sum(1 for r in results_log if key in r and r[key][k])

    def mrr(query):
        key = f"{scope}_{mode}_{query}"
        rrs = [r[key]["rr"] for r in results_log if key in r]
        return round(float(np.mean(rrs)), 3) if rrs else 0.0

    def pct(n):
        return round(n / total * 100, 1)

    return {
        "Query_A": {
            "hit@1": f"{hits('a','hit@1')}/{total} = {pct(hits('a','hit@1'))}%",
            "hit@3": f"{hits('a','hit@3')}/{total} = {pct(hits('a','hit@3'))}%",
            "hit@5": f"{hits('a','hit@5')}/{total} = {pct(hits('a','hit@5'))}%",
            "mrr":   mrr("a"),
            "avg_latency_ms": round(float(np.mean(lat_a)), 1) if lat_a else 0,
        },
        "Query_B": {
            "hit@1": f"{hits('b','hit@1')}/{total} = {pct(hits('b','hit@1'))}%",
            "hit@3": f"{hits('b','hit@3')}/{total} = {pct(hits('b','hit@3'))}%",
            "hit@5": f"{hits('b','hit@5')}/{total} = {pct(hits('b','hit@5'))}%",
            "mrr":   mrr("b"),
            "avg_latency_ms": round(float(np.mean(lat_b)), 1) if lat_b else 0,
        },
        "Query_C": {
            "hit@1": f"{hits('c','hit@1')}/{total} = {pct(hits('c','hit@1'))}%",
            "hit@3": f"{hits('c','hit@3')}/{total} = {pct(hits('c','hit@3'))}%",
            "hit@5": f"{hits('c','hit@5')}/{total} = {pct(hits('c','hit@5'))}%",
            "mrr":   mrr("c"),
            "avg_latency_ms": round(float(np.mean(lat_c)), 1) if lat_c else 0,
        },
    }


def fmt_table_row(
    results_log, scope, mode_a, mode_b, mode_c, query, metric
):
    """Extract a single metric value for three modes for one query — used in txt tables."""
    def extract(mode):
        key = f"{scope}_{mode}_{query}"
        rrs = [r[key]["rr"] for r in results_log if key in r]
        hits_count = sum(1 for r in results_log if key in r and r[key][metric])
        total = len(results_log)
        if metric == "mrr":
            return f"{float(np.mean(rrs)):.3f}" if rrs else "0.000"
        return f"{round(hits_count / total * 100, 1)}%" if total else "0.0%"

    return f"  {metric:<8}  {extract(mode_a):<12}  {extract(mode_b):<12}  {extract(mode_c)}"


def build_txt_section(results_log, scope, label):
    modes = ["dense", "sparse", "hybrid"]
    lines = [
        f"\n{'━'*61}",
        f"  {label}",
        f"{'━'*61}",
        f"  {'':20}  {'Dense Only':<12}  {'Sparse Only':<12}  Hybrid (RRF)",
    ]
    for q, q_label in [("a", "Query A — Excerpt"), ("b", "Query B — Case Title"), ("c", "Query C — Legal Topic")]:
        lines.append(f"\n  {q_label}")
        for metric in ["hit@1", "hit@3", "hit@5", "mrr"]:
            lines.append(fmt_table_row(results_log, scope, "dense", "sparse", "hybrid", q, metric))
        # latency
        for m, lbl in [("dense", "Dense Only"), ("sparse", "Sparse Only"), ("hybrid", "Hybrid (RRF)")]:
            pass
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LexFind Hybrid Search Evaluation")
    parser.add_argument("--full",  action="store_true", dest="full",   help="Evaluate on full 46k corpus")
    parser.add_argument("--500",   action="store_true", dest="apples", help="Evaluate restricted to 500 cases (apples-to-apples vs FAISS)")
    args = parser.parse_args()

    if not args.full and not args.apples:
        print("Please specify at least one mode: --full or --500")
        sys.exit(1)

    print("=" * 61)
    print("  LexFind — Hybrid Search Evaluation")
    print("=" * 61)

    # ── Connect ────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} ...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)

    try:
        info = client.get_collection(COLLECTION_NAME)
        print(f"      Collection '{COLLECTION_NAME}': {info.points_count:,} points ✓")
    except Exception as e:
        print(f"\n❌  Cannot connect to Qdrant collection: {e}")
        sys.exit(1)

    # ── Load models ────────────────────────────────────────────────────────────
    print(f"\n[2/4] Loading models ...")
    dense_model = SentenceTransformer(DENSE_MODEL)
    bm25_model  = SparseTextEmbedding(model_name="Qdrant/bm25")
    print("      Dense + BM25 models loaded ✓")

    # ── Load sampled files ─────────────────────────────────────────────────────
    print(f"\n[3/4] Loading sampled file list ...")
    sampled_files = load_sampled_files()
    eval_pdfs     = sorted([PDF_DIR / fn for fn in sampled_files if (PDF_DIR / fn).exists()])
    print(f"      Files in build report : {len(sampled_files)}")
    print(f"      Found on disk         : {len(eval_pdfs)}")

    # Pre-build the apples-to-apples MatchAny filter
    restrict_filter = Filter(
        must=[FieldCondition(key="filename", match=MatchAny(any=list(sampled_files)))]
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print(f"\n[4/4] Running evaluation (3 modes × 3 queries × {len(eval_pdfs)} docs) ...")
    print("      This will take a few minutes.\n")

    results_log = []
    skipped     = 0

    # Latency buckets: gen_dense_a, gen_sparse_a, gen_hybrid_a, app_*, etc.
    lat: dict[str, list[float]] = {
        f"{scope}_{mode}_{q}": []
        for scope in ("gen", "app")
        for mode  in ("dense", "sparse", "hybrid")
        for q     in ("a", "b", "c")
    }

    for i, pdf_path in enumerate(eval_pdfs):
        info = extract_pdf_info(pdf_path)
        if not info["extractable"]:
            skipped += 1
            continue

        filename   = info["filename"]
        first_page = info["first_page_text"]

        # Build queries
        query_a = first_page[:400].replace("\n", " ").strip()
        query_b = (
            f"What was the judgment in {info['title']}?"
            if info["title"]
            else f"Find legal case from {info.get('court', 'Supreme Court')} in {info.get('year', '2000')}"
        )
        sentences = [s.strip() for s in first_page.split(".") if len(s.strip()) > 40]
        query_c   = sentences[len(sentences) // 2] if sentences else query_a

        log_entry = {
            "filename": filename,
            "title":    info["title"],
            "query_a":  query_a[:80] + "...",
            "query_b":  query_b,
            "query_c":  query_c,
        }

        for scope, q_filter in [
            ("gen",  None           if args.full   else None),
            ("app",  restrict_filter if args.apples else None),
        ]:
            if scope == "gen"  and not args.full:   continue
            if scope == "app"  and not args.apples: continue

            for query_key, query_text in [("a", query_a), ("b", query_b), ("c", query_c)]:
                # Dense
                t0 = time.time()
                res_d = search_dense_only(query_text, client, dense_model, TOP_K, q_filter)
                lat[f"{scope}_dense_{query_key}"].append((time.time() - t0) * 1000)
                log_entry[f"{scope}_dense_{query_key}"] = check_hit(filename, res_d)

                # Sparse
                t0 = time.time()
                res_s = search_sparse_only(query_text, client, bm25_model, TOP_K, q_filter)
                lat[f"{scope}_sparse_{query_key}"].append((time.time() - t0) * 1000)
                log_entry[f"{scope}_sparse_{query_key}"] = check_hit(filename, res_s)

                # Hybrid
                t0 = time.time()
                res_h = search_hybrid(query_text, client, dense_model, bm25_model, TOP_K, q_filter)
                lat[f"{scope}_hybrid_{query_key}"].append((time.time() - t0) * 1000)
                log_entry[f"{scope}_hybrid_{query_key}"] = check_hit(filename, res_h)

        results_log.append(log_entry)

        if (i + 1) % 20 == 0:
            print(f"      {i+1}/{len(eval_pdfs)} done ...")

    print(f"      Skipped (unextractable): {skipped}")

    # ── Run filter accuracy tests ──────────────────────────────────────────────
    filter_results = run_filter_accuracy_tests(client, dense_model, bm25_model)

    # ── Build summary JSON ─────────────────────────────────────────────────────
    total   = len(results_log)
    summary = {"total_evaluated": total, "filter_accuracy": filter_results}

    for scope in (["gen"] if args.full else []) + (["app"] if args.apples else []):
        scope_label = "Full_46k" if scope == "gen" else "Apples_500"
        for mode in ["dense", "sparse", "hybrid"]:
            key = f"{scope_label}_{mode.capitalize()}"
            summary[key] = compute_metrics(
                results_log, mode, scope,
                lat[f"{scope}_{mode}_a"],
                lat[f"{scope}_{mode}_b"],
                lat[f"{scope}_{mode}_c"],
            )

    # ── Build human-readable txt ───────────────────────────────────────────────
    txt = f"""
╔═════════════════════════════════════════════════════════════╗
║         LexFind — Hybrid Search Evaluation Report         ║
╚═════════════════════════════════════════════════════════════╝

Total Documents Evaluated : {total}
"""

    def make_section(scope_key, label):
        lines = [
            f"\n{'━'*61}",
            f"  {label}",
            f"{'━'*61}",
            f"  {'':20}  {'Dense Only':<14}  {'Sparse Only':<14}  Hybrid (RRF)",
        ]
        for q_key, q_label in [("a", "Query A (Excerpt)"), ("b", "Query B (Title)"), ("c", "Query C (Topic)")]:
            lines.append(f"\n  ── {q_label} ──")
            for metric in ["hit@1", "hit@3", "hit@5", "mrr"]:
                def get_val(mode):
                    key = f"{scope_key}_{mode}_{q_key}"
                    rrs = [r[key]["rr"] for r in results_log if key in r]
                    hc  = sum(1 for r in results_log if key in r and r[key].get(metric, False))
                    t   = len(results_log)
                    if metric == "mrr":
                        return f"{float(np.mean(rrs)):.3f}" if rrs else "0.000"
                    return f"{round(hc / t * 100, 1)}%" if t else "0.0%"
                d = get_val("dense")
                s = get_val("sparse")
                h = get_val("hybrid")
                lines.append(f"  {metric:<8}  {d:<16}  {s:<16}  {h}")
            # latency
            for mode, lbl in [("dense", "Dense"), ("sparse", "Sparse"), ("hybrid", "Hybrid")]:
                lat_key  = f"{scope_key}_{mode}_{q_key}"
                avg_lat  = round(float(np.mean(lat[lat_key])), 1) if lat[lat_key] else 0
                lines.append(f"  Latency ({lbl:<7}) {avg_lat} ms")
        return "\n".join(lines)

    if args.full:
        txt += make_section("gen", "FULL 46k CORPUS — HEAD TO HEAD")

    if args.apples:
        txt += make_section("app", "APPLES-TO-APPLES (Restricted to 500 cases)")

    # Filter accuracy section
    txt += f"\n\n{'━'*61}\n  FILTER ACCURACY TESTS\n{'━'*61}\n"
    for fr in filter_results:
        txt += f"  [{fr['status']:^14}]  {fr['test']}\n"

    txt += f"\nFull report: hybrid_eval_report.json\n"

    # ── Save ───────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results_log}, f, indent=2, ensure_ascii=False)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt)

    print(txt)
    print(f"\nSaved:\n  {OUT_JSON}\n  {OUT_TXT}")


if __name__ == "__main__":
    main()
