# Retrieval Metrics

Retrieval quality is measured against the index the running system actually serves, so the
figures below describe the deployed configuration rather than a prototype.

| | |
|---|---|
| Collection | Qdrant `legal_corpus_hybrid` |
| Vectors | dense `all-mpnet-base-v2` (768-d) + sparse `Qdrant/bm25` |
| Corpus | 46,865 judgments (44,958 tagged Supreme Court), 1,162,730 chunks |
| Measured | 2026-11-05, on the machine that hosts the demo stack |
| Harness | [`backend/scripts/hybrid/hybrid_eval.py`](../backend/scripts/hybrid/hybrid_eval.py) |
| Raw evidence | [`eval_sample_192.json`](../backend/scripts/hybrid/eval_sample_192.json), [`hybrid_eval_report.json`](../backend/scripts/hybrid/hybrid_eval_report.json), [`latency_report.json`](../backend/scripts/hybrid/latency_report.json), [`hybrid_eval_summary.txt`](../backend/scripts/hybrid/hybrid_eval_summary.txt) |

## Sample and queries

192 judgments are drawn deterministically by
[`make_eval_sample.py`](../backend/scripts/hybrid/make_eval_sample.py): PostgreSQL orders all
candidate filenames by `md5(filename || '20261105')`, the first 900 are taken, and a candidate is
kept only if its PDF is present, its first page yields at least 100 characters, and a `X v. Y`
case title can be parsed from it. The resulting span is 1950--2024, all Supreme Court, and the
drawn list is committed.

Each judgment is queried three ways, always with text taken from the judgment it is meant to
find, so the benchmark measures *source-document recovery*:

| Query | Construction | What it simulates |
|---|---|---|
| A -- excerpt | first 400 characters of page 1 | pasting a passage to find its case |
| B -- title | `What was the judgment in <title>?` | looking a known case up by name |
| C -- sentence | the middle sentence of page 1 longer than 40 characters | typing one line of a fact pattern |

192 x 3 = 576 queries per configuration. Retrieval returns chunks, so the harness keeps the
best-ranked chunk per judgment, takes the top five distinct judgments, and records whether the
source judgment is among them (document-level Hit@k, plus MRR).

## Configurations compared

The same 576 queries are issued three ways against the same collection, chunks, over-fetch (50)
and cut-off (5). Only the query body sent to Qdrant differs, so any difference between the
columns belongs to the lexical list and the fusion step, not to the index, the chunking or the
sampling.

* **Dense** -- vector search on the `all-mpnet-base-v2` named vector.
* **Sparse** -- vector search on the `Qdrant/bm25` named sparse vector.
* **Hybrid RRF** -- both, fused inside Qdrant by Reciprocal Rank Fusion with the
  library's constant `k = 1` (neither `app/services/qdrant_search_service.py`
  nor the harness overrides it). Verified on the running collection rather than assumed:
  fusing two *identical* prefetch lists returns 1.000, 0.667, 0.500, 0.400, 0.333 for
  ranks 1-5, i.e. each list contributes `1 / (rank + 1)`. Qdrant's RRF therefore does not
  use the `k = 60` of the original paper, and no earlier text in this repository or in the
  report should be read as claiming it does.

## Results

Counts are out of 192 queries per cell; intervals are 95% Wilson.

### Query A -- excerpt (400 characters)

| Configuration | Hit@1 | Hit@3 | Hit@5 (95% CI) | MRR |
|---|---|---|---|---|
| Dense | 41 (21.4%) | 51 (26.6%) | 56 (29.2%) [23.2, 36.0] | 0.244 |
| Sparse | 135 (70.3%) | 142 (74.0%) | 146 (76.0%) [69.5, 81.5] | **0.724** |
| Hybrid RRF | 87 (45.3%) | 145 (75.5%) | **150 (78.1%)** [71.8, 83.4] | 0.604 |

### Query B -- case title

| Configuration | Hit@1 | Hit@3 | Hit@5 (95% CI) | MRR |
|---|---|---|---|---|
| Dense | 4 (2.1%) | 6 (3.1%) | 9 (4.7%) [2.5, 8.7] | 0.028 |
| Sparse | 40 (20.8%) | 47 (24.5%) | **51 (26.6%)** [20.8, 33.2] | **0.230** |
| Hybrid RRF | 22 (11.5%) | 46 (24.0%) | 49 (25.5%) [19.9, 32.1] | 0.179 |

### Query C -- single sentence

| Configuration | Hit@1 | Hit@3 | Hit@5 (95% CI) | MRR |
|---|---|---|---|---|
| Dense | 26 (13.5%) | 31 (16.1%) | 41 (21.4%) [16.1, 27.7] | 0.157 |
| Sparse | 124 (64.6%) | 137 (71.4%) | **143 (74.5%)** [67.9, 80.1] | **0.683** |
| Hybrid RRF | 80 (41.7%) | 133 (69.3%) | 139 (72.4%) [65.7, 78.2] | 0.556 |

### Mean over the three query types

| Configuration | Hit@5 (95% CI) | Hits | MRR |
|---|---|---|---|
| Dense | 18.4% [15.5, 21.8] | 106 / 576 | 0.143 |
| Sparse | **59.0%** [55.0, 63.0] | **340 / 576** | **0.546** |
| Hybrid RRF | 58.7% [54.6, 62.6] | 338 / 576 | 0.446 |

## Latency

[`eval_latency.py`](../backend/scripts/hybrid/eval_latency.py) times query encoding plus the
Qdrant round trip plus document de-duplication, client-side, and discards the first 12 queries
of each mode so model warm-up does not contaminate the distribution. 188 queries per mode per
type are timed.

| Mode | Query | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| Dense | excerpt | 109.9 | 102.8 | 155.6 | 451.0 |
| Dense | title | 40.7 | 40.0 | 54.1 | 93.0 |
| Dense | sentence | 58.7 | 52.2 | 103.4 | 296.4 |
| Sparse | excerpt | 10.1 | 10.0 | 12.9 | 17.3 |
| Sparse | title | 8.4 | 7.8 | 13.1 | 16.5 |
| Sparse | sentence | 12.8 | 12.8 | 16.0 | 23.5 |
| Hybrid | excerpt | 104.2 | 99.9 | 144.9 | 238.7 |
| Hybrid | title | 40.4 | 39.5 | 49.7 | 65.4 |
| Hybrid | sentence | 49.4 | 43.6 | 78.9 | 285.0 |

All values in ms. The dense and hybrid columns are the same number because both pay the sentence
encoder; the retrieval itself is not the cost.

## Filter accuracy

`hybrid_eval.py` asserts that every chunk returned under a metadata filter satisfies it. All
three assertions pass on the running index:

| Filter | Result |
|---|---|
| `court = 'Supreme Court'` | PASS |
| `year in [2000, 2010]` | PASS |
| `case_type = 'Writ Petition'` | PASS |

## What the numbers say

1. **Hybrid beats dense at every rank and every query type.** Mean Hit@5 18.4% -> 58.7%, a
   factor of 3.19 and 232 further questions answered inside the top five; the Wilson intervals
   do not overlap at any of the nine cells.
2. **The lexical list carries the gain.** Sparse-only gains 40.6 points of mean Hit@5 over
   dense-only, fusion gains 40.3 -- the improvement is BM25 finding party names, case numbers,
   citations and section numbers that a mean-pooled 768-dim vector averages away against the
   boilerplate thousands of judgments share.
3. **The gain grows as the query shortens.** 2.67x on a 400-character excerpt, 3.38x on one
   sentence, 5.43x on a case title -- which is the practical case, since users type a line, not
   a page.
4. **Fusion buys coverage, not accuracy.** At the depth the assistant works at (five candidates
   in the prompt) sparse-only and hybrid are statistically identical, but the sparse list ranks
   better at the head: excerpt Hit@1 70.3% vs 45.3%, MRR 0.546 vs 0.446. RRF is under-weighting
   the list that ranks best, which is where a cross-encoder re-ranker or a re-tuned fusion
   constant would pay.
5. **Title queries are limited by the extractor, not only by retrieval.** The `X v. Y` pattern
   yields a party-style title for 79 of the 192 sampled judgments and a fragment or a
   jurisdiction line (`civil`, `Pravin`, `IN THE SUPREME COURT OF INDIA ... I.A. No.`) for the
   other 113. Splitting Query B on that criterion:

   | Title extraction | n | Dense Hit@5 | Sparse Hit@5 | Hybrid Hit@5 |
   |---|---|---|---|---|
   | Party-style title parsed | 79 | 11.4% | 48.1% | **49.4%** |
   | Fragment / jurisdiction line | 113 | 0.0% | 11.5% | 8.8% |

   So the 25.5% title cell is a blend of a working case (49.4%) and a broken input (8.8%). The
   court-and-year fallback in `hybrid_eval.py` never fired, because the sample builder only
   accepts a judgment whose title parses.
6. **The encoder window truncates 7.2% of chunks.** `all-mpnet-base-v2` accepts 384 word-piece
   tokens; the chunker caps chunks at 280 words (`MAX_CHUNK_WORDS`), which is soft because a
   single long sentence becomes one oversized chunk. On an unbiased random sample of 725
   indexed chunks (`tablesample system (0.06)`), tokenised with the model's own tokenizer: mean
   267, median 309, p95 396, p99 484, max 1,029, and 52 of 725 (7.2%) exceed 384. Those chunks
   are embedded from their first ~290 words only; the sparse vector is not truncated. The
   `sentence_aware` fallback strategy is the worst group (median 332, 15% over).

## Reproducing

Requires a running index (the Docker stack, with the corpus ingested) and the PDF corpus in
`backend/data/pdfs`. From `backend/`:

```bash
python scripts/hybrid/make_eval_sample.py      # rewrites the sample (same seed -> same 192)
python scripts/hybrid/hybrid_eval.py           # accuracy + filter assertions -> hybrid_eval_report.json
python scripts/hybrid/eval_latency.py          # latency percentiles -> latency_report.json
python scripts/hybrid/analyze_eval_report.py   # recomputes every figure above from the hit log alone
python scripts/hybrid/chunk_token_probe.py     # chunk lengths against the 384-token encoder window
```

Every percentage above can also be recomputed from the committed
`hybrid_eval_report.json` without re-running anything.

## Predecessor: the FAISS prototype

The first implementation retrieved from a flat FAISS index over chunk embeddings
(`backend/scripts/faiss_ingestion/`, superseded by the Qdrant hybrid collection). It is kept in
the repository for the history of the design; it had no payload filtering, no keyword path and
no document-level de-duplication, and its raw run outputs are not part of the committed evidence
above, so no accuracy figures from that stage are quoted here. The measurements in this document
are all from the Qdrant hybrid index that serves requests today.
