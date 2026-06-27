# Retrieval-Augmented Generation (RAG) Metrics

We conducted rigorous empirical evaluations of multiple vector search strategies to ensure high retrieval accuracy across our production corpus of **46,456 indexed Indian Supreme Court judgments** (approx. 1.1 million vector chunks). 

The evaluation dataset consists of 192 distinct queries across three categories:
- **Query A (Excerpt):** Paragraphs extracted directly from the text.
- **Query B (Title):** Searching using the exact case title.
- **Query C (Topic):** Searching for abstract legal topics and precedents.

---

## 1. The Baseline: FAISS
Initially, the system used a flat FAISS index. While fast on small datasets, it lacked metadata payload filtering and hybrid capabilities.

| Metric | Query A (Excerpt) | Query B (Title) |
|--------|-------------------|-----------------|
| Hit@1  | 40.1%            | 5.7%            |
| Hit@3  | 46.9%            | 11.5%           |
| Hit@5  | 53.6%            | 15.1%           |
| Latency | ~287 ms          | ~101 ms         |

**Limitations Found:**
- **No metadata storage:** Returned raw chunk text instead of case metadata.
- **No pre-filtering:** Impossible to filter by court, year, or judge.
- **Pure Cosine Only:** No keyword fallback for exact text matches.

---

## 2. The Move to Qdrant (Dense Only)
We migrated to Qdrant (HNSW index) to gain payload filtering and better scalability. However, evaluating pure Dense embeddings (`all-mpnet-base-v2`) on the **full 46k corpus** revealed the limitations of purely semantic search.

| Metric | Query A (Excerpt) | Query B (Title) | Query C (Topic) |
|--------|-------------------|-----------------|-----------------|
| Hit@1  | 28.1%            | 2.6%            | 11.5%           |
| Hit@3  | 32.3%            | 6.8%            | 18.8%           |
| Hit@5  | 35.9%            | 8.9%            | 19.8%           |
| Latency| ~462 ms          | ~188 ms         | ~401 ms         |

*Note: Pure semantic search struggles significantly with exact keyword matching (like Case Titles), dropping accuracy when scaled from a small test batch to 46,000 cases.*

---

## 3. Production Architecture: Hybrid RRF Search
To solve the exact-match keyword problem inherent to pure dense vectors, we implemented **Hybrid Search with Reciprocal Rank Fusion (RRF)**. This executes both a Dense semantic query and a Sparse BM25 keyword query, fusing the ranked results.

**Full 46k Corpus Performance (Hybrid RRF):**

| Metric | Query A (Excerpt) | Query B (Title) | Query C (Topic) |
|--------|-------------------|-----------------|-----------------|
| Hit@1  | 48.4%            | 12.0%           | 43.2%           |
| Hit@3  | 79.2%            | 21.9%           | 71.4%           |
| **Hit@5**  | **82.3%**            | **26.0%**           | **74.5%**           |
| Latency| ~341 ms          | ~149 ms         | ~181 ms         |

### Conclusion & System Limitations

By moving to **Hybrid RRF**, we saw massive improvements across all query types compared to Dense alone:
- **Excerpt queries** Hit@5 jumped from 35.9% to **82.3%**
- **Title queries** Hit@5 jumped from 8.9% to **26.0%**
- **Topic queries** Hit@5 jumped from 19.8% to **74.5%**

**Current Limitations:**
- **Short Title Queries:** Searching strictly by case titles without filters still has relatively low top-5 accuracy (26.0%). Because the corpus is massive, case titles share many common legal terms (e.g., "State of Maharashtra vs..."). Users searching for specific case names should utilize metadata filters (year, court) in the UI to narrow the semantic space.
- **Sparse Bias:** Sparse BM25 sometimes over-indexes on exact term matches. While RRF balances this, highly semantic matches can occasionally be pushed down if they lack the exact keywords.
