# LangGraph Agent

The LangGraph agent (`lex_graph`) is the intelligence layer of LexFind. Every chat request flows through a compiled 14-node state machine (`app/agents/graph.py`) that classifies the request, rewrites the query, retrieves, grades the retrieval, retries once with a different strategy when needed, and only then generates an answer.

## State

`LexFindState` is a `TypedDict` defined in `app/agents/state.py`. A fresh instance is created at the start of every chat request and discarded after the SSE stream closes. There is no checkpointing or persistence at the LangGraph layer.

```python
class LexFindState(TypedDict):
    # Input fields
    session_id:       str
    user_id:          str
    question:         str
    history:          list[dict]    # last 6 messages loaded from DB
    explicit_mode:    Optional[str] # "auto" | "document" | "corpus"
    document_ids:     list[str]     # IDs of documents attached to this session

    # Classifier output
    is_legal:         bool
    intent:           str           # "general" | "document_chat" | "corpus_search"

    # Retrieval output
    retrieved_chunks: list[dict]
    citations:        list[dict]    # built from chunk payload, not LLM output

    # Final output
    answer:           str
    error:            Optional[str]
```

## Graph Topology

```
classifier
    ├── blocked            → END
    ├── general            → general_chat        → END
    ├── general_summary    → general_summary     → END
    ├── document_chat      → query_enhancement ─┐
    └── corpus_search      → query_enhancement ─┤
                                                 ↓
                                        qdrant_retrieval / document_retrieval
                                                 ↓
                                          retrieval_grader
                                    (pass)  /  (fail, 1st retry)  \  (exhausted)
                                           /                        \
                                    recency_check        corpus/document re-retrieval
                                      /      \                     ↓
                              web_augment   └──→ response_generation ← graceful_degradation
                                   ↓                  ↓
                              response_generation   END
```

1. **classifier** — one LLM call (gpt-oss-120b, temperature 0, JSON mode) labels the request `blocked`, `general`, `general_summary`, `document_chat` or `corpus_search`. If a document is explicitly attached, the LLM call is skipped and the request routes to document chat.
2. **blocked / general_chat / general_summary** — non-legal rejection, plain legal Q&A, and multi-message summarisation without retrieval.
3. **query_enhancement** — a rewriter (gpt-oss-20b) turns the message plus recent history into a standalone query and flags recent-law cues.
4. **qdrant_retrieval / document_retrieval** — hybrid dense+BM25 search over the shared corpus, or retrieval scoped to the attached document (Qdrant for corpus cases, pgvector for private uploads, after an ownership check).
5. **retrieval_grader** — a small model (gpt-oss-20b) judges whether the retrieved passages answer the question and returns a reason (`too_specific`, `wrong_domain`, `no_coverage`) when they do not. If the grader call fails, the results are accepted.
6. **corpus_re_retrieval / document_re_retrieval** — one retry with a different strategy: `too_specific` relaxes the query, `wrong_domain` searches sparse-only, anything else searches dense-only.
7. **graceful_degradation** — after a failed retry, the pipeline continues with a disclaimer in the prompt.
8. **recency_check / web_augment** — if recent-law cues were flagged and a Tavily key is configured, up to three web results are appended.
9. **response_generation** — builds the grounded prompt from the chunks, web results, optional disclaimer and the last messages.

Compiled in `app/agents/graph.py` as a module-level singleton `lex_graph = build_graph()`.

## LLM usage

| Request path | LLM calls (typical) |
|---|---|
| Blocked | 1 (classifier) |
| General question | 2 (classifier + answer) |
| Document chat | 3–5 |
| Corpus search | 4–6 |
| First message of a session | +1 (title generation) |

The final answer itself is streamed token-by-token by the API layer (Groq `stream=True`) after the graph finishes; citations are emitted before the `[DONE]` marker and are built entirely from retrieval payload metadata, never from LLM output.

## classifier

**File:** `app/agents/nodes/classifier.py`

**Purpose:** One LLM call that both guards the legal domain and classifies the request's intent.

**Fast-path override:** If `explicit_mode` is "document" or "corpus", the LLM call is skipped entirely and the intent is set directly.

**Routing (`route_after_classifier`):**
- `blocked` -> blocked
- `general` -> general_chat
- `general_summary` -> general_summary
- `document_chat` -> query_enhancement
- `corpus_search` -> query_enhancement

**Edge cases handled:**
- If intent is "document_chat" but `document_ids` is empty, it falls back to "corpus_search".
- If the Groq call fails, it defaults to "general" with the error logged.

**Intent classification rules:**
- `general`: pure legal knowledge question, no retrieval required. Example: "What is Article 21?"
- `general_summary`: summarise or condense the conversation so far.
- `document_chat`: question about attached documents. Only classified here if `document_ids` is non-empty.
- `corpus_search`: question that needs searching across cases. Example: "How has the Supreme Court ruled on right to privacy?"
- Non-legal topics classify as `blocked`.

## general_chat

**File:** `app/agents/nodes/general_chat.py`

**Purpose:** Builds the prompt for pure legal knowledge questions with no retrieval; the answer itself is generated and streamed by the API layer.

**Output:** `prompt_messages` for the response node. No citations or retrieved chunks.

## document_chat

**File:** `app/agents/nodes/document_chat.py`

**Purpose:** Retrieves chunks from attached session documents; the answer is generated later by the response node.

**Retrieval routing per document:**

| `source_type` | Vector store | Method |
|---|---|---|
| `legal_case` | Qdrant | Hybrid search filtered by corpus `document_id`, limit 8 |
| `uploaded` | pgvector | Cosine search filtered by `document_id`, limit 8 |

The node iterates over all `document_ids` in state, fetches chunks from each, and concatenates them into one context block. This allows multi-document RAG in a single session.

**Ownership re-check:** `legal_case` documents are public by design; uploaded documents are usable only by their owner, and an uploaded document with no owner is never accessible.

**Citation construction:** Built directly from the chunk payload/row without any LLM call:
- Qdrant chunks: `document_id`, `chunk_id`, `title`, `court`, `year`, `citation`, `source_type = legal_case`, `score`
- pgvector chunks: `document_id`, `chunk_id`, `title`, `page_number`, `filename`, `source_type = uploaded`, `score`

## corpus_search

**File:** `app/agents/nodes/corpus_search.py`

**Purpose:** Searches the full 46k-case Qdrant corpus and prepares the grounded context for the response node.

**Retrieval:**
1. Hybrid dense+BM25 query against Qdrant (`legal_corpus_hybrid`), limit 15.
2. Deduplicates results by `document_id`: keeps the highest-scoring chunk per document.
3. Takes the top 5 unique documents.

**Citation construction:** Built from the Qdrant payload with `source_type = legal_case`; chunk text is fetched from PostgreSQL.

## Other nodes

- **query_enhancement** (`app/agents/nodes/query_enhancement.py`) — rewrites the query into a standalone form (gpt-oss-20b).
- **retrieval_grader** (`app/agents/nodes/retrieval_grader.py`) — grades the retrieved passages (gpt-oss-20b); fails open on LLM errors.
- **corpus_re_retrieval / document_re_retrieval** — the single retry with an alternative search strategy.
- **recency_check / web_augment** — optional Tavily web search for recent-law questions.
- **graceful_degradation** — adds a disclaimer to the prompt after exhausted retries.
- **response_generation** (`app/agents/nodes/response_generation.py`) — builds the final grounded prompt for the streamed answer.
- **blocked** (inline in `app/agents/graph.py`) — fixed rejection message, no LLM call.

## Shared Utilities

**`_embedder.py`**
Delegates to the project-wide `app.services.embedding_service.embed_query`. The `SentenceTransformer` model is loaded once per process via the singleton in `embedding_service.py`.

**`_qdrant.py`**
Returns a module-level `QdrantClient` singleton. Connection parameters from `QDRANT_HOST` and `QDRANT_PORT` environment variables (defaults: `localhost:6333`).

## FastAPI Integration

The `send_message` endpoint in `app/api/sessions.py` runs the compiled graph in a thread pool, then streams the answer from Groq (`stream=True`) token by token over SSE:

1. token events `data: {"content": ...}` (or a single direct-answer event for blocked/error paths),
2. a `citations` event carrying the retrieved sources,
3. `data: [DONE]`,
4. the assistant message is persisted to PostgreSQL.

Because the graph nodes use synchronous Groq calls, the graph itself is executed with `await loop.run_in_executor(...)`; only the final answer is streamed. This means classification, retrieval, grading and any retry happen before the first answer token appears.
