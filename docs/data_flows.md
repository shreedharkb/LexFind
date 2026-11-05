# End-to-End Data Flows

This document traces what happens internally for every major user action, from the browser click to the database write.

---

## Flow 1: Global Semantic Search

User types a legal query in the Search page and clicks Search.

1. React calls `POST /api/cases/search` with `{ query, filters }`.
2. The `search.py` router embeds the query into both a Dense vector (via `SentenceTransformer` `all-mpnet-base-v2`, 768-dim) and a Sparse BM25 vector.
3. Both vectors are sent to Qdrant (`legal_corpus_hybrid` collection) using a **Hybrid RRF (Reciprocal Rank Fusion)** query. Metadata filters (court, year, etc.) are applied at the Qdrant layer.
4. Qdrant fuses the semantic and keyword matches and returns raw chunk hits with scores and payload.
5. The service groups hits by `document_id` and queries the `legal_documents` table (PostgreSQL) to hydrate full case metadata (title, court, year, citation).
6. A `SearchResponse` JSON is returned and React renders ranked case cards.

---

## Flow 2: Analyze a Corpus Case

User clicks "Analyze in Assistant" on a search result card.

1. React calls `POST /api/cases/{case_id}/analyze`.
2. The backend checks if a `Document` record already exists for this corpus case.
   - If it does not exist: creates a `Document` with `source_type="legal_case"`, `status="ready"`, and the absolute PDF path. No Celery task is dispatched — the corpus case is already in Qdrant.
   - If it already exists and is `ready`: proceeds directly.
3. Backend creates a new `AssistantSession` and a `SessionDocument` link.
4. Returns `{ session_id, document_id }`.
5. React navigates to `/chat/{session_id}`.

Corpus PDFs are served via `GET /api/search/pdf/{id}` (accepts the case UUID or filename, no auth) directly from local storage or Azure Blob — no re-processing.

---

## Flow 3: Uploading a Private PDF

User clicks the paperclip icon in the chat interface, selects a PDF, and sends.

1. React calls `POST /api/documents/upload` with `multipart/form-data`.
2. Backend computes a SHA-256 hash of the file bytes and checks for duplicates in the `documents` table. If a duplicate exists, returns the existing document ID without re-processing.
3. `BlobStorageService` saves the bytes to Azure Blob Storage (or local disk if Azure is not configured).
4. A `Document` record is created in PostgreSQL: `source_type="uploaded"`, `status="uploaded"`, `owner_id=current_user.id`.
5. `process_document_task.delay(document_id, blob_path)` is dispatched to RabbitMQ.
6. The API returns `{ document_id, status: "uploaded" }`.
7. React calls `POST /api/sessions/{session_id}/documents` to attach the document to the active session.
8. The Celery worker executes asynchronously:
   - Downloads PDF bytes from blob storage.
   - Extracts text and page numbers via PyMuPDF.
   - Splits into 1000-character chunks using `RecursiveCharacterTextSplitter`.
   - Generates 768-dim embeddings using the local `SentenceTransformer`.
   - Bulk inserts into `document_chunks` and `document_embeddings` (pgvector).
   - Calls Groq to generate a brief document summary.
   - Updates `documents.status` to `ready`.
9. React polls `GET /api/documents/{id}/status` until `ready`.

---

## Flow 4: Sending a Chat Message (LangGraph)

User types a question in an active session and presses Enter.

1. React calls `POST /api/sessions/{session_id}/messages` with `{ content, explicit_mode }`.
2. FastAPI validates the JWT and queries `assistant_sessions WHERE id = session_id AND user_id = current_user_id`. If not found, returns 404.
3. The user's message is saved to the `messages` table with `role="user"`.
4. If the session title is still "New Session", it is auto-renamed to the first 40 characters of the question.
5. The backend loads the last 6 messages from the session as conversation history.
6. The backend fetches all attached `document_ids` from `session_documents`.
7. The `LexFindState` dict is assembled and the graph is executed synchronously via `run_in_threadpool(lex_graph.invoke, ...)` — the graph nodes themselves are synchronous.

**Inside LangGraph:**

**Node 1 — classifier_node**
- If `explicit_mode` is "document" or "corpus", skips the LLM and hard-routes.
- Otherwise, makes one Groq call with `response_format=json_object` to determine `is_legal` and `intent`.
- If `is_legal=false`, routes to `blocked` node which sets a rejection message and ends.
- Otherwise routes to `general_chat`, `document_chat`, or `corpus_search`.

**Pipeline nodes** (see [agent.md](agent.md) for the full 14-node topology)
- `query_enhancement` may rewrite the query (one optional LLM call).
- `general_chat_node` (intent = "general"): builds `prompt_messages` from system prompt + history + question. No retrieval.
- `document_chat_node` (intent = "document_chat"):
  - For each attached document:
    - If `source_type="legal_case"`: queries Qdrant with a `document_id` filter, limit 8.
    - If `source_type="uploaded"`: queries pgvector via raw SQL with a `document_id` filter, limit 8.
  - Builds citations directly from chunk payload/row data.
  - Builds `prompt_messages` with the retrieved context blocks.
- `corpus_search_node` (intent = "corpus_search"):
  - Embeds the question into Dense and Sparse vectors and queries Qdrant using **Hybrid RRF** without any document filter, limit 15.
  - Deduplicates results by `document_id`, keeping the highest-scoring chunk per document, and takes the top 5 unique documents.
  - Builds citations from Qdrant payload.
- `retrieval_grader` grades the retrieved chunks (one LLM call) and may trigger one re-retrieval with an alternative strategy (`corpus_re_retrieval_node`).
- `web_augment` optionally runs a Tavily search for recency-sensitive queries (requires `TAVILY_API_KEY`).
- The final node builds the complete `prompt_messages` for answer generation. The nodes themselves do not call the LLM for the final answer — the API layer does.

8. Back in the API layer, if the graph produced `prompt_messages`, the answer is generated with the Async Groq client (`stream=True`) and each token is yielded as `data: {"content": "..."}` SSE.
9. If citations exist, FastAPI yields `event: citations\ndata: [...]`.
10. FastAPI yields `data: [DONE]`.
11. The complete answer and citations are saved to the `messages` table with `role="assistant"`, and after the first exchange an LLM-generated session title replaces the placeholder title.

---

## Flow 5: Viewing a PDF

User clicks "View PDF" on a citation pill or document card.

1. Corpus cases: React opens `GET /api/search/pdf/{id}` (case UUID or filename, no auth). Uploaded documents: React opens `GET /api/documents/{document_id}/pdf` (auth required).
2. The documents route fetches the `Document` record and checks authorization:
   - Corpus cases (`source_type="legal_case"`): accessible to any authenticated user.
   - Uploaded docs: accessible only to `owner_id`.
3. The backend retrieves bytes from local storage or Azure Blob.
4. Returns the PDF with `Content-Disposition: inline` so the browser renders it natively in the viewer modal.

---

## Authentication and Session Isolation

Every request to a protected endpoint passes through `get_current_user`:
- Extracts and validates the JWT signature.
- Checks token expiry.
- Returns the `user_id` UUID from the token payload.
- On any failure: 401 Unauthorized.

Every session access additionally runs:
```sql
SELECT * FROM assistant_sessions WHERE id = :session_id AND user_id = :user_id
```
If the query returns nothing, the endpoint returns 404 — making it impossible for one user to read or write another user's sessions or messages.
