# System Overview

LexFind is a legal research platform built over 46,456 indexed Indian Supreme Court judgments. It provides semantic search, persistent chat sessions, document-specific RAG, and full-corpus synthesis — all orchestrated by a LangGraph agent that routes each request to the appropriate execution path.

## High-Level Architecture

The system is composed of five layers:

1. **Frontend (React / Vite):** Single Page Application. Manages UI state, JWT authentication, and streams AI responses via Server-Sent Events.

2. **Backend (FastAPI):** Stateless API server. Handles JWT validation, session ownership enforcement, SSE streaming, and dispatches tasks to Celery.

3. **Agent Orchestrator (LangGraph):** A state machine (`lex_graph`) that runs on each chat request. Classifies intent in one LLM call and routes to the correct answer node, which makes the second and final LLM call.

4. **Background Processing (Celery + RabbitMQ):** Handles PDF text extraction, chunking, and embedding generation asynchronously without blocking the API.

5. **Databases:**
   - **PostgreSQL + pgvector:** Stores all relational data (users, sessions, messages, documents) and 768-dim vector embeddings for private user uploads.
   - **Qdrant:** Hosts the pre-indexed 46k corpus (1,163,447 vectors). Queries are executed using a **Hybrid RRF (Reciprocal Rank Fusion)** approach, combining dense embeddings (`all-mpnet-base-v2`) with sparse BM25 vectors for maximum accuracy.

```
User
 |
 v
React SPA
 |  HTTP/SSE
 v
FastAPI (stateless)
 |          |
 |          v (upload)
 |        RabbitMQ --> Celery Worker --> pgvector
 |
 v (chat)
LangGraph State Machine
 |
 +-- classifier_node  (LLM Call 1)
 |
 +-- general_chat     (LLM Call 2 - no retrieval)
 +-- document_chat    (pgvector or Qdrant filtered + LLM Call 2)
 +-- corpus_search    (Hybrid RRF Qdrant search + LLM Call 2)
 +-- blocked          (guardrail rejection - no LLM call)
 |
 v
SSE Stream --> Frontend --> DB persist
```

## Core Design Decisions

**Dual vector store:** Private uploads go to pgvector (scoped by `owner_id` and `document_id`). The shared corpus stays in Qdrant. The agent checks `source_type` on each document to route retrieval correctly — there is no cross-contamination.

**Exactly two LLM calls per request:** The classifier (Node 1) does intent classification and guardrail checking in a single JSON-mode Groq call. The selected answer node (Node 2) does retrieval and synthesis in one more call. No intermediate LLM steps.

**Citations from payload, not LLM:** Qdrant search results contain `title`, `court`, `year`, `citation`, and `chunk_id` in their payload. Citation objects are built directly from this data. The LLM is never asked to extract or generate citations.

**Ephemeral state:** `LexFindState` is a plain Python dict created per-request and discarded after the response is sent. No LangGraph checkpointing, no Redis. Persistence is handled by the `messages` table in PostgreSQL.

**Stateless API:** Every request re-validates the JWT and re-checks session ownership against the database. There is no in-memory session store on the server.

## Database Schema

| Table | Purpose |
|---|---|
| `users` | Authenticated user accounts |
| `assistant_sessions` | Chat threads owned by a user |
| `messages` | Individual messages within a session (with citations JSON) |
| `documents` | PDF resource registry — both uploaded and corpus |
| `session_documents` | Many-to-many join: sessions to attached documents |
| `document_chunks` | Raw text segments extracted from private uploads |
| `document_embeddings` | pgvector 768-dim embeddings for private chunks |

The `documents.source_type` column (`uploaded` or `legal_case`) is the single discriminator that determines which vector store the agent queries.
