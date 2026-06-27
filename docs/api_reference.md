# API Reference

All endpoints are prefixed with `/api`. Protected endpoints require an `Authorization: Bearer <token>` header containing a valid JWT.

---

## Authentication

### POST /api/auth/register
Register a new user account.

Request body:
```json
{ "email": "user@example.com", "password": "..." }
```
Response: `{ "id", "email", "role", "created_at" }`

### POST /api/auth/login
Authenticate and receive a JWT token.

Request body:
```json
{ "email": "...", "password": "..." }
```
Response: `{ "access_token", "token_type": "bearer" }`

### GET /api/auth/me
Returns the current authenticated user's profile. Protected.

---

## Sessions

### POST /api/sessions
Create a new chat session. Protected.

Request body: `{ "title": "New Session" }` (optional)

Response: `{ "id", "title", "created_at", "updated_at" }`

### GET /api/sessions
List all sessions for the current user, ordered by `updated_at` descending. Protected.

### GET /api/sessions/{session_id}
Get a single session. Returns 404 if it does not belong to the current user. Protected.

### PATCH /api/sessions/{session_id}
Rename a session. Protected.

Request body: `{ "title": "New Title" }`

### DELETE /api/sessions/{session_id}
Delete a session and all its messages. Cascades to `session_documents`. Protected.

---

## Messages

### GET /api/sessions/{session_id}/messages
Retrieve the full message history for a session. Protected.

Response: list of `{ id, role, message_type, content, citations, created_at }`

The `citations` field is a JSON array of objects. Format depends on source:
- Corpus/Qdrant: `{ document_id, chunk_id, title, court, year, citation, score }`
- Uploaded/pgvector: `{ document_id, chunk_id, title, page_number, filename, score }`

### POST /api/sessions/{session_id}/messages
Send a message to the AI and receive an SSE stream response. Protected.

Request body: `{ "content": "What is Article 21?", "explicit_mode": "auto" }`

`explicit_mode` values:
- `"auto"` (default): LangGraph classifies intent via LLM
- `"document"`: Force document_chat path regardless of question content
- `"corpus"`: Force corpus_search path regardless of question content

Response: `text/event-stream` (SSE)

SSE event format:
```
data: {"content": "token..."}     <- streamed per chunk
data: [DONE]                       <- signals stream complete
event: citations
data: [{"title": "...", ...}]      <- only if citations exist
```

---

## Session Documents

### POST /api/sessions/{session_id}/documents
Attach a document to a session (for RAG context). Protected.

Request body: `{ "document_id": "uuid" }`

### DELETE /api/sessions/{session_id}/documents/{document_id}
Detach a document from a session. Protected.

---

## Documents

### POST /api/documents/upload
Upload a private PDF for processing. Protected.

Request: `multipart/form-data` with `file` field (PDF only).

Response:
```json
{
  "id": "uuid",
  "title": "filename.pdf",
  "status": "uploaded",
  "message": "Document uploaded and processing started"
}
```

If the file hash matches an existing document, the existing record is returned immediately without re-processing.

### GET /api/documents/{document_id}/status
Poll the processing status of a document. Protected.

Response: `{ "id", "title", "status", "summary", "error_message" }`

Status values: `uploaded` -> `processing` -> `ready` | `failed`

### GET /api/documents/{document_id}/pdf
Serve a document PDF inline (for both corpus cases and user uploads). Protected.

Returns `application/pdf` with `Content-Disposition: inline`. Corpus documents are accessible to any authenticated user. Uploaded documents require `owner_id` match.

---

## Cases (Corpus)

### POST /api/cases/search
Semantic search across the 46k Qdrant corpus.

Request body:
```json
{
  "query": "right to privacy",
  "limit": 10,
  "filters": {
    "court": "Supreme Court",
    "year_min": 2000,
    "year_max": 2023
  }
}
```

Response: list of case cards with similarity score and metadata.

### GET /api/cases/{case_id}
Retrieve full metadata for a specific corpus case.

### POST /api/cases/{case_id}/analyze
Attach a corpus case to a new chat session. Creates the session and `Document` record if needed, then returns the session ID for navigation.

Response: `{ "session_id", "document_id" }`

Note: Corpus cases are not re-processed through Celery. The Document record is created with `status="ready"` and the agent queries Qdrant directly using the `document_id` filter.

---

## Search

### POST /api/search
Alternative semantic search endpoint.

### POST /api/search/by-name
Search for cases by exact or partial case name.

### GET /api/search/case/{document_id}
Get detailed case information.

### GET /api/search/similar/{document_id}
Find cases similar to a given case.

### POST /api/search/ask
Ask a question directly against the corpus search service.

---

## Health

### GET /api/health
Returns server status and configuration. No authentication required.
