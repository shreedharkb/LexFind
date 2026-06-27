# Database Schema & Data Models

LexFind utilizes a dual-database persistence layer: **PostgreSQL** (with the `pgvector` extension) for relational data, session management, and private document storage, and **Qdrant** for the massive, read-heavy pre-indexed legal corpus.

---

## 1. Relational Schema (PostgreSQL)

The primary datastore is PostgreSQL. We use SQLAlchemy as the ORM. The domain model revolves around user accounts, active assistant sessions, message histories, and private document processing.

### Core Tables

#### `users`
Represents an authenticated user.
- **id**: UUID (Primary Key)
- **email**: String (Unique)
- **hashed_password**: String
- **role**: String (e.g., 'user', 'admin')
- *Relationships*: Owns `assistant_sessions` and uploaded `documents`.

#### `assistant_sessions`
A ChatGPT-style conversational thread.
- **id**: UUID (Primary Key)
- **user_id**: UUID (Foreign Key)
- **title**: String
- *Relationships*: Contains ordered `messages`. Can link to multiple `documents` via `session_documents`.

#### `messages`
A single turn in the conversation (persists the chat history for the LangGraph state).
- **id**: UUID (Primary Key)
- **session_id**: UUID (Foreign Key)
- **role**: Enum (`user`, `assistant`, `system`)
- **message_type**: Enum (`text`, `summary_card`, `event_card`, `legal_notice_card`)
- **content**: Text
- **citations**: JSON (Array of objects containing doc_name, page_number, excerpt)

---

### Document Processing & Vector Storage (pgvector)

LexFind allows users to upload their own PDFs. To ensure complete privacy and strict data isolation, uploaded documents are chunked and embedded directly into **PostgreSQL using `pgvector`**, keeping them completely separate from the public Qdrant corpus.

#### `documents`
Tracks a PDF resource. This table manages the async ingestion pipeline state.
- **id**: UUID (Primary Key)
- **owner_id**: UUID (Foreign Key, Nullable for public docs)
- **source_type**: Enum (`uploaded`, `legal_case`)
- **file_hash**: String (SHA-256 for strict deduplication at the user level)
- **status**: Enum (`uploaded`, `processing`, `ready`, `failed`)
- **blob_path**: String (Azure/Local file path)

#### `session_documents` (Join Table)
Many-to-many link between sessions and documents, allowing a user to query specific subsets of their uploaded files in a single chat.

#### `document_chunks`
Extracted text segments mapped sequentially.
- **document_id**: UUID (Foreign Key)
- **page_number**: Integer
- **chunk_index**: Integer
- **chunk_text**: Text

#### `document_embeddings`
The `pgvector` table for semantic search on user uploads.
- **chunk_id**: UUID (Foreign Key)
- **document_id**: UUID (Foreign Key - denormalized for fast filtering)
- **embedding**: Vector(768)
- *Note:* We use an **HNSW index** on this column to ensure sub-50ms cosine similarity searches, even when users upload hundreds of pages.

---

## 2. Global Search Corpus (Qdrant)

While `pgvector` handles private, ephemeral user uploads, **Qdrant** is the workhorse for the global corpus search. 

### Collection: `legal_corpus`
This collection holds **1.1 million vector chunks** generated from 46,456 Indian Supreme Court judgments. 

- **Dense Vectors**: 768-dimensional float vectors (`all-mpnet-base-v2`).
- **Sparse Vectors**: BM25 keyword vectors.
- **Payload (Metadata)**: 
  - `document_id`: UUID (maps back to the `documents` table in Postgres for full hydration)
  - `title`: String
  - `court`: String
  - `year`: Integer
  - `judges`: Array of Strings
  - `case_type`: String

*Qdrant performs the Hybrid Reciprocal Rank Fusion (RRF) search, utilizing payload-based pre-filtering (e.g., matching a specific year or court BEFORE performing the vector distance calculation).*
