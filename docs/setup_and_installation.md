# Setup and Installation

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| Node.js | 18+ |
| Docker + Docker Compose | Latest |
| Groq API Key | From console.groq.com |
| Qdrant | Started by Docker Compose on port 6333 |

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/LexFind.git
cd LexFind
```

---

## 2. Configure the Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

From `backend/`, copy the example configuration and edit `backend/.env`:

```bash
cp .env.example .env
```

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/lexfind
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672//
CELERY_RESULT_BACKEND=db+postgresql://postgres:postgres@localhost:5433/lexfind
REDIS_URL=redis://localhost:6379/0
GROQ_API_KEY=gsk_...
JWT_SECRET_KEY=replace_with_a_random_64_character_hex_secret

# Qdrant (defaults shown)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LLM model (default shown)
GROQ_MODEL=llama-3.3-70b-versatile

# Keep uploads on the local filesystem for the demo.
USE_LOCAL_FILES=true
```

---

## 3. Start the Local Stack (recommended)

PostgreSQL, RabbitMQ, Redis, Qdrant, the API, and the Celery worker are managed
by Docker Compose. Run this from the repository root after configuring `backend/.env`:

```bash
docker compose up -d --build
```

This automatically runs `alembic upgrade head` before starting the API and worker.

---

## 4. Direct Backend Run (alternative)

```bash
alembic upgrade head
```

This creates all tables: `users`, `assistant_sessions`, `messages`, `documents`, `session_documents`, `document_chunks`, `document_embeddings`.

---

## 5. Start the API Server

```bash
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

API documentation available at `http://localhost:8000/docs`.

---

## 6. Start the Celery Worker

Open a separate terminal (with the venv activated):

```bash
celery -A app.workers.celery_app worker --loglevel=info -Q LexFind_documents
```

The Celery worker handles asynchronous PDF processing (text extraction, chunking, embedding generation). It must be running for private PDF uploads to become ready for chat. `docker compose up -d --build` starts it automatically.

---

## 7. Frontend Setup

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Start the dev server:

```bash
npm run dev
```

Open `http://localhost:5173`.

---

## Verification

```bash
# API health check
curl http://localhost:8000/api/health

# Check Qdrant is accessible
curl http://localhost:6333/collections
```

---

## Qdrant Corpus Loading

The legal corpus must be pre-loaded into Qdrant before search and corpus chat work. The active collection is named `legal_corpus_hybrid`, with named `dense` and `sparse` vectors, and contains the following payload schema:

| Field | Type | Description |
|---|---|---|
| `document_id` | string (UUID) | Links to `documents` table |
| `chunk_id` | string (UUID) | Unique chunk identifier |
| `chunk_text` | string | Raw text content |
| `chunk_index` | int | Position within document |
| `title` | string | Case name |
| `petitioner` | string | Petitioner name |
| `respondent` | string | Respondent name |
| `court` | string | Court name |
| `state` | string | State |
| `year` | int | Year of judgment |
| `citation` | string | Official citation |
| `case_type` | string | Classification |
| `section_type` | string | Chunk section type |

Payload indexes must exist on: `court`, `year`, `state`, `case_type`, `section_type`, `document_id`.

---

## Deployment

Deployment is intentionally deferred. The currently supported workflow is the local Docker Compose stack above.
