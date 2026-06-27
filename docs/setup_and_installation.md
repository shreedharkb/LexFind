# Setup and Installation

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| Node.js | 18+ |
| Docker + Docker Compose | Latest |
| Groq API Key | From console.groq.com |
| Qdrant | Running on port 6333 |

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/LexFind.git
cd LexFind
```

---

## 2. Start Infrastructure Services

PostgreSQL (database) and RabbitMQ (task broker) are managed by Docker Compose.

```bash
docker compose up db rabbitmq -d
```

Qdrant must be running separately. To run it in Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## 3. Backend Setup

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

## 4. Configure Environment

```bash
cp .env.example .env
```

Edit `backend/.env`:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/LexFind
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
GROQ_API_KEY=gsk_...
SECRET_KEY=your-random-secret-key

# Qdrant (defaults shown)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LLM model (default shown)
GROQ_MODEL=llama-3.3-70b-versatile

# Azure Blob Storage (optional — omit to use local storage)
AZURE_STORAGE_CONNECTION_STRING=
AZURE_DATA_CONTAINER=data
```

---

## 5. Run Database Migrations

```bash
alembic upgrade head
```

This creates all tables: `users`, `assistant_sessions`, `messages`, `documents`, `session_documents`, `document_chunks`, `document_embeddings`.

---

## 6. Start the API Server

```bash
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

API documentation available at `http://localhost:8000/docs`.

---

## 7. Start the Celery Worker

Open a separate terminal (with the venv activated):

```bash
celery -A app.workers.celery_app worker --loglevel=info -Q LexFind_documents
```

The Celery worker handles asynchronous PDF processing (text extraction, chunking, embedding generation). It must be running for private PDF uploads to become ready for chat.

---

## 8. Frontend Setup

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

The 46k legal corpus must be pre-loaded into Qdrant before search and corpus chat work. The collection is named `legal_corpus` and must contain documents with the following payload schema:

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

## Production Deployment

The current production deployment uses:
- **Azure VM** for FastAPI and Celery
- **Azure Static Web Apps** for the React frontend
- **Azure Blob Storage** for PDF file storage
- **Docker Compose** for local orchestration of Postgres and RabbitMQ

Nginx is used as a reverse proxy on the Azure VM to forward traffic to the Uvicorn process.
