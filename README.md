# LexFind

AI-powered legal research and conversational analysis platform. Search across 46,000+ indexed Indian Supreme Court judgments using semantic similarity, chat with specific cases or your own uploaded PDFs, and ask general legal questions — all from a single persistent workspace backed by a LangGraph agent orchestrator.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Documentation](#documentation)
- [RAG Metrics & Evaluation](#rag-metrics--evaluation)

## Features

**Semantic Search**
Natural language search over 46,456 pre-indexed Indian Supreme Court judgments. Queries are embedded using `sentence-transformers/all-mpnet-base-v2` and searched against a Qdrant vector database (1.1M+ vectors) with hybrid retrieval — dense vectors fused with BM25 sparse vectors via Reciprocal Rank Fusion. Results include metadata filters for court, year, state, and case type.

**Agentic RAG Chat**
Every chat request flows through a 14-node LangGraph state machine that classifies intent, guards against off-topic input, and routes to one of three execution paths: general legal knowledge, document-specific RAG, or full corpus search. The number of LLM calls varies by path (see [docs/agent.md](docs/agent.md) for the per-path breakdown). Citations are built directly from Qdrant payload metadata — no LLM extraction.

**Private PDF Upload**
Users can upload their own PDFs. A Celery worker asynchronously extracts text via PyMuPDF, chunks it with LangChain, generates 768-dim embeddings, and stores them in the private pgvector store isolated by `owner_id`. Private documents never interact with the shared Qdrant corpus. Note: LLM inference is done via the Groq API — uploaded document content is sent to Groq's API when you chat with it, so do not upload genuinely confidential material.

**Document-Specific Chat**
Clicking Analyze on a corpus search result or uploading a private PDF attaches it to a chat session. The agent automatically searches the correct vector store: Qdrant (filtered by document ID) for corpus cases, pgvector for private uploads. Both paths produce cited, grounded answers.

**Session Management**
Conversations are persistent and multi-document. Each session stores ordered messages, attached documents, and citations in PostgreSQL. Sessions are strictly isolated by `user_id` and `session_id` — every API request re-validates JWT ownership before any data is read or written.

## Architecture

```mermaid
flowchart TD
    classDef page fill:#2563eb,stroke:#1e3a8a,color:#fff
    classDef agent fill:#4f46e5,stroke:#312e81,color:#fff
    classDef backend fill:#059669,stroke:#064e3b,color:#fff
    classDef userNode fill:#64748b,stroke:#475569,color:#fff
    
    User([User]):::userNode
    
    subgraph Frontend ["React Frontend"]
        SearchPage["Search Page"]:::page
        AssistantPage["Assistant Page (Chat)"]:::page
    end
    
    subgraph Backend ["FastAPI Backend & Services"]
        Qdrant[("Qdrant Database<br>(46k Cases)")]:::backend
        SyncWorker["Async Document Processing<br>(Celery + Extract + Embed)"]:::backend
        Agent{"LangGraph Agent<br>Orchestrator"}:::agent
    end
    
    User --> SearchPage
    User --> AssistantPage
    
    SearchPage -- "Clicks 'Analyze'" --> AssistantPage
    SearchPage -- "Hybrid & Keyword Search" --> Qdrant
    
    AssistantPage -- "Uploads PDF" --> SyncWorker
    AssistantPage -- "Asks Question" --> Agent
    Agent -. "Streams Answer + Citations" .-> AssistantPage
```

Infrastructure layers:

- **FastAPI** handles HTTP, JWT auth, session ownership, and SSE streaming.
- **LangGraph** orchestrates intent classification and retrieval routing.
- **Celery + RabbitMQ** processes document uploads asynchronously.
- **PostgreSQL + pgvector** stores relational data and private document vectors.
- **Qdrant** stores the 46k-case shared corpus (1.1M vectors, metadata indexed).
- **Groq** runs `openai/gpt-oss-120b` and `openai/gpt-oss-20b` for LLM inference.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | React 18, Vite, TailwindCSS |
| Backend | FastAPI, Python 3.11+, SQLAlchemy, Alembic |
| Agent Orchestration | LangGraph, LangChain |
| Task Queue | Celery, RabbitMQ |
| Database | PostgreSQL 17, pgvector |
| Vector Store (Corpus) | Qdrant |
| LLM | Groq openai/gpt-oss-120b (primary), openai/gpt-oss-20b |
| Embeddings | sentence-transformers/all-mpnet-base-v2 (768-dim) |
| PDF Processing | PyMuPDF, LangChain RecursiveCharacterTextSplitter |
| File Storage | Local Storage (USE_LOCAL_FILES=true) |
| Deployment | Local Docker Compose (current demo); Azure VM + Azure Static Web Apps (historical deployment) |

## Project Structure

```
LexFind/
├── backend/
│   ├── alembic/                  # Database migrations
│   ├── app/
│   │   ├── agents/               # LangGraph agent orchestrator
│   │   │   ├── graph.py          # Compiled StateGraph singleton
│   │   │   ├── state.py          # LexFindState TypedDict
│   │   │   └── nodes/
│   │   │       ├── classifier.py     # Node 1: intent + guardrail
│   │   │       ├── general_chat.py   # Node 2A: general legal Q&A
│   │   │       ├── document_chat.py  # Node 2B: document RAG
│   │   │       ├── corpus_search.py  # Node 2C: full corpus search
│   │   │       ├── _embedder.py      # Shared embedding helper
│   │   │       └── _qdrant.py        # Shared Qdrant client
│   │   ├── api/                  # FastAPI routers
│   │   │   ├── auth.py           # Register, login, profile
│   │   │   ├── sessions.py       # Chat sessions + SSE streaming
│   │   │   ├── documents.py      # PDF upload and status
│   │   │   ├── cases.py          # Corpus case management
│   │   │   ├── search.py         # Semantic search endpoints
│   │   │   └── dependencies/     # JWT validation
│   │   ├── db/
│   │   │   ├── models.py         # SQLAlchemy models
│   │   │   ├── crud/             # Repository functions
│   │   │   └── session.py        # DB session management
│   │   ├── schemas/              # Pydantic request/response models
│   │   ├── services/             # Embedding, retrieval, blob storage
│   │   ├── workers/              # Celery document processing task
│   │   └── main.py               # App factory
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/           # Reusable UI components
│       ├── context/              # React Auth Context
│       └── pages/                # Search, Assistant, Login pages
├── docs/                         # Technical documentation
└── docker-compose.yml
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- Groq API key from console.groq.com
- Qdrant running locally on port 6333

### Backend

```bash
# 1. Start infrastructure
docker compose up db rabbitmq -d

# 2. Set up Python environment
cd backend
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set GROQ_API_KEY, DATABASE_URL, CELERY_BROKER_URL

# 4. Run database migrations
alembic upgrade head

# 5. Start API server
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload

# 6. Start Celery worker (separate terminal)
celery -A app.workers.celery_app worker --loglevel=info -Q lexfind_documents
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Environment Variables

### Backend (`backend/.env`)

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `CELERY_BROKER_URL` | RabbitMQ connection string (Celery broker) |
| `CELERY_RESULT_BACKEND` | Celery result backend (PostgreSQL) |
| `GROQ_API_KEY` | Required for LLM inference |
| `JWT_SECRET_KEY` | JWT signing key |
| `QDRANT_HOST` | Qdrant host (default: localhost) |
| `QDRANT_PORT` | Qdrant port (default: 6333) |
| `AZURE_STORAGE_CONNECTION_STRING` | Optional, enables Azure Blob Storage |
| `GROQ_MODEL` | LLM model name (default: openai/gpt-oss-120b) |
| `REDIS_URL` | Optional, enables Redis rate limiting |

### Frontend (`frontend/.env`)

| Variable | Description |
|---|---|
| `VITE_API_BASE_URL` | Backend URL (default: http://localhost:8000) |

## RAG Metrics & Evaluation

Retrieval was evaluated empirically against 192 randomly sampled judgments × 3 query types (excerpt, title, sentence) = 576 queries per mode, run in two modes: dense-only retrieval and hybrid retrieval (dense + BM25 sparse fused via Reciprocal Rank Fusion). Hybrid retrieval lifted Hit@5 for excerpt queries from 35.9% to **82.3%**.

The full evaluation report — per-query-type Hit@1/3/5, MRR, and known limitations — is in **[docs/rag_metrics.md](docs/rag_metrics.md)**. Note: the 46k-judgment corpus itself and the raw evaluation output files are not committed to this repository (both are git-ignored), so the experiments cannot be re-run from the repo alone; the numbers above come from the runs documented in the metrics report.

## Documentation

Detailed technical documentation is available in the `docs/` directory:

- [System Overview & Architecture](docs/overview.md)
- [Setup & Installation](docs/setup_and_installation.md)
- [Local Demo Script (for evaluations)](docs/demo_script.md)
- [Data Flows & API Design](docs/data_flows.md)
- [LangGraph Agent Orchestrator](docs/agent.md)
- [Database Schema & Migrations](docs/database.md)
- [API Reference](docs/api_reference.md)
- [RAG Search Accuracy Metrics](docs/rag_metrics.md)
