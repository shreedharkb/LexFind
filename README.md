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
Natural language search over 46,456 pre-indexed Indian Supreme Court judgments. Queries are embedded using `sentence-transformers/all-mpnet-base-v2` and compared against a Qdrant vector database (1.1M+ vectors). Results are ranked by cosine similarity and include metadata filters for court, year, state, and case type. To guarantee stability and prevent namespace conflicts, Qdrant search is implemented via direct HTTP REST API calls (`httpx`) rather than the Python SDK.

**Agentic RAG Chat**
Every chat request flows through a LangGraph state machine that classifies intent and routes to one of three execution paths: general legal knowledge, document-specific RAG, or full corpus search. The classifier and the answer node together make exactly two LLM calls per request. Citations are built directly from Qdrant payload metadata — no LLM extraction.

**Document-Specific Chat**
Clicking Analyze on a corpus search result or uploading a private PDF attaches it to a chat session. The agent automatically searches the correct vector store: Qdrant (filtered by document ID) for corpus cases, pgvector for private uploads. Both paths produce cited, grounded answers.

**Private PDF Upload**
Users can upload their own confidential PDFs. A Celery worker asynchronously extracts text via PyMuPDF, chunks it with LangChain, generates 768-dim embeddings, and stores them in the private pgvector store isolated by `owner_id`. Private documents never interact with the shared Qdrant corpus.

**Session Management**
Conversations are persistent and multi-document. Each session stores ordered messages, attached documents, and citations in PostgreSQL. Sessions are strictly isolated by `user_id` and `session_id` — every API request re-validates JWT ownership before any data is read or written.

## Architecture

```mermaid
flowchart TD
    classDef client fill:#2563eb,stroke:#1e3a8a,color:#fff
    classDef api fill:#059669,stroke:#064e3b,color:#fff
    classDef db fill:#d97706,stroke:#78350f,color:#fff
    classDef queue fill:#7c3aed,stroke:#4c1d95,color:#fff
    classDef llm fill:#db2777,stroke:#831843,color:#fff
    classDef lgraph fill:#4f46e5,stroke:#312e81,color:#fff

    React["React SPA<br>Vite + TailwindCSS"]:::client

    subgraph FastAPI["FastAPI Backend"]
        AuthRoute["POST /api/auth<br>JWT Auth"]:::api
        SearchRoute["POST /api/search<br>Corpus Search"]:::api
        UploadRoute["POST /api/documents/upload<br>PDF Upload"]:::api
        ChatRoute["POST /api/sessions/id/messages<br>SSE Stream"]:::api
        CasesRoute["POST /api/cases/id/analyze<br>Attach Case"]:::api
        DocStatusRoute["GET /api/documents/id/status<br>Poll Status"]:::api
    end

    React --> AuthRoute
    React --> SearchRoute
    React --> UploadRoute
    React --> ChatRoute
    React --> CasesRoute
    React --> DocStatusRoute

    subgraph Databases["Data Persistence Layer"]
        PG["PostgreSQL<br>Users, Sessions"]:::db
        PGVector["PostgreSQL pgvector<br>Private Embeddings"]:::db
        Qdrant["Qdrant<br>46k Cases, 1.1M Vectors"]:::db
        BlobStorage["Azure Blob / Local Disk<br>Raw PDFs"]:::db
    end

    AuthRoute <--> PG
    SearchRoute -->|"Direct HTTP<br>Dense Search"| Qdrant
    SearchRoute --> PG
    CasesRoute --> PG

    subgraph AsyncProcessing["Async Background Processing"]
        RabbitMQ["RabbitMQ<br>Task Broker"]:::queue
        CeleryWorker["Celery Worker<br>lexfind_documents"]:::queue
    end

    UploadRoute -->|"SHA-256<br>dedup check"| PG
    UploadRoute --> BlobStorage
    UploadRoute --> RabbitMQ
    RabbitMQ --> CeleryWorker
    CeleryWorker -->|"PyMuPDF extract<br>LangChain chunk"| BlobStorage
    CeleryWorker -->|"all-mpnet-base-v2<br>768-dim embed"| PGVector
    CeleryWorker -->|"status = ready"| PG

    LangGraphAgent["lex_graph State Machine<br>(See docs/agent.md for details)"]:::lgraph

    ChatRoute -->|"Validate JWT<br>ownership"| PG
    ChatRoute -->|"Persist user<br>message"| PG
    ChatRoute -->|"Load history<br>+ doc IDs"| PG
    ChatRoute --> LangGraphAgent

    LangGraphAgent <-->|"LLM API Calls"| GroqLLM["Groq API<br>llama-3.3-70b-versatile"]:::llm
    LangGraphAgent <-->|"Cosine search<br>by doc_id"| PGVector
    LangGraphAgent <-->|"Direct HTTP<br>Dense Search"| Qdrant

    LangGraphAgent -->|"answer + citations"| SSE["SSE Streamer"]
    SSE -->|"text/event-stream"| React
    SSE -->|"Persist assistant<br>message"| PG
```

Infrastructure layers:

- **FastAPI** handles HTTP, JWT auth, session ownership, and SSE streaming.
- **LangGraph** orchestrates intent classification and retrieval routing.
- **Celery + RabbitMQ** processes document uploads asynchronously.
- **PostgreSQL + pgvector** stores relational data and private document vectors.
- **Qdrant**: High-performance semantic search for the 46k case corpus. Queried via direct REST HTTP requests (bypassing the Python SDK) for robust unnamed-vector execution.
- **pgvector**: Local, isolated semantic search for user-uploaded private PDFs.

**Infrastructure**
- **Nginx (Reverse Proxy)**: Terminates SSL/HTTPS (`certbot`) on the Azure VM and forwards traffic to the FastAPI uvicorn workers.
- **Vercel**: Hosts the React frontend and handles proxy rewrites (`vercel.json`) to the secure backend domain.
- **RabbitMQ**: Message broker for Celery document processing.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | React 18, Vite, TailwindCSS |
| Backend | FastAPI, Python 3.11+, SQLAlchemy, Alembic |
| Agent Orchestration | LangGraph, LangChain |
| Task Queue | Celery, RabbitMQ |
| Database | PostgreSQL 17, pgvector |
| Vector Store (Corpus) | Qdrant |
| LLM | Groq llama-3.3-70b-versatile |
| Embeddings | sentence-transformers/all-mpnet-base-v2 (768-dim) |
| PDF Processing | PyMuPDF, LangChain RecursiveCharacterTextSplitter |
| File Storage | Azure Blob Storage (or local fallback) |
| Deployment | Docker Compose, Azure VM, Azure Static Web Apps |

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
│   ├── tests/                    # Unit & integration test suites
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
docker compose up db rabbitmq qdrant -d

# 2. Set up Python environment
cd backend
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set GROQ_API_KEY, DATABASE_URL, RABBITMQ_URL

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

### Running Tests

To run the backend test suite (including unit and integration tests for Azure Blob Storage and endpoints):

```bash
cd backend
pytest tests/ -v
```

## Environment Variables

### Backend (`backend/.env`)

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `RABBITMQ_URL` | RabbitMQ connection string |
| `GROQ_API_KEY` | Required for LLM inference |
| `SECRET_KEY` | JWT signing key |
| `QDRANT_HOST` | Qdrant host (default: localhost) |
| `QDRANT_PORT` | Qdrant port (default: 6333) |
| `AZURE_STORAGE_CONNECTION_STRING` | Optional, enables Azure Blob Storage |
| `GROQ_MODEL` | LLM model name (default: llama-3.3-70b-versatile) |

### Frontend (`frontend/.env`)

| Variable | Description |
|---|---|
| `VITE_API_BASE_URL` | Backend URL (default: http://localhost:8000) |

## RAG Metrics & Evaluation

LexFind underwent rigorous multi-system empirical evaluations against 192 test queries over the full 46,000 document corpus. 
We migrated from a baseline FAISS index to a **Qdrant Hybrid RRF** pipeline, boosting Hit@5 accuracy for excerpt queries from 35.9% to **82.3%**.

For full evaluation numbers, performance comparisons, and known limitations, please read the **[RAG Metrics Report](docs/rag_metrics.md)**.

## Documentation

Detailed technical documentation is available in the `docs/` directory:

- [System Overview & Architecture](docs/overview.md)
- [Data Flows & API Design](docs/data_flows.md)
- [Database Schema & Migrations](docs/database.md)
- [LangGraph Agent Orchestrator](docs/agent.md)
- [RAG Metrics & Limitations](docs/rag_metrics.md)
- [API Reference](docs/api_reference.md)
- [Setup & Installation](docs/setup_and_installation.md)

## Author

**Shreedhar K B** — Design, development, and deployment.

## License

This project is for educational and research purposes.
