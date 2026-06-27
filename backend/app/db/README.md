# LexFind Database

PostgreSQL database module using SQLAlchemy and pgvector.


## Tables

- **`users`**: Platform users (auth).
- **`assistant_sessions`**: Chat threads.
- **`messages`**: Chat messages with role and citations.
- **`documents`**: PDF resources (uploaded or corpus).
- **`session_documents`**: Join table for M:N session attachments.
- **`document_chunks`**: Parsed text from PDFs.
- **`document_embeddings`**: 768-dim vectors using `pgvector`.

## Setup

1. Start Postgres: `docker compose up db -d`
2. Run migrations: `alembic upgrade head`
