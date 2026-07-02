"""
FastAPI application factory for LexFind.

Creates and configures the main FastAPI instance with CORS, error handlers,
route registration, and the lifespan context manager.


"""

import logging
import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend/ is importable
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    The FastAPI process itself is stateless — all heavy document processing
    is offloaded to an external Celery worker via RabbitMQ.

    To start the worker:
        celery -A workers.celery_app worker --loglevel=info -Q lexfind_documents
    """
    logger.info(
        "LexFind API starting — document processing delegated to Celery (RabbitMQ)."
    )
    yield
    logger.info("LexFind API shutting down.")


# ── Factory ─────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""

    # Load .env before anything else touches env vars
    env_path = Path(__file__).resolve().parent.parent / ".env"
    try:
        load_dotenv(dotenv_path=env_path, override=False)
        key = os.environ.get("GROQ_API_KEY", "")
        masked = f"{key[:4]}...{key[-4:]}" if key and len(key) >= 8 else "(not set)"
        print(f"Loaded env from {env_path} | GROQ_API_KEY={masked}")
    except Exception as err:
        print(f"Warning: could not load .env ({env_path}): {err}")

    app = FastAPI(
        title="LexFind API",
        description=(
            "AI-powered legal document search and analysis platform. "
            "Semantic search over 46,000+ Supreme Court cases, agentic RAG chat, "
            "private document analysis, and a legal-domain chatbot."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",      # Vite dev
            "http://localhost:3000",      # CRA dev
            "https://lex-find.vercel.app", # Vercel production
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Error handlers ────────────────────────────────────────────────────
    from app.core.exceptions import register_error_handlers
    register_error_handlers(app)

    # ── Route registration ────────────────────────────────────────────────
    from app.api.auth import router as auth_router
    from app.api.cases import router as cases_router
    from app.api.sessions import router as sessions_router
    from app.api.documents import router as documents_router
    from app.api.search import router as search_router

    app.include_router(auth_router, prefix="/api")
    app.include_router(cases_router, prefix="/api")
    app.include_router(sessions_router, prefix="/api")
    app.include_router(documents_router, prefix="/api")
    app.include_router(search_router, prefix="/api")

    # ── Health & root ─────────────────────────────────────────────────────
    @app.get("/api/health", tags=["Health"])
    async def health():
        return {"status": "ok"}

    @app.get("/", tags=["Root"])
    async def root():
        storage = (
            "Azure Blob Storage"
            if os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            else "Local Files"
        )
        return {
            "message": "LexFind API — Legal AI Platform",
            "version": "2.0.0",
            "storage_backend": storage,
            "docs": "/docs",
        }

    return app


# ── CLI entry ───────────────────────────────────────────────────────────────

def main():
    """Launch the API server from the command line."""
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    reload = os.environ.get("RELOAD", "false").lower() == "true"

    print("=" * 50)
    print("🚀 Starting LexFind API")
    print(f"   → http://{host}:{port}")
    print(f"   → Docs: http://{host}:{port}/docs")
    print("=" * 50)

    try:
        uvicorn.run(
            "main:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")


if __name__ == "__main__":
    main()
