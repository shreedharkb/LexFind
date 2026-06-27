"""
Global exception handlers for the LexFind API.

Every error response follows a consistent shape:
    { "detail": str, "error_code": str, "timestamp": str, "request_id": str }

Registered in create_app() via register_error_handlers(app).


"""

import logging
import traceback
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _build_error(detail: str, error_code: str, request_id: str) -> dict:
    """Construct the standard error payload."""
    return {
        "detail": detail,
        "error_code": error_code,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
    }


def _request_id(request: Request) -> str:
    """Use the X-Request-ID header if present, otherwise generate one."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


def register_error_handlers(app: FastAPI) -> None:
    """Attach all global exception handlers to the FastAPI instance."""

    @app.exception_handler(HTTPException)
    async def handle_http(request: Request, exc: HTTPException):
        rid = _request_id(request)
        logger.warning(
            "HTTPException %s | %s | path=%s rid=%s",
            exc.status_code, exc.detail, request.url.path, rid,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error(exc.detail, f"HTTP_{exc.status_code}", rid),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        rid = _request_id(request)
        errors = exc.errors()
        parts = []
        for e in errors:
            loc = " → ".join(str(l) for l in e.get("loc", []))
            msg = e.get("msg", "Validation error")
            parts.append(f"{loc}: {msg}" if loc else msg)
        detail = "; ".join(parts) or "Invalid request data."
        logger.warning("Validation error | path=%s rid=%s errors=%s", request.url.path, rid, errors)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_build_error(detail, "VALIDATION_ERROR", rid),
        )

    @app.exception_handler(Exception)
    async def handle_generic(request: Request, exc: Exception):
        rid = _request_id(request)
        logger.error(
            "Unhandled exception | path=%s rid=%s\n%s",
            request.url.path, rid, traceback.format_exc(),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_build_error(
                "An internal server error occurred. Please try again later.",
                "INTERNAL_SERVER_ERROR", rid,
            ),
        )
