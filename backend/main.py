"""GUI-1.1 local FastAPI application. No React frontend in this phase.

Start from the repo root with:
    uv run --extra dev --extra gui python -m backend.main

For local-only research. Do not publish this unauthenticated session server.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from holdem_ai.game_service.session_manager import HUNLSessionRegistry
from holdem_ai.game_service.checkpoint_catalog import CheckpointCatalog

from .game_router import build_router

LOCAL_FRONTENDS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)


def create_app(*, max_sessions: int = 128, catalog: CheckpointCatalog | None = None) -> FastAPI:
    app = FastAPI(
        title="holdem-ai HUNL Local Game API",
        version="1.6.0",
        description="Local-only research API for human-versus-agent HUNL200 sessions.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(LOCAL_FRONTENDS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    registry = HUNLSessionRegistry(max_sessions=max_sessions)
    catalog = catalog or CheckpointCatalog()
    app.include_router(build_router(registry, catalog))
    # Explicit reference for in-process tests and local debugging only.
    app.state.gui11_registry = registry
    app.state.gui14_catalog = catalog
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
