"""Local FastAPI app for Heads-Up GUI and Tournament ENV Phase 1.1.

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
from .tournament_router import build_tournament_router
from holdem_ai.tournament_service.session import TournamentSessionRegistry
from holdem_ai.tournament_service.journal import TournamentJournal

LOCAL_FRONTENDS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)


def create_app(*, max_sessions: int = 128, catalog: CheckpointCatalog | None = None,
               tournament_journal_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="holdem-ai Local HUNL and Tournament API",
        version="1.6.0",
        description="Local-only HUNL200 GUI and six-seat Tournament ENV APIs.",
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
    tournament_registry = TournamentSessionRegistry(
        max_sessions=max_sessions, journal=TournamentJournal(tournament_journal_path))
    app.include_router(build_tournament_router(tournament_registry))
    # Explicit reference for in-process tests and local debugging only.
    app.state.gui11_registry = registry
    app.state.gui14_catalog = catalog
    app.state.tournament_env_p1_registry = tournament_registry
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
