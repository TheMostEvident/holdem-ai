"""HTTP routes for Tournament ENV 1.1; separate from the HUNL200 routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
import json

from holdem_ai.tournament_service.session import TournamentSessionRegistry
from holdem_ai.tournament_service.trained_actor import status as trained_actor_status

from .tournament_schemas import (
    TournamentActionRequest,
    TournamentAgentStatus,
    TournamentCreateRequest,
    TournamentGameResponse,
    TournamentHealthResponse,
    TournamentHistoryResponse,
    TournamentRaiseToRequest,
)


def build_tournament_router(registry: TournamentSessionRegistry) -> APIRouter:
    router = APIRouter(prefix="/api/v1/tournament", tags=["local-6max-tournament"])

    def session_or_404(session_id: str):
        try:
            return registry.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="tournament session not found") from None

    def perform(operation):
        try:
            return operation()
        except ValueError as exc:
            if "stale revision" in str(exc):
                raise HTTPException(status_code=409, detail="stale revision; refresh tournament") from None
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @router.get("/health", response_model=TournamentHealthResponse)
    def health():
        return {"status": "ok", "service": "tournament-env-local", "phase": "1.1"}

    @router.get("/agent-status", response_model=TournamentAgentStatus)
    def agent_status(response: Response):
        response.headers["Cache-Control"] = "no-store"
        return trained_actor_status()

    @router.get("/archive")
    def archive(response: Response, offset: int = Query(default=0, ge=0),
                limit: int = Query(default=50, ge=1, le=100)):
        response.headers["Cache-Control"] = "no-store"
        return registry.journal.list_sessions(offset=offset, limit=limit)

    @router.get("/archive/{session_id}/hands")
    def archive_hands(session_id: str, response: Response,
                      offset: int = Query(default=0, ge=0),
                      limit: int = Query(default=50, ge=1, le=100)):
        response.headers["Cache-Control"] = "no-store"
        try:
            return registry.journal.hands(session_id, offset=offset, limit=limit)
        except KeyError:
            raise HTTPException(status_code=404, detail="archived session not found") from None

    @router.get("/archive/{session_id}/hands/{hand_number}/replay")
    def replay(session_id: str, hand_number: int, response: Response):
        response.headers["Cache-Control"] = "no-store"
        if hand_number < 1:
            raise HTTPException(status_code=422, detail="invalid hand number")
        try:
            return registry.journal.replay(session_id, hand_number)
        except KeyError:
            raise HTTPException(status_code=404, detail="archived hand not found") from None

    @router.get("/archive/{session_id}/export")
    def actor_export(session_id: str):
        try:
            rows = registry.journal.actor_export_rows(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="archived session not found") from None
        def lines():
            for row in rows:
                yield json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        return StreamingResponse(lines(), media_type="application/x-ndjson", headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="tournament_{session_id[:12]}_actor.jsonl"',
            "X-Content-Type-Options": "nosniff"})

    @router.post("/sessions", response_model=TournamentGameResponse,
                 status_code=status.HTTP_201_CREATED)
    def create_session(request: TournamentCreateRequest, response: Response):
        try:
            session_id, state = registry.create(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        response.headers["Cache-Control"] = "no-store"
        return {"session_id": session_id, "state": state}

    @router.get("/sessions/{session_id}", response_model=TournamentGameResponse)
    def get_session(session_id: str, response: Response):
        response.headers["Cache-Control"] = "no-store"
        return {"session_id": session_id, "state": session_or_404(session_id).view()}

    @router.get("/sessions/{session_id}/history", response_model=TournamentHistoryResponse)
    def history(session_id: str, response: Response,
                offset: int = Query(default=0, ge=0),
                limit: int = Query(default=50, ge=1, le=100)):
        response.headers["Cache-Control"] = "no-store"
        return session_or_404(session_id).history(offset=offset, limit=limit)

    @router.post("/sessions/{session_id}/actions", response_model=TournamentGameResponse)
    def action(session_id: str, request: TournamentActionRequest, response: Response):
        session = session_or_404(session_id)
        state = perform(lambda: session.play_action(
            request.action_index, expected_revision=request.expected_revision))
        response.headers["Cache-Control"] = "no-store"
        return {"session_id": session_id, "state": state}

    @router.post("/sessions/{session_id}/raise-to", response_model=TournamentGameResponse)
    def raise_to(session_id: str, request: TournamentRaiseToRequest, response: Response):
        session = session_or_404(session_id)
        state = perform(lambda: session.play_raise_to(
            request.raise_to, expected_revision=request.expected_revision))
        response.headers["Cache-Control"] = "no-store"
        return {"session_id": session_id, "state": state}

    @router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session(session_id: str):
        session_or_404(session_id)
        try:
            registry.delete(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="tournament session not found") from None
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return router
