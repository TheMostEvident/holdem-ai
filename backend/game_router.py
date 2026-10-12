"""HTTP adapter for GUI-1.0 sessions: no poker rules implemented here.

A session's RLock guards action transitions and next-hand revision checks.
Only strict public projections reach the API. Local research use only: no
remote authentication, persistence, or public-server safety guarantees.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response, status

from holdem_ai.game_service.checkpoint_catalog import CheckpointCatalog
from holdem_ai.game_service.session_manager import HUNLSessionRegistry

from .schemas import (
    GameResponse,
    BotsResponse,
    HealthResponse,
    NewSessionRequest,
    NextHandRequest,
    RaiseToRequest,
    SubmitActionRequest,
    HistoryResponse,
    SessionStatsResponse,
)


def build_router(registry: HUNLSessionRegistry, catalog: CheckpointCatalog | None = None) -> APIRouter:
    catalog = catalog or CheckpointCatalog()
    router = APIRouter(prefix="/api/v1/hunl", tags=["local-hunl"])

    def session_or_404(session_id: str):
        try:
            return registry.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found") from None

    def run_game_operation(callback):
        try:
            return callback()
        except ValueError as exc:
            if "stale revision" in str(exc):
                raise HTTPException(status_code=409, detail="stale revision; refresh state") from None
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @router.get("/health", response_model=HealthResponse)
    def health():
        return {"status": "ok", "service": "hunl-gui-local"}

    @router.get("/bots", response_model=BotsResponse)
    def bots(response: Response):
        response.headers["Cache-Control"] = "no-store"
        return {"bots": catalog.options(), "device": catalog.device,
                "policy_mode": catalog.mode}

    @router.post("/sessions", response_model=GameResponse, status_code=status.HTTP_201_CREATED)
    def create_session(request: NewSessionRequest, response: Response):
        if request.small_blind >= request.big_blind:
            raise HTTPException(status_code=422, detail="small_blind must be less than big_blind")
        if request.bot in ("ppo", "qboost") and (
            request.big_blind != 100 or request.small_blind != 50 or request.stack_bb != 200
        ):
            raise HTTPException(status_code=422,
                                detail="Trained HUNL200 checkpoints require blinds 50/100 and 200BB stacks")
        try:
            policy = catalog.build_policy(request.bot)
            session_id, game = registry.create(
                bot_policy=policy, seed=request.seed,
                small_blind=request.small_blind, big_blind=request.big_blind,
                stack_bb=request.stack_bb, stack_mode=request.stack_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        response.headers["Cache-Control"] = "no-store"
        return {"session_id": session_id, "state": game}

    @router.get("/sessions/{session_id}", response_model=GameResponse)
    def get_session(session_id: str, response: Response):
        response.headers["Cache-Control"] = "no-store"
        return {"session_id": session_id, "state": session_or_404(session_id).view()}

    @router.get("/sessions/{session_id}/stats", response_model=SessionStatsResponse)
    def session_stats(session_id: str, response: Response):
        response.headers["Cache-Control"] = "no-store"
        return session_or_404(session_id).summary()

    @router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
    def hand_history(session_id: str, response: Response,
                     offset: int = Query(default=0, ge=0),
                     limit: int = Query(default=50, ge=1, le=100)):
        response.headers["Cache-Control"] = "no-store"
        return session_or_404(session_id).history(offset=offset, limit=limit)

    @router.post("/sessions/{session_id}/actions", response_model=GameResponse)
    def take_action(session_id: str, request: SubmitActionRequest, response: Response):
        session = session_or_404(session_id)
        state = run_game_operation(
            lambda: session.play_human_action(
                request.action_index, expected_revision=request.expected_revision
            )
        )
        response.headers["Cache-Control"] = "no-store"
        return {"session_id": session_id, "state": state}

    @router.post("/sessions/{session_id}/raise-to", response_model=GameResponse)
    def human_raise_to(session_id: str, request: RaiseToRequest, response: Response):
        session = session_or_404(session_id)
        state = run_game_operation(lambda: session.play_human_raise_to(
            request.raise_to, expected_revision=request.expected_revision))
        response.headers["Cache-Control"] = "no-store"
        return {"session_id": session_id, "state": state}

    @router.post("/sessions/{session_id}/next-hand", response_model=GameResponse)
    def next_hand(session_id: str, request: NextHandRequest, response: Response):
        session = session_or_404(session_id)
        state = run_game_operation(lambda: session.start_next_hand(expected_revision=request.expected_revision))
        response.headers["Cache-Control"] = "no-store"
        return {"session_id": session_id, "state": state}

    @router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session(session_id: str):
        session_or_404(session_id)
        try:
            registry.delete(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found") from None
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return router
