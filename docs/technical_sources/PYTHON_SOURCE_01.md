# Full Python implementation — Volume 01 / 28

Part of [Poker AI final integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). These are **literal complete source files** from immutable commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. Test, CLI and package files included; no compression or omitted functions. Source text is reproduced exactly inside marked fenced blocks.


## PY-001 — `backend/__init__.py`

**Blob SHA:** `236c61b492556b4a4db7dae15fb72737c9140c75` · **Original bytes:** 67

<!-- BEGIN_PY_SOURCE path="backend/__init__.py" sha="236c61b492556b4a4db7dae15fb72737c9140c75" -->

````````python
"""Local-only HTTP transport for the GUI-1.0 HUNL game service."""

````````

<!-- END_PY_SOURCE path="backend/__init__.py" -->


## PY-002 — `backend/game_router.py`

**Blob SHA:** `ee2818d119f15e725da15485244ad043fba054fc` · **Original bytes:** 6122

<!-- BEGIN_PY_SOURCE path="backend/game_router.py" sha="ee2818d119f15e725da15485244ad043fba054fc" -->

````````python
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

````````

<!-- END_PY_SOURCE path="backend/game_router.py" -->


## PY-003 — `backend/main.py`

**Blob SHA:** `23cb998aa1217f728023fc1707e1790fba919685` · **Original bytes:** 2189

<!-- BEGIN_PY_SOURCE path="backend/main.py" sha="23cb998aa1217f728023fc1707e1790fba919685" -->

````````python
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

````````

<!-- END_PY_SOURCE path="backend/main.py" -->


## PY-004 — `backend/schemas.py`

**Blob SHA:** `6e4aaf2faba752d85bb1e215bc6383f9603ce436` · **Original bytes:** 4556

<!-- BEGIN_PY_SOURCE path="backend/schemas.py" sha="6e4aaf2faba752d85bb1e215bc6383f9603ce436" -->

````````python
"""Strict API request and intentionally whitelisted response models.

These are HTTP contracts, not model-training observations.  In particular,
no opponent private cards, training CriticState, or undealt deck appears in a
live GameView.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NewSessionRequest(StrictModel):
    bot: Literal["random", "calling_station", "tight", "pot_pressure", "ppo", "qboost"] = "random"
    seed: StrictInt | None = None
    big_blind: StrictInt = Field(default=100, ge=2, le=1_000_000)
    small_blind: StrictInt = Field(default=50, ge=1, le=999_999)
    stack_bb: StrictInt = Field(default=200, ge=2, le=100_000)
    stack_mode: Literal["continuous", "independent"] = "continuous"


class SubmitActionRequest(StrictModel):
    action_index: StrictInt = Field(ge=0, le=6)
    expected_revision: StrictInt = Field(ge=1)


class RaiseToRequest(StrictModel):
    raise_to: StrictInt = Field(ge=1)
    expected_revision: StrictInt = Field(ge=1)


class NextHandRequest(StrictModel):
    expected_revision: StrictInt = Field(ge=1)


class SeatView(StrictModel):
    seat: StrictInt
    role: Literal["human", "bot"]
    stack: StrictInt
    street_bet: StrictInt
    total_contribution: StrictInt
    status: str


class ActionOffer(StrictModel):
    index: StrictInt
    label: str
    kind: Literal["fold", "check", "call", "raise_to", "all_in"]
    raise_to: StrictInt | None


class PublicActionEvent(StrictModel):
    seat: StrictInt
    street: str
    index: StrictInt
    kind: Literal["fold", "check", "call", "raise_to", "all_in"]
    raise_to: StrictInt | None


class HandResultView(StrictModel):
    payoffs_chips: list[StrictInt]
    payoffs_bb: list[float]
    final_stacks: list[StrictInt]
    awards: list[StrictInt]
    showdown: bool
    opponent_cards: list[str] | None


class GameView(StrictModel):
    hand_number: StrictInt
    revision: StrictInt
    phase: Literal["playing", "complete"]
    game_type: Literal["HUNL200"]
    stack_mode: Literal["continuous", "independent"]
    starting_stacks: list[StrictInt]
    match_over: bool
    winner_seat: StrictInt | None
    hero_seat: StrictInt
    button: StrictInt
    small_blind: StrictInt
    big_blind: StrictInt
    street: Literal["preflop", "flop", "turn", "river"]
    board: list[str]
    hero_cards: list[str]
    seats: list[SeatView]
    pot: StrictInt
    to_act: StrictInt | None
    your_turn: bool
    actions: list[ActionOffer]
    to_call: StrictInt | None
    min_raise_to: StrictInt | None
    max_raise_to: StrictInt | None
    can_raise_to: bool
    action_log: list[PublicActionEvent]
    completed_hands: StrictInt
    cumulative_bb: float
    bb_per_100: float
    hand_result: HandResultView | None


class GameResponse(StrictModel):
    session_id: str
    state: GameView


class HealthResponse(StrictModel):
    status: Literal["ok"]
    service: Literal["hunl-gui-local"]


class BotOption(StrictModel):
    id: Literal["random", "calling_station", "tight", "pot_pressure", "ppo", "qboost"]
    label: str
    trained: bool
    available: bool
    message: str


class BotsResponse(StrictModel):
    bots: list[BotOption]
    device: Literal["cpu", "cuda"]
    policy_mode: Literal["sampled", "greedy"]


class CompletedHandView(StrictModel):
    hand_number: StrictInt
    starting_stacks: list[StrictInt]
    button: StrictInt
    street: Literal["preflop", "flop", "turn", "river"]
    big_blind: StrictInt
    small_blind: StrictInt
    hero_cards: list[str]
    board: list[str]
    action_log: list[PublicActionEvent]
    hero_net_chips: StrictInt
    hero_net_bb: float
    cumulative_bb: float
    outcome: Literal["win", "loss", "even"]
    result: HandResultView


class HistoryResponse(StrictModel):
    items: list[CompletedHandView]
    offset: StrictInt
    limit: StrictInt
    total: StrictInt
    completed_hands: StrictInt
    has_more: bool
    records_truncated: bool


class SessionStatsResponse(StrictModel):
    completed_hands: StrictInt
    wins: StrictInt
    losses: StrictInt
    break_even: StrictInt
    showdowns: StrictInt
    net_chips: StrictInt
    net_bb: float
    bb_per_100: float
    mean_bb_per_hand: float
    std_bb_per_hand: float | None
    ci95_lower_bb100: float | None
    ci95_upper_bb100: float | None
    max_drawdown_bb: float
    retained_hand_records: StrictInt
    records_truncated: bool
    note: str

````````

<!-- END_PY_SOURCE path="backend/schemas.py" -->


## PY-005 — `backend/tournament_router.py`

**Blob SHA:** `19464b0639069d89cd72eebde3010eaaf98fa11e` · **Original bytes:** 6543

<!-- BEGIN_PY_SOURCE path="backend/tournament_router.py" sha="19464b0639069d89cd72eebde3010eaaf98fa11e" -->

````````python
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

````````

<!-- END_PY_SOURCE path="backend/tournament_router.py" -->


## PY-006 — `backend/tournament_schemas.py`

**Blob SHA:** `5c5c424cac8d1a25aba117ac5f7a310e15cc4171` · **Original bytes:** 4253

<!-- BEGIN_PY_SOURCE path="backend/tournament_schemas.py" sha="5c5c424cac8d1a25aba117ac5f7a310e15cc4171" -->

````````python
"""Strict, whitelisted API contract for Tournament ENV Phase 1.1."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TournamentCreateRequest(StrictModel):
    seed: StrictInt | None = None
    lineup: Literal["baseline_5", "improved_4", "trained_1_improved_4"] = "baseline_5"
    starting_stack: StrictInt = Field(default=1000, ge=1, le=10_000_000)
    button: StrictInt = Field(default=0, ge=0, le=5)
    hands_per_level: StrictInt = Field(default=10, ge=1, le=100_000)
    max_hands: StrictInt = Field(default=10_000, ge=1, le=100_000)
    blind_levels: list[tuple[StrictInt, StrictInt]] | None = None


class TournamentActionRequest(StrictModel):
    action_index: StrictInt = Field(ge=0, le=4)
    expected_revision: StrictInt = Field(ge=1)


class TournamentRaiseToRequest(StrictModel):
    raise_to: StrictInt = Field(ge=1)
    expected_revision: StrictInt = Field(ge=1)


class TournamentSeatView(StrictModel):
    seat: StrictInt
    role: Literal["human", "baseline_bot", "tournament_agent_placeholder", "tournament_agent", "improved_bot"]
    policy_name: str
    stack: StrictInt
    status: str
    street_bet: StrictInt
    total_contribution: StrictInt
    rank: StrictInt | None


class TournamentActionOffer(StrictModel):
    index: StrictInt
    kind: str


class TournamentActionEvent(StrictModel):
    seat: StrictInt
    index: StrictInt
    kind: str
    raise_to: StrictInt | None
    hand_number: StrictInt | None = None
    street: str | None = None


class TournamentShowdownSeat(StrictModel):
    seat: StrictInt
    cards: tuple[str, str]


class TournamentHandView(StrictModel):
    hand_number: StrictInt
    button: StrictInt
    level_index: StrictInt
    small_blind: StrictInt
    big_blind: StrictInt
    stacks_before: list[StrictInt]
    stacks_after: list[StrictInt]
    eliminated: list[StrictInt]
    board: list[str]
    payoffs_chips: list[StrictInt]
    awards: list[StrictInt]
    # Only confirmed non-folded contenders in a completed multiway showdown.
    showdown_cards: list[TournamentShowdownSeat] = Field(default_factory=list)
    action_history: list[TournamentActionEvent]


class BlindProgressView(StrictModel):
    hands_per_level: StrictInt
    hand_in_level: StrictInt | None
    hands_to_next: StrictInt | None
    next_small_blind: StrictInt | None
    next_big_blind: StrictInt | None
    final_level: bool


class TournamentView(StrictModel):
    game_type: Literal["TOURNAMENT_6MAX"]
    lineup: Literal["baseline_5", "improved_4", "trained_1_improved_4"]
    revision: StrictInt
    phase: Literal["playing", "complete"]
    tournament_over: bool
    winner_seat: StrictInt | None
    human_eliminated: bool
    hand_number: StrictInt
    hands_completed: StrictInt
    alive_seats: list[StrictInt]
    tournament_stacks: list[StrictInt]
    blind_progress: BlindProgressView
    level_index: StrictInt
    small_blind: StrictInt
    big_blind: StrictInt
    button: StrictInt
    small_blind_seat: StrictInt | None
    big_blind_seat: StrictInt | None
    street: str | None
    board: list[str]
    hero_cards: list[str]
    seats: list[TournamentSeatView]
    total_chips: StrictInt
    pot: StrictInt
    to_act: StrictInt | None
    your_turn: bool
    to_call: StrictInt | None
    min_raise_to: StrictInt | None
    max_raise_to: StrictInt | None
    can_raise_to: bool
    actions: list[TournamentActionOffer]
    action_log: list[TournamentActionEvent]
    last_hand: TournamentHandView | None
    starting_stacks: list[StrictInt]


class TournamentGameResponse(StrictModel):
    session_id: str
    state: TournamentView


class TournamentHistoryResponse(StrictModel):
    total: StrictInt
    offset: StrictInt
    limit: StrictInt
    hands: list[TournamentHandView]


class TournamentAgentStatus(StrictModel):
    configured: bool
    file_found: bool
    checkpoint_name: str | None
    format: str
    model_family: str
    device: str
    mode: str
    validated: bool


class TournamentHealthResponse(StrictModel):
    status: Literal["ok"]
    service: Literal["tournament-env-local"]
    phase: Literal["1.1"]

````````

<!-- END_PY_SOURCE path="backend/tournament_schemas.py" -->


## PY-007 — `main.py`

**Blob SHA:** `2c3e05d604c3ad2adc8dd869a7a58a11a0f2e555` · **Original bytes:** 1480

<!-- BEGIN_PY_SOURCE path="main.py" sha="2c3e05d604c3ad2adc8dd869a7a58a11a0f2e555" -->

````````python
"""Play independent heads-up hands with corrected positions by default."""
import argparse

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine


def main():
    parser = argparse.ArgumentParser(description='holdem-ai corrected HUNL smoke test')
    parser.add_argument('--hands', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stack', type=int, default=200)
    parser.add_argument('--rules', choices=['corrected', 'upstream'], default='corrected')
    args = parser.parse_args()
    if args.hands < 1:
        parser.error('--hands must be positive')
    engine = HandEngine(starting_stack=args.stack, seed=args.seed, rules_mode=args.rules)
    agents = (RandomAgent(seed=args.seed), RandomAgent(seed=args.seed + 1))
    total = 0.0
    print(f'HUNL rules={args.rules}, hands={args.hands}, stack={args.stack}, BB={engine.big_blind}')
    for hand in range(1, args.hands + 1):
        result = engine.play_hand(agents)
        total += result.payoffs_bb[0]
        print(f'Hand {hand:>3}: P0={result.payoffs_bb[0]:+7.2f} BB, '
              f'P1={result.payoffs_bb[1]:+7.2f} BB, actions={len(result.action_history)}')
    print(f'P0 observed bb/100: {100 * total / args.hands:+.2f} (not a performance estimate for tiny samples)')
    print('This is a development baseline; not certified for real-money or tournament use.')


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="main.py" -->


## PY-008 — `scripts/audit_corrected.py`

**Blob SHA:** `5632af29cb4567963a69d0b3b54e50b5e3468c2f` · **Original bytes:** 1903

<!-- BEGIN_PY_SOURCE path="scripts/audit_corrected.py" sha="5632af29cb4567963a69d0b3b54e50b5e3468c2f" -->

````````python
"""uv run python scripts/audit_corrected.py --hands 100 --seed 42"""
import argparse

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine
from holdem_ai.environment.rules_audit import audit_preflop, summarize


def main():
    parser = argparse.ArgumentParser(description='Strict rule checks for corrected HUNL adapter')
    parser.add_argument('--hands', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.hands < 1:
        parser.error('--hands must be positive')
    engine = HandEngine(seed=args.seed)
    for button in (0, 1):
        engine.env.game.dealer_id = button
        engine.env.reset()
        checks = audit_preflop(engine.env.game)
        print(f'BUTTON={button}')
        print(summarize(checks))
        if not all(c.passed for c in checks):
            return 1
    for button in (0, 1):
        check_engine = HandEngine(seed=args.seed, button=button)
        _, player = check_engine.reset()
        if player != button:
            raise AssertionError('Button must act first preflop')
        check_engine.step(1)
        check_engine.step(1)
        assert check_engine.env.game.stage.name == 'FLOP'
        assert check_engine.current_player == 1 - button
        print(f'PASS bb_acts_first_postflop (button={button})')

    agents = (RandomAgent(seed=args.seed), RandomAgent(seed=args.seed + 1))
    for h in range(args.hands):
        result = engine.play_hand(agents)
        if sum(result.payoffs_chips) != 0:
            raise AssertionError(f'Zero-sum violation on hand {h+1}')
    print(f'PASS {args.hands}/{args.hands} random hands: zero-sum and no runtime error')
    print('STRICT RESULT: PASS (covered scenarios only; not a complete poker rules certification)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

````````

<!-- END_PY_SOURCE path="scripts/audit_corrected.py" -->


## PY-009 — `scripts/audit_multiplayer.py`

**Blob SHA:** `e8a309997f28705d4f9b3545de1512ee8134a214` · **Original bytes:** 1888

<!-- BEGIN_PY_SOURCE path="scripts/audit_multiplayer.py" sha="e8a309997f28705d4f9b3545de1512ee8134a214" -->

````````python
"""Quick random-benchmark smoke audit. Does not certify poker rules."""
from __future__ import annotations

import argparse
from collections import Counter

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', type=int, default=6)
    parser.add_argument('--hands', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 2 <= args.players <= 6 or args.hands <= 0:
        parser.error('players must be 2..6 and hands > 0')
    showdown = 0
    wins = Counter()
    total_actions = 0
    for h in range(args.hands):
        engine = MultiPlayerHandEngine(num_players=args.players, seed=args.seed + h,
                                       button=h % args.players)
        agents = [RandomAgent(args.seed + h * 100 + i) for i in range(args.players)]
        result = engine.play_hand(agents)
        assert sum(result.payoffs_chips) == 0
        assert sum(result.final_stacks) == args.players * 200
        assert min(result.final_stacks) >= 0
        assert sum(result.contributions) == sum(result.awards)
        showdown += len(result.board) == 5
        total_actions += len(result.action_history)
        if max(result.payoffs_chips) > 0:
            for i, p in enumerate(result.payoffs_chips):
                if p == max(result.payoffs_chips):
                    wins[i] += 1
    print(f'PASS: {args.hands}/{args.hands} hands, {args.players} seats, seed={args.seed}')
    print(f'Showdowns: {showdown} | Total actions: {total_actions}')
    print(f'Max-payoff seat counts (ties counted for all): {dict(wins)}')
    print('Scope: chip conservation and runtime smoke check; not a full rules certification.')


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/audit_multiplayer.py" -->


## PY-010 — `scripts/audit_rlcard.py`

**Blob SHA:** `86c772c7a70206d9830cf41b41ece905c5f0bf6a` · **Original bytes:** 1978

<!-- BEGIN_PY_SOURCE path="scripts/audit_rlcard.py" sha="86c772c7a70206d9830cf41b41ece905c5f0bf6a" -->

````````python
"""Run against installed RLCard: uv run python scripts/audit_rlcard.py --hands 100.

Exits 1 on rule violations, distinguishing strict audit from green pytest unit
suite. Does not change the original RLCard engine or the Phase-1 HandEngine.
"""

import argparse
import sys

import rlcard

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine
from holdem_ai.environment.rules_audit import audit_preflop, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description='Diagnostic audit of RLCard HUNL, not a repair tool')
    parser.add_argument('--hands', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.hands < 1:
        parser.error('--hands must be positive')

    raw = rlcard.make('no-limit-holdem', config={
        'seed': args.seed, 'game_num_players': 2, 'chips_for_each': 200,
    })
    raw.reset()
    checks = audit_preflop(raw.game)
    print('=== RLCard 1.2.0: STRICT HEADS-UP PREFLOP AUDIT ===')
    print(summarize(checks))

    engine = HandEngine(starting_stack=200, seed=args.seed)
    agents = (RandomAgent(seed=args.seed), RandomAgent(seed=args.seed + 1))
    failures = 0
    for idx in range(args.hands):
        try:
            result = engine.play_hand(agents)
            if abs(sum(result.payoffs_chips)) > 1e-6:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f'ERROR hand={idx + 1}: {type(exc).__name__}: {exc}')
    print(f'PAYOFF: {args.hands - failures}/{args.hands} hands passed zero-sum/runtime check')
    print('Note: payoff checks do NOT establish correct betting, showdown or side-pot rules.')
    issues = sum(not check.passed for check in checks) + failures
    print(f'STRICT RESULT: {"PASS" if issues == 0 else "FAIL"} ({issues} checks/issues)')
    return 1 if issues else 0


if __name__ == '__main__':
    sys.exit(main())

````````

<!-- END_PY_SOURCE path="scripts/audit_rlcard.py" -->


## PY-011 — `scripts/audit_tournament.py`

**Blob SHA:** `7f36522cc6b27bd9bd10b91647a753e29dbb5262` · **Original bytes:** 2399

<!-- BEGIN_PY_SOURCE path="scripts/audit_tournament.py" sha="7f36522cc6b27bd9bd10b91647a753e29dbb5262" -->

````````python
"""Single-table tournament smoke audit; run `uv run python scripts/audit_tournament.py`."""
from __future__ import annotations

import argparse
from collections import Counter

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.tournament_env import TournamentEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', type=int, default=6)
    parser.add_argument('--tournaments', type=int, default=100)
    parser.add_argument('--stack', type=int, default=1000)
    parser.add_argument('--hands-per-level', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 2 <= args.players <= 6 or args.tournaments <= 0 or args.stack <= 0:
        parser.error('players must be 2..6; tournaments and stack must be positive')
    wins = Counter()
    durations = []
    total_hands = 0
    total_eliminations = 0
    for t in range(args.tournaments):
        env = TournamentEnv(num_players=args.players, starting_stack=args.stack,
                            hands_per_level=args.hands_per_level, seed=args.seed + t)
        agents = [RandomAgent(args.seed + 1_000 * t + i) for i in range(args.players)]
        outcome = env.play_tournament(agents)
        assert sum(outcome.final_stacks) == args.players * args.stack
        assert outcome.final_stacks[outcome.winner] == args.players * args.stack
        assert sum(1 for x in outcome.final_stacks if x > 0) == 1
        assert len(outcome.history) == outcome.hands_played
        assert sum(len(record.eliminated) for record in outcome.history) == args.players - 1
        assert all(sum(r.hand_result.payoffs_chips) == 0 for r in outcome.history)
        wins[outcome.winner] += 1
        durations.append(outcome.hands_played)
        total_hands += outcome.hands_played
        total_eliminations += args.players - 1
    print(f'PASS: {args.tournaments}/{args.tournaments} tournaments, {args.players} seats, seed={args.seed}')
    print(f'Hands: {total_hands} | Mean hands/tournament: {sum(durations) / len(durations):.2f}')
    print(f'Eliminations: {total_eliminations} | Winner seats: {dict(sorted(wins.items()))}')
    print('Scope: termination, stack continuity, seat elimination and chip conservation.')
    print('Not a complete rules/side-pot/ICM certification.')


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/audit_tournament.py" -->


## PY-012 — `scripts/expand_phase01_smoke.py`

**Blob SHA:** `baa7aeeca906dd9e3799e46dbdf4819260093f8f` · **Original bytes:** 5137

<!-- BEGIN_PY_SOURCE path="scripts/expand_phase01_smoke.py" sha="baa7aeeca906dd9e3799e46dbdf4819260093f8f" -->

````````python
"""Real-engine Expand Phase 1 acceptance smoke: each table target 2..6 seats.

Usage from repository root:
  uv run --extra dev python -m scripts.expand_phase01_smoke --hands-per-n 1000
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import time
from holdem_ai.expand.environment import Action, DynamicTable, TableConfig


def run(*, hands_per_n: int, seed: int, output: str | None = None,
        journal_directory: str | None = None) -> dict:
    if type(hands_per_n) is not int or hands_per_n <= 0:
        raise ValueError('hands-per-n must be a positive integer')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be nonnegative integer')
    rng = random.Random(seed)
    stats = []
    start = time.monotonic()
    for max_seated in range(2, 7):
        filename = Path(journal_directory) / f'e01_{max_seated}.jsonl' if journal_directory else None
        table = DynamicTable(TableConfig(), session_id=f'e01_{seed}_{max_seated}', journal_path=filename)
        next_id = 0
        hands = 0  # hands played with exactly max_seated participants
        warmup_hands = 0
        decisions = 0
        busts = 0
        raises = 0
        joins = 0
        attempts = 0
        while hands < hands_per_n:
            attempts += 1
            if attempts > hands_per_n * 30 + 500:
                raise RuntimeError(f'could not open {hands_per_n} hands for {max_seated} seats')
            occupied = sum(s.player_id is not None for s in table.snapshot().seats)
            for _ in range(max_seated - occupied):
                pid = f'p{next_id:08d}'
                next_id += 1
                buyin = rng.randint(20, 300) * 100
                table.request_join(pid, buyin)
                joins += 1
            hand_seed = rng.getrandbits(63)
            hand = table.start_hand(hand_seed)
            if hand is None:
                continue
            for _ in range(250):
                player_id = table.current_player_id
                if player_id is None:
                    break
                raw = table.observe_current()
                legal = raw['legal_actions']
                to_call = raw['raw_obs']['to_call']
                draw = rng.random()
                if 4 in legal and draw < 0.008:
                    action = Action('all_in')
                    raises += 1
                elif 3 in legal and draw < 0.06:
                    action = Action('raise_pot')
                    raises += 1
                elif 2 in legal and draw < 0.10:
                    action = Action('raise_half_pot')
                    raises += 1
                elif to_call and draw < 0.18:
                    action = Action('fold')
                else:
                    action = Action('call' if to_call else 'check')
                table.act(player_id, action)
                decisions += 1
            else:
                raise RuntimeError('250 decisions exceeded in one hand')
            settled = table.settle_hand()
            if sum(settled.payoffs_chips) != 0 or sum(settled.final_stacks) != sum(settled.starting_stacks):
                raise AssertionError('hand chip invariant violated')
            busts += len(settled.busted_player_ids)
            if len(hand.participants) == max_seated:
                hands += 1
            else:
                warmup_hands += 1
        stats.append({
            'target_seats': max_seated, 'hands': hands, 'warmup_hands': warmup_hands,
            'all_hands': hands + warmup_hands, 'decisions': decisions,
            'join_requests': joins, 'busts': busts, 'raises': raises,
            'final_table_chips': table.snapshot().total_chips,
            'buyins_minus_cashouts': table.total_buyins - table.total_cashouts,
        })
    result = {'expand_phase': 1, 'mode': 'cash', 'hands_per_target': hands_per_n,
              'targets': [2, 3, 4, 5, 6], 'target_hands': sum(x['hands'] for x in stats),
              'total_hands': sum(x['all_hands'] for x in stats),
              'seed': seed, 'elapsed_seconds': round(time.monotonic()-start, 3),
              'source': 'real Base Phase 3 MultiPlayerHandEngine', 'results': stats}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite prior experiments.
        with path.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
    return result


def main():
    p = argparse.ArgumentParser(description='Expand Phase 1 real 2..6 player hand smoke')
    p.add_argument('--hands-per-n', type=int, default=10)
    p.add_argument('--seed', type=int, default=170001)
    p.add_argument('--output-json', default=None)
    p.add_argument('--journal-directory', default=None)
    args = p.parse_args()
    print(json.dumps(run(hands_per_n=args.hands_per_n, seed=args.seed,
                         output=args.output_json, journal_directory=args.journal_directory),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/expand_phase01_smoke.py" -->
