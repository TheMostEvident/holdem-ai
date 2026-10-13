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
