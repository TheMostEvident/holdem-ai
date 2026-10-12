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
