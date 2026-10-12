"""Expand Phase 2: allowlisted, immutable acting-player-only information.

No table engine object, seed, opponents' hole cards, deck or critic state is
stored in these types. All amounts are publicly visible chip/BB units.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

STREETS = ('preflop', 'flop', 'turn', 'river')
ACTION_NAMES = ('fold', 'check', 'call', 'raise_half_pot', 'raise_pot', 'raise_double_pot', 'all_in')


@dataclass(frozen=True)
class PublicSeatFeatures:
    occupied: bool
    dealt_in: bool
    folded: bool
    all_in: bool
    seat_status: str  # hand-specific: active/folded/allin/out
    table_status: str  # cash session: seated/waiting_bb/sitting_out/etc
    stack_bb: float
    invested_bb: float
    street_bet_bb: float
    relative_position: int
    is_hero: bool


@dataclass(frozen=True)
class PublicAction:
    actor_seat: int
    street: Literal['preflop', 'flop', 'turn', 'river']
    kind: str
    raise_to_bb: float | None
    to_call_bb: float
    pot_bb: float


@dataclass(frozen=True)
class LegalActionSpec:
    mask: tuple[bool, ...]  # ACTION_NAMES order
    raise_to_chips: tuple[int | None, ...]
    min_raise_to_chips: int
    max_raise_to_chips: int
    to_call_chips: int

    def __post_init__(self) -> None:
        if len(self.mask) != len(ACTION_NAMES) or len(self.raise_to_chips) != len(ACTION_NAMES):
            raise ValueError('action mask/targets must have seven slots')
        if not any(self.mask):
            raise ValueError('at least one legal action required')

    def index(self, action_name: str) -> int:
        if action_name not in ACTION_NAMES:
            raise ValueError(f'unknown action {action_name}')
        return ACTION_NAMES.index(action_name)

    def allows(self, action_name: str) -> bool:
        return self.mask[self.index(action_name)]


@dataclass(frozen=True)
class InformationState:
    hero_seat: int
    hero_hole: tuple[str, str]
    board: tuple[str, ...]
    street: Literal['preflop', 'flop', 'turn', 'river']
    button_seat: int
    seats: tuple[PublicSeatFeatures, ...]  # six physical seats
    action_history: tuple[PublicAction, ...]
    legal_actions: LegalActionSpec
    pot_bb: float
    to_call_bb: float
    big_blind_chips: int
    num_dealt_in: int

    def __post_init__(self) -> None:
        if len(self.seats) != 6 or not 0 <= self.hero_seat < 6:
            raise ValueError('six seats and valid hero seat required')
        if not self.seats[self.hero_seat].dealt_in:
            raise ValueError('acting hero must be dealt in')


@dataclass(frozen=True)
class TrainingOnlyCriticState:
    """Explicitly privileged state, cannot be passed as an InformationState.

    Not produced by an actor-facing E2 factory or encoder. Phase 3 may build
    this in a separate trusted training-only pipeline.
    """
    full_hand_state: object
