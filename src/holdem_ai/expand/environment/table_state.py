"""Public, immutable table metadata; does not contain hole cards or hidden deck."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TableConfig:
    mode: Literal['cash', 'tournament'] = 'cash'
    small_blind: int = 50
    big_blind: int = 100
    max_seats: int = 6
    min_buyin_bb: int = 20
    max_buyin_bb: int = 300
    rake_bps: int = 0
    seating_rule: str = 'moving_button'
    join_rule: str = 'wait_for_big_blind'

    def __post_init__(self):
        if self.mode not in ('cash', 'tournament'):
            raise ValueError('invalid table mode')
        if self.max_seats != 6:
            raise ValueError('Expand v1 requires exactly six physical seats')
        if type(self.small_blind) is not int or type(self.big_blind) is not int or not 0 < self.small_blind < self.big_blind:
            raise ValueError('invalid blind configuration')
        if (type(self.min_buyin_bb) is not int or type(self.max_buyin_bb) is not int
                or not 1 <= self.min_buyin_bb <= self.max_buyin_bb):
            raise ValueError('invalid buyin range')
        if type(self.rake_bps) is not int or self.rake_bps != 0:
            raise ValueError('E1 supports no rake; use rake_bps=0')
        if self.seating_rule != 'moving_button' or self.join_rule != 'wait_for_big_blind':
            raise ValueError('unsupported E1 button or entry policy')


@dataclass(frozen=True)
class PlayerSeat:
    seat_id: int
    player_id: str | None
    stack_chips: int
    seat_status: str


@dataclass(frozen=True)
class TableSnapshot:
    config: TableConfig
    seats: tuple[PlayerSeat, ...]
    button_seat: int | None
    hand_number: int
    session_id: str
    hand_in_progress: bool
    current_actor_seat: int | None
    pending_event_count: int
    total_chips: int


@dataclass(frozen=True)
class Action:
    kind: Literal['fold', 'check', 'call', 'raise_to', 'all_in', 'raise_half_pot', 'raise_pot']
    raise_to_chips: int | None = None

    def __post_init__(self):
        if self.kind not in ('fold', 'check', 'call', 'raise_to', 'all_in', 'raise_half_pot', 'raise_pot'):
            raise ValueError('invalid action')
        if self.kind == 'raise_to':
            if type(self.raise_to_chips) is not int or self.raise_to_chips <= 0:
                raise ValueError('raise_to needs positive total street wager')
        elif self.raise_to_chips is not None:
            raise ValueError('only raise_to accepts raise_to_chips')


@dataclass(frozen=True)
class HandHandle:
    hand_id: str
    hand_number: int
    button_seat: int
    small_blind_seat: int
    big_blind_seat: int
    participants: tuple[tuple[int, str], ...]
    seed: int
    terminal: bool


@dataclass(frozen=True)
class StepEvent:
    hand_id: str
    actor_seat: int
    player_id: str
    action: Action
    terminal: bool
    next_actor_seat: int | None


@dataclass(frozen=True)
class Settlement:
    hand_id: str
    hand_number: int
    starting_stacks: tuple[int, ...]
    final_stacks: tuple[int, ...]
    payoffs_chips: tuple[int, ...]
    payoffs_bb: tuple[float, ...]
    busted_player_ids: tuple[str, ...]
    button_seat: int
