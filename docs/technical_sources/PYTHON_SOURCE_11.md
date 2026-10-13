# Python full source volume 11/28

[Canonical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) — original commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`; exact code with individual blob SHA.

## src/holdem_ai/expand/environment/table_state.py

Original SHA `3db9e6111750f7de67d545433d134915527b35cf` (3204 bytes).

<a name="py-121"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/environment/table_state.py" sha="3db9e6111750f7de67d545433d134915527b35cf" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/environment/table_state.py" -->

## src/holdem_ai/expand/environment/tournament_rules.py

Original SHA `472d24273ddf933a7233c8757b318f2d7680ef4a` (589 bytes).

<a name="py-122"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/environment/tournament_rules.py" sha="472d24273ddf933a7233c8757b318f2d7680ef4a" -->

````````python
"""Protocol boundary for E7; E1 deliberately does not simulate tournaments."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TournamentRuleSet:
    late_registration_enabled: bool = False
    reentry_enabled: bool = False
    ante_chips: int = 0

    def __post_init__(self):
        if type(self.ante_chips) is not int or self.ante_chips < 0:
            raise ValueError('ante must be nonnegative integer')

    def require_implemented(self):
        raise NotImplementedError('Tournament operation is reserved for Expand Phase 7')

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/environment/tournament_rules.py" -->

## src/holdem_ai/expand/information/__init__.py

Original SHA `8e514d720d57c6303812fbd1f4dee1fefc48ba8f` (1010 bytes).

<a name="py-123"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/information/__init__.py" sha="8e514d720d57c6303812fbd1f4dee1fefc48ba8f" -->

````````python
"""Expand Phase 2 acting-player information, action translation and encoding."""
from .schema import (PublicSeatFeatures, PublicAction, LegalActionSpec,
                     InformationState, TrainingOnlyCriticState, ACTION_NAMES)
from .builder import make_information_state
from .actions import legal_action_spec, to_table_action
from .encoder import (EncodedObservation, encode_observation, SEAT_DIM,
                      GLOBAL_DIM, HISTORY_DIM, MAX_HISTORY)
from .adapter import Phase2TableAdapter

__all__ = [
    'PublicSeatFeatures', 'PublicAction', 'LegalActionSpec', 'InformationState',
    'TrainingOnlyCriticState', 'ACTION_NAMES', 'make_information_state',
    'legal_action_spec', 'to_table_action', 'EncodedObservation',
    'encode_observation', 'SEAT_DIM', 'GLOBAL_DIM', 'HISTORY_DIM',
    'MAX_HISTORY', 'Phase2TableAdapter', 'PolicyV2',
]


def __getattr__(name: str):
    if name == 'PolicyV2':
        from .policy_v2 import PolicyV2
        return PolicyV2
    raise AttributeError(name)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/information/__init__.py" -->

## src/holdem_ai/expand/information/actions.py

Original SHA `bb8cc8a5352cace8b9c6e736ab0bbfb428c8c1a0` (3375 bytes).

<a name="py-124"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/information/actions.py" sha="bb8cc8a5352cace8b9c6e736ab0bbfb428c8c1a0" -->

````````python
"""Exact legality-preserving discrete action translation, with no silent clamp."""
from __future__ import annotations
from .schema import ACTION_NAMES, LegalActionSpec
from holdem_ai.expand.environment.table_state import Action


def legal_action_spec(raw: dict, legal_ids: tuple[int, ...], bb: int) -> LegalActionSpec:
    """Translate Phase 3 legal IDs and exact public raise-to bounds.

    The Phase3 ALL_IN (4) legality is the *raise* capability marker. A short
    stack can call all-in via ID 1; we intentionally represent that as CALL to
    avoid duplicated identical actions in the policy categorical distribution.
    """
    if type(bb) is not int or bb <= 0:
        raise ValueError('positive big blind required')
    for key in ('pot', 'to_call', 'min_raise_to', 'street_bets', 'stacks', 'player_id'):
        if key not in raw:
            raise ValueError(f'missing engine observation key: {key}')
    player = raw['player_id']
    if type(player) is not int or not 0 <= player < 6:
        raise ValueError('bad player_id')
    bets, stacks = raw['street_bets'], raw['stacks']
    if len(bets) != 6 or len(stacks) != 6:
        raise ValueError('six stack/bet entries required')
    if any(type(x) is not int or x < 0 for x in (*bets, *stacks)):
        raise ValueError('invalid public chip values')
    b, s = bets[player], stacks[player]
    pot, to_call, minimum = (raw[x] for x in ('pot', 'to_call', 'min_raise_to'))
    if any(type(v) is not int or v < 0 for v in (pot, to_call, minimum)):
        raise ValueError('invalid pot/call/minimum')
    maximum = b + s
    high = b + to_call
    ids = set(legal_ids)
    if any(type(i) is not int or i not in range(5) for i in ids) or not (ids & {0, 1}):
        raise ValueError('unsupported engine legal ID set')
    allowed = [False] * len(ACTION_NAMES)
    targets: list[int | None] = [None] * len(ACTION_NAMES)
    allowed[0] = 0 in ids
    allowed[1] = (1 in ids and to_call == 0)
    allowed[2] = (1 in ids and to_call > 0)
    for index, factor, engine_id in ((3, 1, 2), (4, 2, 3), (5, 4, None)):
        # exact integer math: floor(k * (pot + to_call)) where k is 1/2/2.
        post_call = pot + to_call
        amount = (post_call * factor) // 2
        target = b + to_call + amount
        permitted = (4 in ids and target > high and minimum <= target <= maximum)
        if engine_id is not None:
            permitted = permitted and engine_id in ids
        # Distinct legal actions, so no duplicate wager-size decisions.
        if permitted and target not in targets:
            allowed[index], targets[index] = True, target
    # ID 4 is raise-only. All-in CALL is canonicalized as CALL (ID 1).
    if 4 in ids and maximum > high and maximum not in targets:
        allowed[6], targets[6] = True, maximum
    return LegalActionSpec(tuple(allowed), tuple(targets), minimum, maximum, to_call)


def to_table_action(spec: LegalActionSpec, name: str) -> Action:
    ix = spec.index(name)
    if not spec.mask[ix]:
        raise ValueError(f'{name} illegal for current observation')
    if name in ('fold', 'check', 'call'):
        return Action(name)
    if name == 'all_in':
        return Action('all_in')
    # Always use exact raise_to target; no clamp, no rounding in the engine.
    target = spec.raise_to_chips[ix]
    assert target is not None
    return Action('raise_to', target)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/information/actions.py" -->

## src/holdem_ai/expand/information/adapter.py

Original SHA `33b71451e421baaa7b5ea78ce728bf9f395d3655` (3795 bytes).

<a name="py-125"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/information/adapter.py" sha="33b71451e421baaa7b5ea78ce728bf9f395d3655" -->

````````python
"""Phase2 interface for Phase1 DynamicTable; records public action street exactly.

Table is still the sole game-rule authority. This wrapper never changes its
live hand engine. A wrapper attached mid-hand fails closed rather than forging
an action history with the wrong street.
"""
from __future__ import annotations
from .builder import make_information_state
from .encoder import encode_observation
from .actions import to_table_action
from .schema import InformationState, PublicAction
from holdem_ai.expand.environment.table_state import Action


class Phase2TableAdapter:
    def __init__(self, table):
        self.table = table
        if table.snapshot().hand_in_progress:
            raise ValueError('attach Phase2TableAdapter before start_hand, not mid-hand')
        self._history: list[PublicAction] = []
        self._hand_id: str | None = None

    def start_hand(self, seed: int):
        if self.table.snapshot().hand_in_progress:
            raise RuntimeError('cannot start a second hand')
        result = self.table.start_hand(seed)
        self._history.clear()
        self._hand_id = result.hand_id if result is not None else None
        return result

    def observe(self) -> InformationState:
        snap = self.table.snapshot()
        if not snap.hand_in_progress or snap.current_actor_seat is None:
            raise RuntimeError('no acting player; cannot form observation')
        obs = self.table.observe_current()
        if obs['raw_obs']['player_id'] != snap.current_actor_seat:
            raise AssertionError('acting player differs from table snapshot')
        return make_information_state(
            obs, big_blind_chips=snap.config.big_blind,
            seat_presence=tuple(s.player_id is not None for s in snap.seats),
            seat_statuses=tuple(s.seat_status for s in snap.seats),
            history=tuple(self._history),
        )

    def encode(self):
        return encode_observation(self.observe())

    def act_abstract(self, name: str):
        state = self.observe()
        return self.act(to_table_action(state.legal_actions, name))

    def act(self, action: Action):
        if not isinstance(action, Action):
            raise TypeError('required phase1 Action')
        before = self.observe()
        bb = before.big_blind_chips
        raw_kind = action.kind
        if raw_kind == 'raise_half_pot':
            # Integer exact same as engine, not floating approximation.
            raw = self.table.observe_current()['raw_obs']
            target_chips = raw['street_bets'][before.hero_seat] + raw['to_call'] + (raw['pot']+raw['to_call'])//2
        elif raw_kind == 'raise_pot':
            raw = self.table.observe_current()['raw_obs']
            target_chips = raw['street_bets'][before.hero_seat] + raw['to_call'] + (raw['pot']+raw['to_call'])
        elif raw_kind == 'raise_to':
            target_chips = action.raise_to_chips
        elif raw_kind == 'all_in':
            raw = self.table.observe_current()['raw_obs']
            target_chips = raw['street_bets'][before.hero_seat] + raw['stacks'][before.hero_seat]
        else:
            target_chips = None
        player_id = self.table.current_player_id
        # Mutate underlying hand FIRST; never track rejected/illegal actions.
        step = self.table.act(player_id, action)
        event = PublicAction(
            actor_seat=before.hero_seat, street=before.street, kind=raw_kind,
            raise_to_bb=(target_chips/bb if target_chips is not None else None),
            to_call_bb=before.to_call_bb, pot_bb=before.pot_bb,
        )
        self._history.append(event)
        return step

    def settle_hand(self):
        result = self.table.settle_hand()
        self._hand_id = None
        self._history.clear()
        return result

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/information/adapter.py" -->

## src/holdem_ai/expand/information/builder.py

Original SHA `3793e2d33596785131062b870bbf7afa97552e9b` (6109 bytes).

<a name="py-126"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/information/builder.py" sha="3793e2d33596785131062b870bbf7afa97552e9b" -->

````````python
"""Construct an InformationState only from acting-player public observation.

Never touch hand_engine.hole_cards / _deck / _rng / journal or training critic.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence

from .actions import legal_action_spec
from .schema import InformationState, PublicAction, PublicSeatFeatures, STREETS

_RANKS = '23456789TJQKA'
_SUITS = 'cdhs'
_ALLOWED_STATUS = frozenset({'active', 'folded', 'allin', 'out'})


def _validate_cards(cards: Sequence[str], lengths: tuple[int, ...], label: str) -> tuple[str, ...]:
    if not isinstance(cards, (tuple, list)) or len(cards) not in lengths:
        raise ValueError(f'invalid {label} length')
    out = tuple(cards)
    if any(type(c) is not str or len(c) != 2 or c[0] not in _SUITS or c[1] not in _RANKS for c in out):
        raise ValueError(f'invalid {label} card')
    if len(set(out)) != len(out):
        raise ValueError('duplicate cards in observation')
    return out


def make_information_state(
    observation: Mapping[str, Any], *,
    big_blind_chips: int,
    seat_presence: Sequence[bool],
    seat_statuses: Sequence[str] | None = None,
    history: Sequence[PublicAction] = (),
) -> InformationState:
    """Allowlist-only observation factory; session seed and opponent cards ignored.

    seat_presence comes from TableSnapshot.seats[].player_id, NOT from engine
    status. A sitting-out player is occupied but not dealt into this hand.
    """
    if type(big_blind_chips) is not int or big_blind_chips <= 0:
        raise ValueError('positive big blind required')
    if len(seat_presence) != 6 or any(type(v) is not bool for v in seat_presence):
        raise ValueError('six boolean seat presence flags required')
    if seat_statuses is None:
        seat_statuses = tuple('occupied' if present else 'vacant' for present in seat_presence)
    if len(seat_statuses) != 6 or any(type(v) is not str for v in seat_statuses):
        raise ValueError('six public table status strings required')
    for occupied, table_status in zip(seat_presence, seat_statuses):
        if (not occupied and table_status != 'vacant') or (occupied and table_status == 'vacant'):
            raise ValueError('table seat presence/status mismatch')
    if 'raw_obs' not in observation or 'legal_actions' not in observation:
        raise ValueError('acting-player observation missing')
    raw = observation['raw_obs']
    if not isinstance(raw, Mapping):
        raise ValueError('raw_obs must be mapping')
    required = ('hand','public_cards','street','button','player_id','stacks','street_bets',
                'total_contributions','status','pot','to_call','min_raise_to','action_history',
                'action_raise_to')
    if any(key not in raw for key in required):
        raise ValueError('incomplete engine observation: ' + ','.join(k for k in required if k not in raw))
    hero, button = raw['player_id'], raw['button']
    if any(type(x) is not int or not 0 <= x < 6 for x in (hero, button)):
        raise ValueError('invalid physical seat')
    street = raw['street']
    if street not in STREETS:
        raise ValueError('unknown street')
    hole = _validate_cards(raw['hand'], (2,), 'hero_hole')
    board = _validate_cards(raw['public_cards'], (0, 3, 4, 5), 'public_board')
    if len(set(hole + board)) != len(hole) + len(board):
        raise ValueError('hero hole and board overlap')
    if (street == 'preflop' and len(board) != 0 or street == 'flop' and len(board) != 3
            or street == 'turn' and len(board) != 4 or street == 'river' and len(board) != 5):
        raise ValueError('street/board mismatch')
    statuses = raw['status']
    stacks, invested, street_bets = (raw['stacks'], raw['total_contributions'], raw['street_bets'])
    if any(len(v) != 6 for v in (statuses, stacks, invested, street_bets)):
        raise ValueError('six physical seat values required')
    if any(st not in _ALLOWED_STATUS for st in statuses):
        raise ValueError('unknown engine seat status')
    if any(type(x) is not int or x < 0 for seq in (stacks, invested, street_bets) for x in seq):
        raise ValueError('negative or noninteger chips')
    if sum(invested) != raw['pot'] or any(street_bets[i] > invested[i] for i in range(6)):
        raise ValueError('pot/contributions mismatch')
    if statuses[hero] != 'active' or not seat_presence[hero]:
        raise ValueError('acting hero must be a live occupied player')
    if any(statuses[i] != 'out' and not seat_presence[i] for i in range(6)):
        raise ValueError('dealt in seat missing from registry')
    visible_seats = tuple(PublicSeatFeatures(
        occupied=seat_presence[i], dealt_in=statuses[i] != 'out',
        folded=statuses[i] == 'folded', all_in=statuses[i] == 'allin',
        seat_status=statuses[i], table_status=seat_statuses[i],
        stack_bb=stacks[i]/big_blind_chips,
        invested_bb=invested[i]/big_blind_chips,
        street_bet_bb=street_bets[i]/big_blind_chips,
        relative_position=(i-button)%6, is_hero=(i==hero),
    ) for i in range(6))
    if not all(isinstance(h, PublicAction) for h in history):
        raise ValueError('public action history must be typed PublicAction')
    if len(history) != len(raw['action_history']) or len(history) != len(raw['action_raise_to']):
        raise ValueError('incomplete or mismatched public action recorder; fail closed')
    for record, action in zip(history, raw['action_history']):
        if record.actor_seat != action[0]:
            raise ValueError('recorded actor differs from engine history')
    legal_ids = tuple(observation['legal_actions'].keys())
    spec = legal_action_spec(raw, legal_ids, big_blind_chips)
    return InformationState(
        hero_seat=hero, hero_hole=hole, board=board, street=street, button_seat=button,
        seats=visible_seats, action_history=tuple(history), legal_actions=spec,
        pot_bb=raw['pot']/big_blind_chips, to_call_bb=raw['to_call']/big_blind_chips,
        big_blind_chips=big_blind_chips,
        num_dealt_in=sum(seat.dealt_in for seat in visible_seats),
    )

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/information/builder.py" -->

## src/holdem_ai/expand/information/encoder.py

Original SHA `21c581fd18c80dcbc16b9f1ff419306fb975d78a` (3985 bytes).

<a name="py-127"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/information/encoder.py" sha="21c581fd18c80dcbc16b9f1ff419306fb975d78a" -->

````````python
"""Stable, deterministic 6-seat tensor-ready encoding (Python only)."""
from __future__ import annotations
from dataclasses import dataclass
import math

from .schema import ACTION_NAMES, InformationState, STREETS

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
SEAT_DIM = 19
GLOBAL_DIM = 52 + 52 + 4 + 6
HISTORY_DIM = 6 + 4 + 8 + 3
MAX_HISTORY = 128
HISTORY_NAMES = ACTION_NAMES + ('raise_to',)


def card_index(card: str) -> int:
    if type(card) is not str or len(card) != 2 or card[0] not in SUITS or card[1] not in RANKS:
        raise ValueError('invalid canonical card')
    return SUITS.index(card[0])*13 + RANKS.index(card[1])


def _scale(x: float) -> float:
    # Preserve sign and prevent overflow in high-stack games; log compression.
    return math.copysign(math.log1p(abs(x)), x)


@dataclass(frozen=True)
class EncodedObservation:
    seat_features: tuple[tuple[float, ...], ...]  # 6 x SEAT_DIM
    seat_mask: tuple[bool, ...]  # True means occupied, even if sitting out
    global_features: tuple[float, ...]  # GLOBAL_DIM
    history_features: tuple[tuple[float, ...], ...]  # fixed 128 x HISTORY_DIM
    history_mask: tuple[bool, ...]  # True means valid
    legal_mask: tuple[bool, ...]  # 7 abstract actions


def encode_observation(state: InformationState) -> EncodedObservation:
    if not isinstance(state, InformationState):
        raise TypeError('actor encoder accepts InformationState only (never TrainingOnlyCriticState)')
    rows = []
    for seat in state.seats:
        status = seat.seat_status
        rows.append((
            float(seat.occupied), float(seat.dealt_in), float(seat.folded),
            float(seat.all_in), float(seat.is_hero),
            float(status == 'out'), float(status == 'active'),
            float(status == 'folded'), float(status == 'allin'),
            _scale(seat.stack_bb), _scale(seat.invested_bb),
            _scale(seat.street_bet_bb), seat.relative_position / 5.0,
            float(seat.dealt_in and not seat.folded and not seat.all_in),
            float(seat.occupied and not seat.dealt_in),
            float(seat.table_status == 'seated'),
            float(seat.table_status == 'waiting_bb'),
            float(seat.table_status == 'sitting_out'),
            float(seat.table_status in ('reserved', 'pending_leave')),
        ))
    hcard = [0.0]*52
    board = [0.0]*52
    for card in state.hero_hole:
        hcard[card_index(card)] = 1.0
    for card in state.board:
        board[card_index(card)] = 1.0
    street_one = [float(state.street == name) for name in STREETS]
    spec = state.legal_actions
    g = tuple(hcard + board + street_one + [
        _scale(state.pot_bb), _scale(state.to_call_bb),
        _scale(spec.min_raise_to_chips/state.big_blind_chips),
        _scale(spec.max_raise_to_chips/state.big_blind_chips),
        state.num_dealt_in/6.0,
        ((state.hero_seat-state.button_seat)%6)/5.0,
    ])
    history = []
    for action in state.action_history[-MAX_HISTORY:]:
        if action.kind not in HISTORY_NAMES:
            raise ValueError('unrecognized public history action')
        position = (action.actor_seat-state.button_seat)%6
        pos = [float(i==position) for i in range(6)]
        st = [float(action.street==s) for s in STREETS]
        kind = [float(action.kind==s) for s in HISTORY_NAMES]
        amounts = [_scale(action.raise_to_bb or 0.0), _scale(action.to_call_bb), _scale(action.pot_bb)]
        history.append(tuple(pos+st+kind+amounts))
    n = len(history)
    history.extend([(0.0,)*HISTORY_DIM]*(MAX_HISTORY-n))
    out = EncodedObservation(
        seat_features=tuple(rows), seat_mask=tuple(seat.occupied for seat in state.seats),
        global_features=g, history_features=tuple(history),
        history_mask=tuple([True]*n+[False]*(MAX_HISTORY-n)), legal_mask=spec.mask,
    )
    assert all(len(r)==SEAT_DIM for r in out.seat_features)
    assert len(g)==GLOBAL_DIM and all(len(r)==HISTORY_DIM for r in out.history_features)
    return out

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/information/encoder.py" -->

## src/holdem_ai/expand/information/policy_v2.py

Original SHA `59448a2d53329f90c9e6d812999d4b884c324890` (4370 bytes).

<a name="py-128"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/information/policy_v2.py" sha="59448a2d53329f90c9e6d812999d4b884c324890" -->

````````python
"""Small, trainable multi-seat neural policy. No training in Expand Phase 2.

Actor input is *only* EncodedObservation; illegal logits are always masked.
History is encoded by GRU over the most recent 128 public betting actions.
"""
from __future__ import annotations

import torch
from torch import nn

from .encoder import (EncodedObservation, GLOBAL_DIM, HISTORY_DIM,
                      SEAT_DIM, MAX_HISTORY)
from .schema import ACTION_NAMES


class PolicyV2(nn.Module):
    def __init__(self, *, width: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        if type(width) is not int or width < 16 or width % heads:
            raise ValueError('width must be >=16 and divisible by heads')
        self.width = width
        self.seat_project = nn.Sequential(nn.Linear(SEAT_DIM, width), nn.LayerNorm(width), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=width*2,
                                           dropout=0.0, batch_first=True)
        self.seat_attention = nn.TransformerEncoder(layer, num_layers=layers,
                                                    enable_nested_tensor=False)
        self.history_project = nn.Sequential(nn.Linear(HISTORY_DIM, width), nn.GELU())
        self.history_rnn = nn.GRU(width, width, num_layers=1, batch_first=True)
        self.policy_head = nn.Sequential(nn.Linear(width*2+GLOBAL_DIM, width), nn.GELU(),
                                         nn.Linear(width, len(ACTION_NAMES)))

    def forward(self, observations: list[EncodedObservation]):
        if not observations or any(not isinstance(x, EncodedObservation) for x in observations):
            raise TypeError('PolicyV2 accepts nonempty list[EncodedObservation] only')
        device = next(self.parameters()).device
        seats = torch.tensor([o.seat_features for o in observations], device=device, dtype=torch.float32)
        seat_valid = torch.tensor([o.seat_mask for o in observations], device=device, dtype=torch.bool)
        globals_ = torch.tensor([o.global_features for o in observations], device=device, dtype=torch.float32)
        hist = torch.tensor([o.history_features for o in observations], device=device, dtype=torch.float32)
        history_valid = torch.tensor([o.history_mask for o in observations], device=device, dtype=torch.bool)
        action_mask = torch.tensor([o.legal_mask for o in observations], device=device, dtype=torch.bool)
        if torch.any(~seat_valid.any(dim=1)) or torch.any(~action_mask.any(dim=1)):
            raise ValueError('at least one occupied seat and legal action required')
        encoded = self.seat_attention(self.seat_project(seats), src_key_padding_mask=~seat_valid)
        pooled = (encoded*seat_valid.unsqueeze(-1)).sum(1)/seat_valid.sum(1, keepdim=True)
        # Use pack_padded_sequence to exclude padding from the recurrent state.
        lengths = history_valid.sum(1)
        packed = nn.utils.rnn.pack_padded_sequence(self.history_project(hist),
                    lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, last = self.history_rnn(packed)
        h_summary = last[-1] * (lengths>0).unsqueeze(-1)
        logits = self.policy_head(torch.cat((pooled, h_summary, globals_), dim=-1))
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return masked_logits

    @torch.no_grad()
    def action_probabilities(self, observation: EncodedObservation) -> tuple[float, ...]:
        self.eval()
        probabilities = self.forward([observation]).softmax(dim=-1)[0]
        return tuple(probabilities.cpu().tolist())

    @torch.no_grad()
    def select_action(self, observation: EncodedObservation, *, rng_seed: int, greedy: bool = False) -> str:
        self.eval()
        if type(rng_seed) is not int or rng_seed < 0:
            raise ValueError('nonnegative rng_seed required')
        logits = self.forward([observation])[0]
        if greedy:
            selected = logits.argmax().item()
        else:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(rng_seed)
            # CPU generator works regardless of policy CUDA placement.
            probs = logits.softmax(-1).detach().cpu()
            selected = torch.multinomial(probs, 1, generator=gen).item()
        return ACTION_NAMES[selected]

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/information/policy_v2.py" -->

## src/holdem_ai/expand/information/schema.py

Original SHA `53cb3a912725871e88f44d063066828a139ffa71` (2902 bytes).

<a name="py-129"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/information/schema.py" sha="53cb3a912725871e88f44d063066828a139ffa71" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/information/schema.py" -->

## src/holdem_ai/expand/integration/__init__.py

Original SHA `db3bc99ab924faafdf224b3e3b1d48428f304869` (703 bytes).

<a name="py-130"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/integration/__init__.py" sha="db3bc99ab924faafdf224b3e3b1d48428f304869" -->

````````python
from .agent import IntegratedPokerAgent, RoutedDecision, RoutingConfig, from_e03_checkpoint, uniform_legal
from .tournament import (SingleTableTournament, TournamentConfig, TournamentHand,
                         TournamentReport, TournamentSettlement)
from .source_bridges import from_e04_checkpoint, from_e05_solver
from .manifest import release_manifest, write_new_json, sha256_file
__all__=['IntegratedPokerAgent','RoutedDecision','RoutingConfig','from_e03_checkpoint','uniform_legal',
         'SingleTableTournament','TournamentConfig','TournamentHand','TournamentReport','TournamentSettlement',
         'release_manifest','write_new_json','sha256_file','from_e04_checkpoint','from_e05_solver']

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/integration/__init__.py" -->

## src/holdem_ai/expand/integration/agent.py

Original SHA `d9c6df49b690b12df82eba45d188a6105134ef71` (7108 bytes).

<a name="py-131"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/integration/agent.py" sha="d9c6df49b690b12df82eba45d188a6105134ef71" -->

````````python
"""E7 routing: strictly actor-visible InformationState -> one legal Phase1 Action.

Policy sources are explicitly supplied/validated; neither this module nor E6
invent an exploitative policy. Prototype CFR/Search sources are opt-in and
incompatible abstractions MUST fall back instead of silently extrapolating.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
import math
import random

from holdem_ai.expand.information.schema import InformationState, ACTION_NAMES
from holdem_ai.expand.information.actions import to_table_action
from holdem_ai.expand.information.encoder import encode_observation
from holdem_ai.expand.opponent.adaptive import SafeAdaptivePolicy, normalize_policy
from holdem_ai.expand.opponent.router import StyleExpertRouter

PolicySource = Callable[[InformationState], Mapping[str, float]]


@dataclass(frozen=True)
class RoutingConfig:
    use_blueprint: bool = False
    use_search: bool = False
    use_adaptation: bool = False
    # A compatible abstraction+action mapping must be externally verified.
    verified_blueprint: bool = False
    verified_search: bool = False
    max_policy_calls: int = 2

    def __post_init__(self):
        if any(type(getattr(self, x)) is not bool for x in ('use_blueprint','use_search','use_adaptation',
                                                           'verified_blueprint','verified_search')):
            raise ValueError('routing flags must be bool')
        if type(self.max_policy_calls) is not int or self.max_policy_calls < 1:
            raise ValueError('max_policy_calls must be positive')


@dataclass(frozen=True)
class RoutedDecision:
    action_name: str
    action: object  # Phase1 Action
    probabilities: dict[str, float]
    source: str
    fallback_reasons: tuple[str, ...]
    adaptation_reason: str
    rng_seed: int


class IntegratedPokerAgent:
    """Deterministic seeded routing, support preserving fallback, no hidden inputs."""

    def __init__(self, neural: PolicySource, *, blueprint: PolicySource | None = None,
                 search: PolicySource | None = None,
                 expert_policies: Mapping[str, PolicySource] | None = None,
                 config: RoutingConfig | None = None, adaptive: SafeAdaptivePolicy | None = None,
                 expert_router: StyleExpertRouter | None = None):
        if not callable(neural):
            raise TypeError('neural fallback policy callable required')
        self.neural = neural
        self.blueprint = blueprint
        self.search = search
        self.experts = dict(expert_policies or {})
        if any(not callable(p) for p in self.experts.values()):
            raise TypeError('expert must be a callable')
        self.config = config or RoutingConfig()
        self.adaptive = adaptive or SafeAdaptivePolicy()
        self.expert_router = expert_router or StyleExpertRouter()
        self.last_decision: RoutedDecision | None = None

    def act(self, obs: InformationState, rng_seed: int, *, evidence=None) -> RoutedDecision:
        if type(obs) is not InformationState:
            raise TypeError('requires E2 InformationState; full engine state prohibited')
        if type(rng_seed) is not int or rng_seed < 0:
            raise ValueError('rng_seed must be nonnegative integer')
        legal = tuple(a for a, allowed in zip(ACTION_NAMES, obs.legal_actions.mask) if allowed)
        if not legal:
            raise ValueError('no legal action')
        reasons: list[str] = []
        selected = None
        probs = None
        candidates = (
            ('search', self.search, self.config.use_search, self.config.verified_search),
            ('blueprint', self.blueprint, self.config.use_blueprint, self.config.verified_blueprint),
            ('neural', self.neural, True, True),
        )
        calls = 0
        for name, source, enabled, verified in candidates:
            if not enabled:
                continue
            if not verified:
                reasons.append(f'{name}:unverified_abstraction')
                continue
            if source is None:
                reasons.append(f'{name}:unavailable')
                continue
            # Always reserve a chance for the trusted neural fallback, even
            # if earlier optional policy calls exhausted the optional budget.
            if name != 'neural' and calls >= self.config.max_policy_calls:
                reasons.append(f'{name}:policy_call_budget')
                continue
            calls += 1
            try:
                raw = source(obs)
                probs = normalize_policy(legal, raw, strict=True)
            except (ValueError, TypeError, RuntimeError, TimeoutError, OverflowError, ArithmeticError) as e:
                reasons.append(f'{name}:invalid_or_failed:{type(e).__name__}')
                continue
            selected = name
            break
        if selected is None or probs is None:
            # If a trusted source is broken or unavailable, do not pick illegal
            # actions or invent strategy. Uniform legal is explicitly labelled.
            selected = 'uniform_legal_emergency'
            probs = {a:1/len(legal) for a in legal}
            reasons.append('no_valid_policy_source')
        adapt_reason = 'disabled'
        if self.config.use_adaptation:
            proposals: dict[str,dict[str,float]] = {}
            for style, source in self.experts.items():
                try:
                    proposals[style] = normalize_policy(legal,source(obs))
                except (ValueError,TypeError,RuntimeError,TimeoutError,ArithmeticError):
                    reasons.append(f'adaptive_expert:{style}:invalid_or_failed')
            chosen = self.expert_router.propose(evidence, proposals, hero_legal=legal)
            adjusted = self.adaptive.adjust(probs, chosen.proposal, evidence, legal=legal)
            probs = adjusted.probabilities
            adapt_reason = f'{chosen.reason};{adjusted.reason}'
        rng = random.Random(rng_seed)
        action_name = rng.choices(legal, weights=[probs[a] for a in legal], k=1)[0]
        action = to_table_action(obs.legal_actions, action_name)
        decision = RoutedDecision(action_name, action, dict(probs), selected,
                                  tuple(reasons), adapt_reason, rng_seed)
        self.last_decision = decision
        return decision


def uniform_legal(obs: InformationState) -> dict[str,float]:
    legal = [a for a,mask in zip(ACTION_NAMES,obs.legal_actions.mask) if mask]
    return {a:1/len(legal) for a in legal}


def from_e03_checkpoint(path: str, *, device: str='cpu') -> PolicySource:
    """Phase3 E3 trust-checked neural checkpoint adapter, no hidden critic state."""
    from holdem_ai.expand.rl.trainer import load_trained_actor
    actor = load_trained_actor(path, device=device)
    def policy(obs: InformationState):
        probabilities,_ = actor.predict(encode_observation(obs))
        return {a:float(p) for a,p in zip(ACTION_NAMES,probabilities)
                if obs.legal_actions.allows(a)}
    return policy

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/integration/agent.py" -->

## src/holdem_ai/expand/integration/benchmark.py

Original SHA `1215d12d60ff1069f21f9c33a46a01391e6f800d` (11947 bytes).

<a name="py-132"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/integration/benchmark.py" sha="1215d12d60ff1069f21f9c33a46a01391e6f800d" -->

````````python
"""E7 reproducible *independent-session* cash and tournament benchmarks.

Cash uses one scored hand per freshly initialized table, not iid sequential
hands. Bootstrap hands used to admit BB-waiting players are reported separately.
Tournament uses one full tournament as one cluster; no hand-iid CI or ICM claim.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable
import math
import random
import statistics
from holdem_ai.expand.environment import DynamicTable, TableConfig
from holdem_ai.expand.information import Phase2TableAdapter
from holdem_ai.expand.information.builder import make_information_state
from holdem_ai.expand.information.schema import PublicAction
from holdem_ai.expand.rl.population import scripted_action
from holdem_ai.expand.opponent.bridge import PublicOpponentTracker
from .agent import IntegratedPokerAgent
from .tournament import SingleTableTournament, TournamentConfig


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str = 'cash'
    groups: int = 4
    seed: int = 700000007
    players: tuple[int,...] = (2,3,4,5,6)
    stacks_bb: tuple[int,...] = (20,100,200)
    opponents: tuple[str,...] = ('random','calling_station','tight','pot_pressure')
    max_actions: int = 400
    max_warmup_hands: int = 90
    max_tournament_hands: int = 120

    def __post_init__(self):
        if self.mode not in ('cash','tournament'):raise ValueError('invalid benchmark mode')
        if type(self.groups) is not int or self.groups<2:raise ValueError('groups >=2 required for CI')
        if type(self.seed) is not int or self.seed<0:raise ValueError('seed must be nonnegative')
        if not self.players or any(type(n) is not int or n<2 or n>6 for n in self.players):
            raise ValueError('players must be 2..6')
        if not self.stacks_bb or any(type(s) is not int or not 20<=s<=300 for s in self.stacks_bb):
            raise ValueError('stack sizes must be 20..300 BB')
        if not self.opponents or any(x not in ('random','calling_station','tight','pot_pressure')
                                      for x in self.opponents):raise ValueError('invalid opponent')
        if any(type(getattr(self,x)) is not int or getattr(self,x)<1
               for x in ('max_actions','max_warmup_hands','max_tournament_hands')):
            raise ValueError('positive episode guards required')


def _stats(values:list[float], factor:float=1.)->dict:
    if len(values)<2:raise ValueError('at least 2 independent blocks required')
    mean=statistics.mean(values)*factor
    se=statistics.stdev(values)*factor/math.sqrt(len(values))
    return {'n_independent_sessions':len(values),'mean':round(mean,6),
            'se':round(se,6),'ci95_normal':[round(mean-1.96*se,6),round(mean+1.96*se,6)]}


def _cash_session(agent:IntegratedPokerAgent,*,n:int,stack_bb:int,style:str,
                  scenario_seed:int,hero_seat:int,engine_factory:Callable|None,
                  max_actions:int,max_warmup_hands:int)->dict:
    rng=random.Random(scenario_seed)
    tc=TableConfig()
    table=DynamicTable(tc,session_id=f'e07-{scenario_seed}',engine_factory=engine_factory)
    adapter=Phase2TableAdapter(table)
    tracker=PublicOpponentTracker()
    tenures={}
    for seat in range(n):
        player='hero' if seat==hero_seat else f'opp_{seat}'
        tenures[player]=table.request_join(player,stack_bb*tc.big_blind,preferred_seat=seat)
    warmups=0
    while True:
        handle=adapter.start_hand(scenario_seed*101+warmups)
        if handle is None:raise RuntimeError('table stalled with insufficient active participants')
        roster=dict(handle.participants)
        tracker.sync_seats(table.snapshot(),tenures)
        ready=(len(roster)==n and roster.get(hero_seat)=='hero')
        actions=0
        while table.current_player_id is not None:
            if actions>=max_actions:raise RuntimeError('cash episode decision guard triggered')
            state=adapter.observe()
            pid=table.current_player_id
            if ready and pid=='hero':
                model=None
                # Opponent evidence is per-opponent & contextual; no oracle.
                # Phase 6 routing falls back if there is insufficient evidence.
                for (key_id,join_id),candidate in tracker.registry.models.items():
                    if key_id!='hero' and candidate.contexts:
                        context=next(reversed(candidate.contexts))
                        model=candidate.evidence(context)
                        break
                chosen=agent.act(state,rng.randrange(2**63),evidence=model).action_name
            else:
                chosen=scripted_action(style,state,rng)
            adapter.act_abstract(chosen)
            tracker.record_accepted_action(state,handle=handle,player_id=pid,
                       player_session_id=tenures[pid],action_index=actions,action=chosen)
            actions+=1
        settled=adapter.settle_hand()
        tracker.close_hand(handle)
        if sum(settled.payoffs_chips)!=0:
            raise AssertionError('cash payoff not zero-sum')
        if ready:
            return {'payoff_bb':settled.payoffs_bb[hero_seat], 'hands':1,
                    'warmup_hands':warmups,'illegal_actions':0,
                    'opponent_events':sum(len(m.seen_events) for m in tracker.registry.models.values())}
        warmups+=1
        if warmups>=max_warmup_hands:
            raise RuntimeError('full roster was not reached before admission guard')
        # A busted player during BB-waiting needs a fresh buy-in before the
        # measured game; give the fresh identity a new tenure.
        snapshot=table.snapshot()
        for seat in range(n):
            if snapshot.seats[seat].player_id is None:
                pid='hero' if seat==hero_seat else f'opp_{seat}'
                tenures[pid]=table.request_join(pid,stack_bb*tc.big_blind,preferred_seat=seat)


def cash_benchmark(agent:IntegratedPokerAgent, config:BenchmarkConfig,
                   *, engine_factory:Callable|None=None)->dict:
    if config.mode!='cash':raise ValueError('requires cash benchmark config')
    rows=[]
    all_values=[]
    total_warmup=0
    for n in config.players:
        for stacks in config.stacks_bb:
            for style in config.opponents:
                values=[]
                for b in range(config.groups):
                    # Shared deterministic scenario seeds for A/B comparisons.
                    scenario=config.seed+n*10000000+stacks*10000+config.opponents.index(style)*1000+b
                    run=_cash_session(agent,n=n,stack_bb=stacks,style=style,scenario_seed=scenario,
                          hero_seat=b%n,engine_factory=engine_factory,
                          max_actions=config.max_actions,max_warmup_hands=config.max_warmup_hands)
                    values.append(run['payoff_bb'])
                    all_values.append(run['payoff_bb'])
                    total_warmup+=run['warmup_hands']
                rows.append({'players':n,'stack_bb':stacks,'opponent':style,
                             'bb100':_stats(values,100.),'block_payoffs_bb':values})
    # Aggregate is descriptive. Independent sessions are the sampling unit,
    # but across non-identical opponents/stack groups it is a mixture estimand.
    return {'mode':'cash','config':asdict(config),'overall_bb100':_stats(all_values,100.),
            'strata':rows,'scored_hands':len(all_values),'admission_warmup_hands':total_warmup,
            'note':'One scored hand per independent table; no training or opponent adaptation advantage claimed.'}


def _tournament_session(agent:IntegratedPokerAgent, *, n:int,style:str,seed:int,hero_seat:int,
                        engine_factory:Callable|None, max_actions:int,max_hands:int):
    rng=random.Random(seed)
    cfg=TournamentConfig(starting_stack=2000,hands_per_level=2,max_hands=max_hands)
    tour=SingleTableTournament(cfg,engine_factory=engine_factory)
    for seat in range(n):
        tour.register('hero' if seat==hero_seat else f'opp_{seat}',seat=seat)
    tour.close_registration()
    public_history=[]
    while not tour.finished:
        hand=tour.start_hand(seed*1000+tour.hands_played)
        if hand is None:
            if tour.finished:break
            raise RuntimeError('unfinishable tournament')
        actions=0
        public_history=[]
        while tour.current_player_id is not None:
            if actions>=max_actions:raise RuntimeError('tournament decision guard triggered')
            raw=tour.observe_current()
            info=make_information_state(raw,big_blind_chips=hand.big_blind,
                seat_presence=tuple(e is not None for e in tour.seats),
                seat_statuses=tuple('seated' if e else 'vacant' for e in tour.seats),
                history=tuple(public_history))
            if tour.current_player_id=='hero':
                chosen=agent.act(info,rng.randrange(2**63)).action_name
            else:
                chosen=scripted_action(style,info,rng)
            from holdem_ai.expand.information.actions import to_table_action
            table_action=to_table_action(info.legal_actions,chosen)
            raise_target=table_action.raise_to_chips
            if chosen=='all_in':
                r=raw['raw_obs']
                raise_target=r['street_bets'][info.hero_seat]+r['stacks'][info.hero_seat]
            tour.act(table_action)
            public_history.append(PublicAction(info.hero_seat,info.street,chosen,
                 raise_target/hand.big_blind if raise_target is not None else None,info.to_call_bb,info.pot_bb))
            actions+=1
        result=tour.settle_hand()
        if sum(result.payoffs_chips)!=0:raise AssertionError('tournament hand nonzero sum')
    report=tour.result()
    hero_prize=sum(e[3] for e in report.entries if e[1]=='hero')
    hero_win=float(any(e[1]=='hero' and e[2]==1 for e in report.entries))
    return hero_prize,hero_win,report.hands_played


def tournament_benchmark(agent:IntegratedPokerAgent, config:BenchmarkConfig,
                         *, engine_factory:Callable|None=None)->dict:
    if config.mode!='tournament':raise ValueError('requires tournament benchmark config')
    rows=[]
    prizes=[]
    for n in config.players:
        for style in config.opponents:
            p=[];wins=[];lengths=[]
            for b in range(config.groups):
                scenario=config.seed+n*10000000+config.opponents.index(style)*1000+b
                value,won,length=_tournament_session(agent,n=n,style=style,seed=scenario,
                   hero_seat=b%n,engine_factory=engine_factory,
                   max_actions=config.max_actions,max_hands=config.max_tournament_hands)
                p.append(value);wins.append(won);lengths.append(length);prizes.append(value)
            rows.append({'players':n,'opponent':style,'prize_ev':_stats(p),
                         'win_rate':sum(wins)/len(wins),
                         'mean_hands':statistics.mean(lengths),'session_prizes':p})
    return {'mode':'tournament','config':asdict(config),'overall_prize_ev':_stats(prizes),
            'strata':rows,'tournaments':len(prizes),
            'note':'Session-clustered prize EV. No ICM estimate, tournament hand bb100, or multi-table support.'}


def paired_cash_difference(left:dict,right:dict)->dict:
    if left['config']!=right['config']:
        raise ValueError('paired comparison requires identical scenario configuration')
    la=left['strata'];ra=right['strata']
    if len(la)!=len(ra):raise ValueError('different stratification')
    diffs=[]
    for a,b in zip(la,ra):
        if (a['players'],a['stack_bb'],a['opponent'])!=(b['players'],b['stack_bb'],b['opponent']):
            raise ValueError('different strata')
        diffs += [x-y for x,y in zip(a['block_payoffs_bb'],b['block_payoffs_bb'])]
    return {'paired_difference_bb100':_stats(diffs,100.), 'direction':'left_minus_right'}

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/integration/benchmark.py" -->
