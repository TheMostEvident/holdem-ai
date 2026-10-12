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
