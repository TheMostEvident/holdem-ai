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
