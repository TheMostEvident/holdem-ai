"""Stable VRPO-Lite v1 token vocabulary (NOT the original VRPO paper encoding)."""
from __future__ import annotations

PAD = 0
BOS = 1
PRIVATE = 2
OPPONENT_PRIVATE = 3  # Training-only token: never used by the actor.
BOARD = 4
STREET = 5
HISTORY = 6
EVENT = 7
AMOUNT = 8
TERMINAL = 9
SEAT_0 = 10
SEAT_1 = 11

SUITS = 'sdch'
RANKS = '23456789TJQKA'
CARDS = tuple(s + r for s in SUITS for r in RANKS)
CARD_TO_ID = {card: 12 + i for i, card in enumerate(CARDS)}
STREETS = ('preflop', 'flop', 'turn', 'river')
STREET_TO_ID = {street: 64 + i for i, street in enumerate(STREETS)}
ACTIONS = ('fold', 'check', 'call', 'raise_to', 'all_in')
ACTION_TO_ID = {name: 68 + i for i, name in enumerate(ACTIONS)}
VOCAB_SIZE = 73


def card_id(card: str) -> int:
    try:
        return CARD_TO_ID[card]
    except (TypeError, KeyError) as exc:
        raise ValueError(f'invalid canonical card: {card!r}') from exc


def seat_id(seat: int) -> int:
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError('seat must be 0 or 1')
    return SEAT_0 if seat == 0 else SEAT_1
