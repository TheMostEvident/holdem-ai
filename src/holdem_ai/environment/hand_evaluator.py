"""Independent exact seven-card Hold'em evaluator for showdown verification.

Card notation: 'sK', 'd7', 'cT', 'hA' (suit first). Larger tuple wins.
This is a reference implementation, not an optimized production evaluator.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Sequence

RANKS = '23456789TJQKA'
SUITS = 'shdc'
Rank = tuple[int, ...]


def parse_card(card: str) -> tuple[str, int]:
    if not isinstance(card, str) or len(card) != 2:
        raise ValueError('card must have two characters: e.g. sK')
    suit, rank = card[0].lower(), card[1].upper()
    if suit not in SUITS or rank not in RANKS:
        raise ValueError(f'invalid Holdem card: {card!r}')
    return suit, RANKS.index(rank) + 2


def _straight_high(values: Sequence[int]) -> int | None:
    unique = set(values)
    if 14 in unique:
        unique.add(1)  # A-2-3-4-5 wheel
    for high in range(14, 4, -1):
        if all(v in unique for v in range(high - 4, high + 1)):
            return high
    return None


def rank_five(cards: Sequence[str]) -> Rank:
    if len(cards) != 5:
        raise ValueError('rank_five requires exactly 5 cards')
    parsed = [parse_card(x) for x in cards]
    if len(set(parsed)) != 5:
        raise ValueError('duplicate cards are not allowed')
    ranks = [rank for _, rank in parsed]
    suits = [suit for suit, _ in parsed]
    count = Counter(ranks)
    groups = sorted(((freq, rank) for rank, freq in count.items()), reverse=True)
    flush = len(set(suits)) == 1
    straight = _straight_high(ranks)
    if flush and straight is not None:
        return (8, straight)
    if groups[0][0] == 4:
        four = groups[0][1]
        kicker = max(r for r in ranks if r != four)
        return (7, four, kicker)
    if [g[0] for g in groups] == [3, 2]:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *sorted(ranks, reverse=True))
    if straight is not None:
        return (4, straight)
    if groups[0][0] == 3:
        triple = groups[0][1]
        return (3, triple, *sorted((r for r in ranks if r != triple), reverse=True))
    if groups[0][0] == 2 and groups[1][0] == 2:
        high_pair = max(groups[0][1], groups[1][1])
        low_pair = min(groups[0][1], groups[1][1])
        kicker = max(r for r in ranks if r not in (high_pair, low_pair))
        return (2, high_pair, low_pair, kicker)
    if groups[0][0] == 2:
        pair = groups[0][1]
        return (1, pair, *sorted((r for r in ranks if r != pair), reverse=True))
    return (0, *sorted(ranks, reverse=True))


def rank_seven(cards: Sequence[str]) -> Rank:
    """Exact max over C(7,5)=21 five-card combinations."""
    if len(cards) != 7:
        raise ValueError('rank_seven requires exactly 7 cards')
    canonical = [f'{parse_card(card)[0]}{card[1].upper()}' for card in cards]
    if len(set(canonical)) != 7:
        raise ValueError('duplicate cards are not allowed')
    return max(rank_five(hand) for hand in combinations(canonical, 5))


def rank_showdown(hole: Sequence[str], board: Sequence[str]) -> Rank:
    if len(hole) != 2 or len(board) != 5:
        raise ValueError('showdown requires 2 hole cards and 5 board cards')
    return rank_seven((*hole, *board))
