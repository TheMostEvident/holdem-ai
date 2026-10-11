"""Moving-button seat and action order for 2-6 seat Texas Hold'em hands.

Seat indexes are fixed across hands; stack == 0 denotes an empty/busted seat.
Heads-up is a special case: button posts SB and acts first preflop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MultiPlayerSeats:
    button: int
    small_blind: int
    big_blind: int
    preflop_first: int
    postflop_first: int
    occupied: tuple[int, ...]


def next_occupied(seat: int, occupied: Sequence[int], num_seats: int) -> int:
    occupied_set = set(occupied)
    for offset in range(1, num_seats + 1):
        candidate = (seat + offset) % num_seats
        if candidate in occupied_set:
            return candidate
    raise ValueError('no occupied seat')


def multi_player_seats(stacks: Sequence[int], button: int) -> MultiPlayerSeats:
    n = len(stacks)
    if n < 2 or n > 6 or any(type(x) is not int or x < 0 for x in stacks):
        raise ValueError('require 2-6 nonnegative integer stacks')
    occupied = tuple(i for i, x in enumerate(stacks) if x > 0)
    if len(occupied) < 2:
        raise ValueError('a hand requires at least two occupied seats')
    if type(button) is not int or button not in occupied:
        raise ValueError('button must be an occupied seat')
    if len(occupied) == 2:
        sb = button
        bb = next_occupied(button, occupied, n)
        pre = sb
        post = bb
    else:
        sb = next_occupied(button, occupied, n)
        bb = next_occupied(sb, occupied, n)
        pre = next_occupied(bb, occupied, n)
        post = sb
    return MultiPlayerSeats(button, sb, bb, pre, post, occupied)


def rotate_button(stacks: Sequence[int], button: int) -> int:
    """Moving-button model. Next occupied seat clockwise from prior button.

    Explicitly not a tournament dead-button/blind-placement policy.
    """
    occupied = tuple(i for i, chips in enumerate(stacks) if chips > 0)
    if not occupied:
        raise ValueError('no occupied players')
    return next_occupied(button, occupied, len(stacks))
