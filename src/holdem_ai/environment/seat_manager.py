"""Rule reference for heads-up seating and betting order.

This module intentionally does not mutate RLCard internals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadsUpSeats:
    """Button is the small blind in a two-player hand."""

    button: int
    small_blind: int
    big_blind: int
    preflop_first: int
    postflop_first: int


def heads_up_seats(button: int) -> HeadsUpSeats:
    if button not in (0, 1):
        raise ValueError('A two-player button must be player 0 or 1.')
    bb = 1 - button
    return HeadsUpSeats(
        button=button,
        small_blind=button,
        big_blind=bb,
        preflop_first=button,
        postflop_first=bb,
    )


def rotate_heads_up_button(button: int) -> int:
    """Switch the button to the other remaining seat between hands."""
    return heads_up_seats(button).big_blind


def post_blinds(stacks: tuple[int, int], button: int, sb: int, bb: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (posted_chips, remaining_stacks), including short-stack blind all-in."""
    if len(stacks) != 2 or any(not isinstance(s, int) or s < 0 for s in stacks):
        raise ValueError('stacks must contain two nonnegative integer balances')
    if not (0 < sb < bb):
        raise ValueError('require 0 < small blind < big blind')
    seats = heads_up_seats(button)
    posts = [0, 0]
    posts[seats.small_blind] = min(sb, stacks[seats.small_blind])
    posts[seats.big_blind] = min(bb, stacks[seats.big_blind])
    remaining = tuple(stacks[i] - posts[i] for i in range(2))
    return (posts[0], posts[1]), remaining
