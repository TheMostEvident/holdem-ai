"""Hand-level chip rewards and terminal tournament placement rewards."""
from __future__ import annotations

from typing import Sequence


def chip_rewards(payoffs_chips: Sequence[int], big_blind: int) -> tuple[float, ...]:
    if type(big_blind) is not int or big_blind <= 0:
        raise ValueError('BB must be positive integer')
    return tuple(x / big_blind for x in payoffs_chips)


def placement_rewards(
    ranks: Sequence[int | None], payouts: Sequence[float] | None = None
) -> tuple[float, ...]:
    """Terminal reward with ties splitting prizes for occupied rank slots.

    Example: six entrants, two tied at rank 5 share prizes for 5th and 6th.
    Default payout scheme awards 1 to the champion, 0 to everybody else.
    """
    n = len(ranks)
    if n < 2 or any(type(r) is not int or not 1 <= r <= n for r in ranks):
        raise ValueError('all ranks must be final integers within 1..N')
    if payouts is None:
        payouts = (1.0,) + (0.0,) * (n - 1)
    if len(payouts) != n or any(not isinstance(v, (float, int)) or isinstance(v, bool) or v < 0 for v in payouts):
        raise ValueError('payouts must contain N nonnegative numbers')
    # A valid tied rank must skip as many following places as seats in the tie.
    groups: dict[int, list[int]] = {}
    for i, rank in enumerate(ranks):
        groups.setdefault(rank, []).append(i)
    next_rank = 1
    for rank, seats in sorted(groups.items()):
        if rank != next_rank:
            raise ValueError('rank sequence has invalid tie/placement gaps')
        next_rank += len(seats)
    out = [0.0] * n
    for rank, seats in groups.items():
        shared = sum(payouts[rank - 1:rank - 1 + len(seats)]) / len(seats)
        for seat in seats:
            out[seat] = float(shared)
    return tuple(out)
