"""Reference side-pot construction and settlement with chip conservation.

Works with 2+ players; prepared for 6-max tests. Winner ranks must be supplied by
an independently verified hand evaluator (larger rank means stronger hand).
No rake. Integer-chip odd chips go clockwise from the seat left of the button.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Pot:
    amount: int
    eligible: tuple[int, ...]


@dataclass(frozen=True)
class PotBreakdown:
    pots: tuple[Pot, ...]
    uncalled_returns: tuple[int, ...]


def build_side_pots(contributions: Sequence[int], folded: Sequence[bool]) -> PotBreakdown:
    if len(contributions) < 2 or len(folded) != len(contributions):
        raise ValueError('need two or more players and matching fold flags')
    if any(not isinstance(v, int) or v < 0 for v in contributions):
        raise ValueError('contributions must be nonnegative integer chips')
    if any(not isinstance(v, bool) for v in folded):
        raise ValueError('folded flags must be bool')
    if all(folded):
        raise ValueError('at least one player must remain eligible')

    levels = sorted(set(v for v in contributions if v > 0))
    previous = 0
    pots: list[Pot] = []
    returned = [0] * len(contributions)
    for upper in levels:
        participants = [i for i, chips in enumerate(contributions) if chips >= upper]
        layer = (upper - previous) * len(participants)
        eligible = tuple(i for i in participants if not folded[i])
        if len(participants) == 1:
            # An unmatched wager belongs to its contributor, even if they folded.
            returned[participants[0]] += layer
        elif not eligible:
            raise ValueError('pot has no eligible winner: invalid hand state')
        else:
            pots.append(Pot(layer, eligible))
        previous = upper

    assert sum(p.amount for p in pots) + sum(returned) == sum(contributions)
    return PotBreakdown(tuple(pots), tuple(returned))


def settle_pots(
    breakdown: PotBreakdown,
    strengths: Sequence[object | None],
    button: int,
) -> tuple[int, ...]:
    """Distribute pots by comparable showdown strengths; equal strengths split.

    Folded players should have strength None. Strength objects must be ordered
    so higher values are better (e.g. tuples of ranking components).
    """
    n = len(breakdown.uncalled_returns)
    if len(strengths) != n or not 0 <= button < n:
        raise ValueError('player count / button mismatch')
    awards = list(breakdown.uncalled_returns)
    for pot in breakdown.pots:
        eligible = [i for i in pot.eligible if strengths[i] is not None]
        if not eligible:
            raise ValueError('eligible pot player must have showdown strength')
        best = max(strengths[i] for i in eligible)
        winners = [i for i in eligible if strengths[i] == best]
        share, odd = divmod(pot.amount, len(winners))
        for winner in winners:
            awards[winner] += share
        # First odd chip goes to winner seated immediately left of button.
        clockwise = sorted(winners, key=lambda i: (i - button - 1) % n)
        for winner in clockwise[:odd]:
            awards[winner] += 1
    return tuple(awards)
