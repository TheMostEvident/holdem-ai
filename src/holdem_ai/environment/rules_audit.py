"""Observational, non-mutating audit of installed RLCard HUNL baseline.

Failures indicate the UPSTREAM engine differs from HUNL requirements, not that
our reference rules are wrong. No RLCard monkey-patching is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from holdem_ai.environment.seat_manager import heads_up_seats


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    expected: str
    observed: str


def audit_preflop(game: Any) -> tuple[AuditCheck, ...]:
    """Call immediately after RLCard's `env.reset()`.

    RLCard has a public `game.dealer_id`, `game.game_pointer`, player.in_chips.
    Checks against standard heads-up: BTN=SB, BB=other, BTN acts first.
    """
    seats = heads_up_seats(int(game.dealer_id))
    sb_paid = int(game.players[seats.small_blind].in_chips)
    bb_paid = int(game.players[seats.big_blind].in_chips)
    first = int(game.game_pointer)
    expected_sb = min(int(game.small_blind), int(game.init_chips[seats.small_blind]))
    expected_bb = min(int(game.big_blind), int(game.init_chips[seats.big_blind]))
    return (
        AuditCheck('button_posts_small_blind', sb_paid == expected_sb, str(expected_sb), str(sb_paid)),
        AuditCheck('opponent_posts_big_blind', bb_paid == expected_bb, str(expected_bb), str(bb_paid)),
        AuditCheck('button_acts_first_preflop', first == seats.preflop_first, str(seats.preflop_first), str(first)),
    )


def summarize(checks: tuple[AuditCheck, ...] | list[AuditCheck]) -> str:
    lines = []
    for check in checks:
        prefix = 'PASS' if check.passed else 'FAIL'
        lines.append(f'{prefix} {check.name}: expected={check.expected}, observed={check.observed}')
    return '\n'.join(lines)
