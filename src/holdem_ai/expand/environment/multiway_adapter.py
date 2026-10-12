"""Late-bound adapter to the unchanged Base Phase 3 6-seat hand engine."""
from __future__ import annotations


def make_hand_engine(**kwargs):
    from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine
    return MultiPlayerHandEngine(**kwargs)


def moving_button_after(previous: int | None, active: tuple[int, ...]) -> int:
    if not active:
        raise ValueError('no active seat')
    if previous is None:
        return active[0]
    for offset in range(1, 7):
        candidate = (previous + offset) % 6
        if candidate in active:
            return candidate
    raise AssertionError('no next button')


def blind_seats(button: int, active: tuple[int, ...]) -> tuple[int, int]:
    from holdem_ai.environment.multi_player_seats import multi_player_seats
    stacks = [100 if i in active else 0 for i in range(6)]
    seats = multi_player_seats(stacks, button)
    return seats.small_blind, seats.big_blind
