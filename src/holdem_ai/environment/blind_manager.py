"""Deterministic hand-count blind schedule for single-table tournaments.

The current hand engine has no antes; this manager intentionally supports only
small and big blinds. A final configured level repeats indefinitely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BlindLevel:
    small_blind: int
    big_blind: int

    def __post_init__(self) -> None:
        if (type(self.small_blind) is not int or type(self.big_blind) is not int
                or not 0 < self.small_blind < self.big_blind):
            raise ValueError('blinds must be integer chips satisfying 0 < SB < BB')


DEFAULT_BLIND_LEVELS = (
    BlindLevel(5, 10), BlindLevel(10, 20), BlindLevel(20, 40),
    BlindLevel(40, 80), BlindLevel(80, 160), BlindLevel(160, 320),
    BlindLevel(320, 640), BlindLevel(640, 1280),
    BlindLevel(1280, 2560), BlindLevel(2560, 5120),
    BlindLevel(5120, 10240),
)


class BlindManager:
    def __init__(
        self,
        levels: Sequence[BlindLevel | tuple[int, int]] | None = None,
        hands_per_level: int = 10,
    ) -> None:
        if type(hands_per_level) is not int or hands_per_level <= 0:
            raise ValueError('hands_per_level must be positive integer')
        if levels is None:
            levels = DEFAULT_BLIND_LEVELS
        parsed = []
        for level in levels:
            if isinstance(level, BlindLevel):
                parsed.append(level)
            elif isinstance(level, (list, tuple)) and len(level) == 2:
                parsed.append(BlindLevel(*level))
            else:
                raise ValueError('each blind level must be (SB, BB)')
        if not parsed:
            raise ValueError('blind schedule cannot be empty')
        for before, after in zip(parsed, parsed[1:]):
            if (after.small_blind < before.small_blind
                    or after.big_blind < before.big_blind):
                raise ValueError('blind levels must not decrease')
        self.levels = tuple(parsed)
        self.hands_per_level = hands_per_level

    def at_hand(self, hand_number: int) -> tuple[int, BlindLevel]:
        """Return (0-based level index, level) for a 1-based hand number."""
        if type(hand_number) is not int or hand_number < 1:
            raise ValueError('hand_number must be >= 1')
        index = min((hand_number - 1) // self.hands_per_level, len(self.levels) - 1)
        return index, self.levels[index]
