"""Terminal reward definitions; preserve distinction between chip and tournament utility."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _vector(values: Sequence[int | float], label: str) -> tuple[float, ...]:
    result = tuple(float(x) for x in values)
    if len(result) < 2 or any(not math.isfinite(x) for x in result):
        raise ValueError(f"{label} requires >=2 finite values")
    return result


@dataclass(frozen=True, slots=True)
class ChipDeltaReward:
    """r_i = (stack_end[i] - stack_start[i]) / scale; scale often BB."""

    scale: float = 1.0
    enforce_conservation: bool = True
    atol: float = 1e-8

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and > 0")

    def calculate(self, starting_stacks: Sequence[float], ending_stacks: Sequence[float]) -> tuple[float, ...]:
        start = _vector(starting_stacks, "starting_stacks")
        end = _vector(ending_stacks, "ending_stacks")
        if len(start) != len(end):
            raise ValueError("starting/ending stack count mismatch")
        if self.enforce_conservation and not math.isclose(
            math.fsum(start), math.fsum(end), rel_tol=0, abs_tol=self.atol
        ):
            raise ValueError("chip conservation violated (or rake/fees not configured)")
        return tuple((e - s) / self.scale for s, e in zip(start, end))


@dataclass(frozen=True, slots=True)
class WinnerTakeAllReward:
    """Centered winner-take-all reward: r_i = 1[i wins] - 1 / N."""

    def calculate(self, winner_id: int, num_players: int) -> tuple[float, ...]:
        if num_players < 2 or not 0 <= winner_id < num_players:
            raise ValueError("invalid tournament winner or player count")
        return tuple(float(i == winner_id) - 1.0 / num_players for i in range(num_players))
