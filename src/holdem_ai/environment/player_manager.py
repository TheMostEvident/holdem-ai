"""Persistent chip stacks and elimination ranks across tournament hands.

Simultaneous busts are ordered by stacks immediately before that hand.
Equal pre-hand stacks share a placement (competition ranking).
"""
from __future__ import annotations

from typing import Sequence


class PlayerManager:
    def __init__(self, initial_stacks: Sequence[int]):
        initial = tuple(initial_stacks)
        if not 2 <= len(initial) <= 6 or any(type(x) is not int or x <= 0 for x in initial):
            raise ValueError('require 2-6 positive integer starting stacks')
        self.initial_stacks = initial
        self.total_chips = sum(initial)
        self.stacks = initial
        self.ranks: list[int | None] = [None] * len(initial)
        self.eliminated_at: list[int | None] = [None] * len(initial)

    @property
    def alive(self) -> tuple[int, ...]:
        return tuple(i for i, chips in enumerate(self.stacks) if chips > 0)

    @property
    def winner(self) -> int | None:
        alive = self.alive
        return alive[0] if len(alive) == 1 else None

    def update(self, final_stacks: Sequence[int], hand_number: int) -> tuple[int, ...]:
        """Commit a completed hand; return busted seats (ascending IDs)."""
        after = tuple(final_stacks)
        if type(hand_number) is not int or hand_number <= 0:
            raise ValueError('hand number must be positive')
        if len(after) != len(self.stacks) or any(type(x) is not int or x < 0 for x in after):
            raise ValueError('invalid final stacks')
        if sum(after) != self.total_chips:
            raise ValueError('chips cannot enter or leave the tournament')
        if any(old == 0 and new != 0 for old, new in zip(self.stacks, after)):
            raise ValueError('busted players cannot be revived')
        if self.winner is not None:
            raise ValueError('tournament already finished')

        busts = [i for i, (old, new) in enumerate(zip(self.stacks, after)) if old > 0 and new == 0]
        # The lowest pre-hand stack finishes worse in a same-hand knockout.
        # Players with equal pre-hand stacks share the occupied prize positions.
        cursor = len(self.alive)
        for start_stack in sorted({self.stacks[i] for i in busts}):
            tied = [i for i in busts if self.stacks[i] == start_stack]
            shared_rank = cursor - len(tied) + 1
            for i in tied:
                self.ranks[i] = shared_rank
                self.eliminated_at[i] = hand_number
            cursor -= len(tied)
        self.stacks = after
        if self.winner is not None:
            self.ranks[self.winner] = 1
        return tuple(sorted(busts))
