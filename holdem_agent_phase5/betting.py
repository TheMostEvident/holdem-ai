"""Optional integer-chip action abstraction for engines with arbitrary raise-to amounts.

This layer NEVER determines poker legality: the engine supplies legal flags and
min/max raise-to boundaries. Rounding, capping and duplicate suppression only
construct engine-safe candidates within those supplied bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BET_ACTION_LABELS = (
    "fold", "check", "call", "raise_half_pot", "raise_pot", "raise_double_pot", "all_in",
)


@dataclass(frozen=True, slots=True)
class EngineBetAction:
    kind: Literal["fold", "check", "call", "raise_to", "all_in"]
    raise_to: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "raise_to":
            if self.raise_to is None or self.raise_to < 0:
                raise ValueError("raise_to actions require nonnegative absolute street bet target")
        elif self.raise_to is not None:
            raise ValueError("only raise_to has a raise_to amount")


@dataclass(frozen=True, slots=True)
class NoLimitBetContext:
    """Poker-engine output, not an independent rules implementation.

    pot: total pot chips BEFORE player's next action (including opponents' bets)
    committed: player's chips already contributed on THIS street
    to_call: chips needed to match current street bet (can exceed stack)
    stack: player's remaining chips
    min_raise_to: engine-legal minimum *total contribution this street*
    max_raise_to: engine-legal maximum *total contribution this street*

    Raise flags must reflect re-opening, caps, tournament stack constraints and
    engine policy. A short all-in raise may be allowed below min_raise_to.
    """

    pot: int
    committed: int
    to_call: int
    stack: int
    min_raise_to: int | None
    max_raise_to: int | None
    can_fold: bool
    can_check: bool
    can_call: bool
    can_raise: bool

    def __post_init__(self) -> None:
        quantities = (self.pot, self.committed, self.to_call, self.stack)
        if any(type(x) is not int or x < 0 for x in quantities):
            raise ValueError("chip quantities must be nonnegative integers")
        for x in (self.min_raise_to, self.max_raise_to):
            if x is not None and (type(x) is not int or x < 0):
                raise ValueError("raise bounds must be nonnegative integers or None")
        if self.can_raise and (self.min_raise_to is None or self.max_raise_to is None):
            raise ValueError("can_raise requires explicit raise bounds")
        if self.can_check and self.to_call != 0:
            raise ValueError("check cannot be legal with outstanding call")
        if self.can_call and self.to_call <= 0:
            raise ValueError("call must be positive; use check when to_call=0")
        if self.max_raise_to is not None and self.max_raise_to > self.committed + self.stack:
            raise ValueError("max_raise_to exceeds all-in stack")


@dataclass(frozen=True, slots=True)
class DiscreteBetActions:
    """Fixed 7-action policy head; None entries are masked out."""

    actions: tuple[EngineBetAction | None, ...]

    @property
    def mask(self) -> tuple[bool, ...]:
        return tuple(a is not None for a in self.actions)

    def resolve(self, action_index: int) -> EngineBetAction:
        if isinstance(action_index, bool) or not isinstance(action_index, int):
            raise TypeError("action_index must be int")
        if not 0 <= action_index < len(self.actions) or self.actions[action_index] is None:
            raise ValueError("action is masked as illegal")
        action = self.actions[action_index]
        assert action is not None
        return action


def abstract_bet_actions(ctx: NoLimitBetContext) -> DiscreteBetActions:
    """Build action mask and engine action mapping using engine-provided bounds.

    For fraction f of the pot AFTER calling, the proposed total street
    contribution is c + call + round(f * (pot + call)).
    The 0.5, 1, 2 pot fractions are rounded to integer chips; this is an
    action-size *abstraction*, NOT a proof of poker-rule completeness.
    """
    actions: list[EngineBetAction | None] = [None] * len(BET_ACTION_LABELS)
    if ctx.can_fold:
        actions[0] = EngineBetAction("fold")
    if ctx.can_check:
        actions[1] = EngineBetAction("check")
    if ctx.can_call:
        actions[2] = EngineBetAction("call")

    if ctx.can_raise and ctx.stack > ctx.to_call:
        assert ctx.min_raise_to is not None and ctx.max_raise_to is not None
        maximum = ctx.max_raise_to
        called_to = ctx.committed + ctx.to_call
        all_in_to = ctx.committed + ctx.stack
        # All-in may be a legal short raise below the normal minimum. The
        # engine's 'can_raise' MUST already account for action reopening.
        if maximum == all_in_to and maximum > called_to:
            actions[6] = EngineBetAction("all_in")
        # Engine bounds define ordinary valid full raises. Short all-ins are
        # represented only by the explicit all_in action above.
        if ctx.min_raise_to <= maximum and maximum > called_to:
            seen_targets: set[int] = set()
            for index, (numer, denom) in enumerate(((1, 2), (1, 1), (2, 1)), start=3):
                extra = (numer * (ctx.pot + ctx.to_call) + denom // 2) // denom
                target = min(max(called_to + extra, ctx.min_raise_to), maximum)
                if target <= called_to or target >= all_in_to or target in seen_targets:
                    continue
                actions[index] = EngineBetAction("raise_to", target)
                seen_targets.add(target)
    if not any(x is not None for x in actions):
        raise ValueError("environment supplied no legal action for this turn")
    return DiscreteBetActions(tuple(actions))
