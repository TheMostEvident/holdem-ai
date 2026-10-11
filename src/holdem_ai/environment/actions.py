"""Independent no-limit betting constraints (amounts are integer chips).

Pure rule helpers. RLCard's 5-action abstraction is NOT modified by these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BetContext:
    """Current street context for one decision.

    actor_commitment: chips already bet this street by the acting player.
    highest_commitment: highest contribution by any player on this street.
    last_full_raise: most recent full raise increment (or BB initially).
    remaining_stack: acting player's chips available before decision.
    may_raise: false when action has not been reopened by a short all-in.
    """

    actor_commitment: int
    highest_commitment: int
    last_full_raise: int
    remaining_stack: int
    big_blind: int
    may_raise: bool = True

    def __post_init__(self) -> None:
        parts = (
            self.actor_commitment,
            self.highest_commitment,
            self.last_full_raise,
            self.remaining_stack,
            self.big_blind,
        )
        if any(not isinstance(v, int) for v in parts):
            raise TypeError('bet sizes must be integer chip values')
        if self.actor_commitment < 0 or self.highest_commitment < self.actor_commitment:
            raise ValueError('invalid street commitments')
        if self.remaining_stack < 0 or self.big_blind <= 0 or self.last_full_raise < self.big_blind:
            raise ValueError('invalid stack, blind, or last full raise')

    @property
    def amount_to_call(self) -> int:
        return min(self.remaining_stack, self.highest_commitment - self.actor_commitment)

    @property
    def max_total(self) -> int:
        return self.actor_commitment + self.remaining_stack

    @property
    def min_full_raise_to(self) -> int:
        if self.highest_commitment == 0:
            return self.big_blind
        return self.highest_commitment + self.last_full_raise

    @property
    def can_raise(self) -> bool:
        return self.may_raise and self.max_total > self.highest_commitment


def validate_raise_to(ctx: BetContext, total: int) -> str:
    """Validate total street commitment and classify full vs short all-in.

    A short all-in raise is allowed but does not automatically reopen betting.
    Caller tracks which opponents have already acted and `may_raise`.
    """
    if not isinstance(total, int):
        raise TypeError('raise-to amount must be an integer')
    if not ctx.may_raise:
        raise ValueError('betting has not reopened for this player')
    if total <= ctx.highest_commitment:
        raise ValueError('raise must exceed the current highest bet')
    if total > ctx.max_total:
        raise ValueError('raise exceeds available stack')
    if total >= ctx.min_full_raise_to:
        return 'full_raise'
    if total == ctx.max_total:
        return 'short_all_in_raise'
    raise ValueError(f'raise below minimum {ctx.min_full_raise_to}; only all-in is allowed')


def legal_basic_actions(ctx: BetContext) -> tuple[str, ...]:
    """Reference actions; separates check and call unlike RLCard's CHECK_CALL."""
    if ctx.remaining_stack == 0:
        return ()
    need = ctx.highest_commitment - ctx.actor_commitment
    options = ['fold', 'check' if need == 0 else 'call']
    if ctx.can_raise:
        options.append('raise')
    return tuple(options)
