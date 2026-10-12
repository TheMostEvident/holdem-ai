"""Protocol boundary for E7; E1 deliberately does not simulate tournaments."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TournamentRuleSet:
    late_registration_enabled: bool = False
    reentry_enabled: bool = False
    ante_chips: int = 0

    def __post_init__(self):
        if type(self.ante_chips) is not int or self.ante_chips < 0:
            raise ValueError('ante must be nonnegative integer')

    def require_implemented(self):
        raise NotImplementedError('Tournament operation is reserved for Expand Phase 7')
