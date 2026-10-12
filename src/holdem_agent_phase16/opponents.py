"""Deterministic benchmark opponents; all actions are legal by construction.

These are diagnostic baselines, not strategically strong poker opponents.
"""
from __future__ import annotations

from dataclasses import dataclass

from holdem_agent_phase5 import RandomLegalPolicy
from holdem_agent_phase5.schema import InformationState

BUILTIN_OPPONENTS = ('random', 'calling_station', 'tight', 'pot_pressure')


class CallingStation:
    """Check if possible, otherwise call; otherwise choose a legal fallback."""
    def act(self, observation: InformationState, rng) -> int:
        legal = observation.legal_actions
        for action in (1, 2, 0, 3, 4, 5, 6):
            if action in legal:
                return action
        raise ValueError('no legal actions')


class TightPolicy:
    """Check for free; otherwise fold when possible. A deliberately exploitable baseline."""
    def act(self, observation: InformationState, rng) -> int:
        legal = observation.legal_actions
        for action in (1, 0, 2, 3, 4, 5, 6):
            if action in legal:
                return action
        raise ValueError('no legal actions')


class PotPressure:
    """Prefer pot-sized raises; short-list smaller raises before passive actions."""
    def act(self, observation: InformationState, rng) -> int:
        legal = observation.legal_actions
        for action in (4, 5, 3, 6, 2, 1, 0):
            if action in legal:
                return action
        raise ValueError('no legal actions')


def build_builtin(name: str):
    if name == 'random':
        return RandomLegalPolicy()
    if name == 'calling_station':
        return CallingStation()
    if name == 'tight':
        return TightPolicy()
    if name == 'pot_pressure':
        return PotPressure()
    raise ValueError(f'unknown built-in opponent: {name}')


@dataclass(frozen=True)
class OpponentSpec:
    name: str
    checkpoint: str | None = None

    def __post_init__(self):
        if not self.name or not self.name.replace('_', '').replace('-', '').isalnum():
            raise ValueError('opponent name must consist of letters, digits, dash, underscore')
        if self.checkpoint is None and self.name not in BUILTIN_OPPONENTS:
            raise ValueError(f'unknown built-in opponent: {self.name}')
        if self.checkpoint is not None and not self.checkpoint:
            raise ValueError('opponent checkpoint cannot be empty')
