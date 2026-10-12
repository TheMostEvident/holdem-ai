"""Training-time opponent selection, separate from final evaluation opponents."""
from __future__ import annotations

import random
from dataclasses import dataclass

from holdem_ai.expand.information.schema import InformationState, ACTION_NAMES


@dataclass(frozen=True)
class OpponentMix:
    shared_policy_fraction: float = 0.25
    snapshot_fraction: float = 0.25
    scripted_names: tuple[str, ...] = ('random', 'calling_station', 'tight', 'pot_pressure')

    def __post_init__(self):
        if not (0 <= self.shared_policy_fraction <= 1 and 0 <= self.snapshot_fraction <= 1
                and self.shared_policy_fraction + self.snapshot_fraction <= 1):
            raise ValueError('fractions must be nonnegative with sum <=1')
        if not self.scripted_names or any(x not in ('random','calling_station','tight','pot_pressure')
                                           for x in self.scripted_names):
            raise ValueError('unsupported scripted policy')


def scripted_action(name: str, state: InformationState, rng: random.Random) -> str:
    legal = [a for a, available in zip(ACTION_NAMES, state.legal_actions.mask) if available]
    if not legal:
        raise ValueError('no legal action')
    if name == 'random':
        return rng.choice(legal)
    if name == 'calling_station':
        order = ('check', 'call', 'fold')
    elif name == 'tight':
        order = ('check', 'fold', 'call')
    elif name == 'pot_pressure':
        order = ('raise_pot', 'raise_double_pot', 'raise_half_pot', 'all_in', 'call', 'check', 'fold')
    else:
        raise ValueError('unknown scripted opponent')
    return next((a for a in order if a in legal), legal[0])


def choose_role(rng: random.Random, mix: OpponentMix, *, has_snapshots: bool) -> str:
    u = rng.random()
    if u < mix.shared_policy_fraction:
        return 'trainable'
    if u < mix.shared_policy_fraction + mix.snapshot_fraction and has_snapshots:
        return 'snapshot'
    return 'scripted:' + rng.choice(mix.scripted_names)
