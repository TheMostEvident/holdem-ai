"""Mask-correct policies usable by random, CFR and neural learners."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from holdem_ai.core.schema import InformationState


class AgentPolicy(Protocol):
    def act(self, observation: InformationState, rng: random.Random) -> int: ...


def masked_distribution(values: Sequence[float], mask: Sequence[bool]) -> tuple[float, ...]:
    """Zero invalid actions, normalize valid nonnegative weights.

    All-zero legal weights use a uniform fallback; nonfinite/negative weights
    raise instead of silently corrupting training probabilities.
    """
    if len(values) != len(mask) or not any(mask):
        raise ValueError("probability vector and nonempty legal mask must align")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in values):
        raise ValueError("weights must be finite and nonnegative")
    masked = [float(v) if allowed else 0.0 for v, allowed in zip(values, mask)]
    denom = math.fsum(masked)
    if denom == 0:
        legal_count = sum(mask)
        return tuple(1.0 / legal_count if allowed else 0.0 for allowed in mask)
    return tuple(v / denom for v in masked)


def sample_distribution(probabilities: Sequence[float], rng: random.Random) -> int:
    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    if any(v < 0 or not math.isfinite(v) for v in probabilities):
        raise ValueError("invalid probabilities")
    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-8):
        raise ValueError("probabilities must sum to one")
    r = rng.random()
    running = 0.0
    last_positive = -1
    for i, prob in enumerate(probabilities):
        if prob > 0:
            last_positive = i
        running += prob
        if r < running and prob > 0:
            return i
    return last_positive  # protects against floating-point sum rounding


@dataclass(frozen=True, slots=True)
class RandomLegalPolicy:
    def act(self, observation: InformationState, rng: random.Random) -> int:
        return rng.choice(observation.legal_actions)


@dataclass(frozen=True, slots=True)
class CFRTablePolicy:
    """Read-only tabular strategy; action probabilities indexed by infoset_key."""

    strategy: Mapping[str, Sequence[float]]

    def act(self, observation: InformationState, rng: random.Random) -> int:
        weights = self.strategy.get(observation.infoset_key(), (0.0,) * len(observation.legal_action_mask))
        return sample_distribution(masked_distribution(weights, observation.legal_action_mask), rng)


@dataclass(frozen=True, slots=True)
class EpsilonGreedyPolicy:
    q_fn: Callable[[InformationState], Sequence[float]]
    epsilon: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

    def act(self, observation: InformationState, rng: random.Random) -> int:
        legal = observation.legal_actions
        if rng.random() < self.epsilon:
            return rng.choice(legal)
        scores = self.q_fn(observation)
        if len(scores) != len(observation.legal_action_mask):
            raise ValueError("Q-vector dimension must match action mask")
        if any(not math.isfinite(float(scores[i])) for i in legal):
            raise ValueError("legal Q-values must be finite")
        return max(legal, key=lambda idx: (scores[idx], -idx))
