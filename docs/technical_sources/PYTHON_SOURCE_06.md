# Python source code: volume 6 of 28

Exact full .py sources from baseline commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. Integral appendix of [final technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md).

## src/holdem_ai/agents/__init__.py

Original Git blob SHA: `5cf12807f331a109f29539a9ae0d1b1e57a687ee` (40 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/agents/__init__.py" sha="5cf12807f331a109f29539a9ae0d1b1e57a687ee" -->

````````python
"""Policy and agent implementations."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/agents/__init__.py" -->

## src/holdem_ai/agents/base_agent.py

Original Git blob SHA: `a07fe762ef949ec625f9bec934e5da236c9f13b7` (460 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/agents/base_agent.py" sha="a07fe762ef949ec625f9bec934e5da236c9f13b7" -->

````````python
"""Common interface for poker agents.

Agents must act only on the acting player's observable RLCard state.
The engine does not expose the opponent's private hole cards.
"""

from typing import Any, Mapping, Protocol


class Agent(Protocol):
    """Any agent implementing select_action can play via HandEngine."""

    def select_action(self, state: Mapping[str, Any]) -> int:
        """Choose one of the integer IDs in state['legal_actions']."""
        ...

````````

<!-- END_PY_SOURCE path="src/holdem_ai/agents/base_agent.py" -->

## src/holdem_ai/agents/cfr_agent.py

Original Git blob SHA: `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (0 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/agents/cfr_agent.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="src/holdem_ai/agents/cfr_agent.py" -->

## src/holdem_ai/agents/diagnostic_opponents.py

Original Git blob SHA: `ac3e65a257942b2303a34feffb75e497ac85edf8` (2418 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/agents/diagnostic_opponents.py" sha="ac3e65a257942b2303a34feffb75e497ac85edf8" -->

````````python
"""Deterministic benchmark opponents; all actions are legal by construction.

These are diagnostic baselines, not strategically strong poker opponents.
"""
from __future__ import annotations

from dataclasses import dataclass

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.core.schema import InformationState

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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/agents/diagnostic_opponents.py" -->

## src/holdem_ai/agents/neural_policy.py

Original Git blob SHA: `e969f8ed3f76ff73265cfcee9de8dbfd31714806` (1680 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/agents/neural_policy.py" sha="e969f8ed3f76ff73265cfcee9de8dbfd31714806" -->

````````python
"""Inference-only adapter implementing the Phase-5 AgentPolicy interface."""
from __future__ import annotations

import random

import torch

from holdem_ai.core.schema import InformationState
from holdem_ai.encoding.tensors import batch_actors
from holdem_ai.encoding.observation import encode_actor
from holdem_ai.models.transformer import TransformerActor


class NeuralActorPolicy:
    """Only observable InformationState reaches TransformerActor.

    The provided Python RNG is used for reproducible action sampling.
    Running GPU inference per decision is intentionally simple; a batched
    inference server should replace it before large-scale RL self-play.
    """

    def __init__(self, actor: TransformerActor, *, device: str = 'cpu',
                 big_blind: int = 100, max_tokens: int | None = None,
                 greedy: bool = False) -> None:
        self.actor = actor.to(device).eval()
        self.device = torch.device(device)
        self.big_blind = big_blind
        self.max_tokens = actor.config.max_tokens if max_tokens is None else max_tokens
        self.greedy = greedy

    @torch.inference_mode()
    def act(self, observation: InformationState, rng: random.Random) -> int:
        encoded = encode_actor(
            observation, big_blind=self.big_blind, max_tokens=self.max_tokens,
        )
        batch = batch_actors([encoded], device=self.device)
        probabilities = self.actor.distribution(batch).probs[0].detach().cpu().tolist()
        if self.greedy:
            return max(observation.legal_actions, key=lambda i: (probabilities[i], -i))
        return rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]

````````

<!-- END_PY_SOURCE path="src/holdem_ai/agents/neural_policy.py" -->

## src/holdem_ai/agents/policies.py

Original Git blob SHA: `862a4fd5cf52f9d8aaab456e683a97aefa7538f8` (3504 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/agents/policies.py" sha="862a4fd5cf52f9d8aaab456e683a97aefa7538f8" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/agents/policies.py" -->

## src/holdem_ai/agents/random_agent.py

Original Git blob SHA: `e2f00722a39a86e9a4b66e9edb576d8a9675a551` (518 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/agents/random_agent.py" sha="e2f00722a39a86e9a4b66e9edb576d8a9675a551" -->

````````python
"""Random legal-action baseline, for testing the environment/agent boundary."""

import random
from typing import Any, Mapping


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: Mapping[str, Any]) -> int:
        legal_actions = tuple(state['legal_actions'])
        if not legal_actions:
            raise ValueError('There are no legal actions in this state.')
        return int(self._rng.choice(legal_actions))

````````

<!-- END_PY_SOURCE path="src/holdem_ai/agents/random_agent.py" -->

## src/holdem_ai/agents/rl_agent.py

Original Git blob SHA: `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (0 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/agents/rl_agent.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="src/holdem_ai/agents/rl_agent.py" -->

## src/holdem_ai/core/__init__.py

Original Git blob SHA: `c45dbcdd525a3fadb2abb368e5264f09bb51e773` (39 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/core/__init__.py" sha="c45dbcdd525a3fadb2abb368e5264f09bb51e773" -->

````````python
"""Consolidated functional package."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/core/__init__.py" -->

## src/holdem_ai/core/schema.py

Original Git blob SHA: `a0f99927c6d91b36893b5e72451c0c93a88f01fa` (3547 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/core/schema.py" sha="a0f99927c6d91b36893b5e72451c0c93a88f01fa" -->

````````python
"""Framework-neutral, seat-relative agent decision contract (Python >= 3.11)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Protocol


@dataclass(frozen=True, slots=True)
class InformationState:
    """Only information available to ``player_id`` at the decision instant.

    ``features`` must be generated by the environment's public player-observation
    API, never by flattening an omniscient game object. ``public_state`` stores
    (name, JSON-encoded value) pairs, not references to mutable engine objects.
    ``history`` must be the complete *observable* action history for perfect
    recall CFR; empty history is permitted for model-free RL baselines only.
    """

    player_id: int
    features: tuple[float, ...]
    legal_action_mask: tuple[bool, ...]
    private_cards: tuple[str, ...] = ()
    public_cards: tuple[str, ...] = ()
    stage: str = "unknown"
    public_state: tuple[tuple[str, str], ...] = ()
    history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.player_id < 0:
            raise ValueError("player_id must be nonnegative")
        if not self.legal_action_mask or not any(self.legal_action_mask):
            raise ValueError("nonterminal InformationState requires legal actions")
        if any(not isinstance(x, bool) for x in self.legal_action_mask):
            raise TypeError("legal_action_mask must contain booleans")
        if any(not math.isfinite(float(x)) for x in self.features):
            raise ValueError("features must be finite")

    @property
    def legal_actions(self) -> tuple[int, ...]:
        return tuple(i for i, allowed in enumerate(self.legal_action_mask) if allowed)

    def assert_legal(self, action: int) -> None:
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("action index must be an integer")
        if action < 0 or action >= len(self.legal_action_mask) or not self.legal_action_mask[action]:
            raise ValueError(f"illegal action {action}; legal={self.legal_actions}")

    def infoset_key(self) -> str:
        """Stable serialization; suitable for tabular CFR only with full recall history."""
        return json.dumps(
            {
                "player": self.player_id,
                "private": self.private_cards,
                "public": self.public_cards,
                "stage": self.stage,
                "public_state": self.public_state,
                "history": self.history,
                "features": self.features,
                "mask": self.legal_action_mask,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


@dataclass(frozen=True, slots=True)
class Turn:
    observation: InformationState

    @property
    def player_id(self) -> int:
        return self.observation.player_id


@dataclass(frozen=True, slots=True)
class Terminal:
    """Terminal utilities in seat index order; always one scalar per seat."""

    rewards: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.rewards) < 2 or any(not math.isfinite(x) for x in self.rewards):
            raise ValueError("terminal reward vector must have >=2 finite entries")


class TurnBasedEnvironment(Protocol):
    """Minimal adapter boundary: one acting seat per step, terminal-only reward."""

    def reset(self, seed: int | None = None) -> Turn | Terminal: ...

    def step(self, action_index: int) -> Turn | Terminal: ...

````````

<!-- END_PY_SOURCE path="src/holdem_ai/core/schema.py" -->

## src/holdem_ai/encoding/__init__.py

Original Git blob SHA: `c45dbcdd525a3fadb2abb368e5264f09bb51e773` (39 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/encoding/__init__.py" sha="c45dbcdd525a3fadb2abb368e5264f09bb51e773" -->

````````python
"""Consolidated functional package."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/encoding/__init__.py" -->

## src/holdem_ai/encoding/actions.py

Original Git blob SHA: `772da4ab52aba2dab7b8105cb2dcf45670ad6637` (1632 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/encoding/actions.py" sha="772da4ab52aba2dab7b8105cb2dcf45670ad6637" -->

````````python
"""Phase-5 compatible fixed 7-head action encoding and masked distributions.

This is VRPO-Lite v1. It is NOT the paper's base-16 multistep action codec.
"""
from __future__ import annotations

from holdem_ai.environment.bet_abstraction import BET_ACTION_LABELS

NUM_ACTIONS = len(BET_ACTION_LABELS)


def encode_action(action_index: int, legal_mask: tuple[bool, ...]) -> tuple[float, ...]:
    if len(legal_mask) != NUM_ACTIONS or not any(legal_mask) or any(type(x) is not bool for x in legal_mask):
        raise ValueError('expected nonempty 7-action bool mask')
    if type(action_index) is not int or not 0 <= action_index < NUM_ACTIONS or not legal_mask[action_index]:
        raise ValueError('action is illegal or out of range')
    return tuple(float(index == action_index) for index in range(NUM_ACTIONS))


def masked_logits(logits, legal_mask):
    """Torch [B,7] logits and bool mask; illegal gradients and probs vanish."""
    import torch

    if logits.ndim != 2 or logits.shape[1] != NUM_ACTIONS:
        raise ValueError('logits must be [B,7]')
    if legal_mask.shape != logits.shape or legal_mask.dtype != torch.bool:
        raise ValueError('mask must be bool [B,7]')
    if not bool(legal_mask.any(dim=-1).all().item()):
        raise ValueError('each observation needs a legal action')
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError('nonfinite logits')
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


def masked_categorical(logits, legal_mask):
    import torch

    return torch.distributions.Categorical(logits=masked_logits(logits, legal_mask))

````````

<!-- END_PY_SOURCE path="src/holdem_ai/encoding/actions.py" -->
