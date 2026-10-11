"""Global-decision trajectories for future CTDE / VRPO learners (no training yet).

Unlike Phase-5 same-seat transitions, adjacent entries here are consecutive
GLOBAL poker decisions. The terminal reward vector is recorded on the final
step; all earlier step rewards are zero. Chance dealing between decisions is
included in the engine transition, not split into a standalone action.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

from holdem_agent_phase5.policy import AgentPolicy
from holdem_agent_phase5.schema import InformationState, Terminal, Turn
from .adapter import CriticState, HUNL200Adapter


@dataclass(frozen=True, slots=True)
class GlobalDecision:
    time_index: int
    player_id: int
    observation: InformationState
    action_index: int
    critic_state: CriticState | None
    next_critic_state: CriticState | None
    next_player_id: int | None
    reward_vector: tuple[float, float]
    done: bool


@dataclass(frozen=True, slots=True)
class GlobalEpisode:
    decisions: tuple[GlobalDecision, ...]
    terminal_rewards: tuple[float, float]


def collect_global_episode(
    environment: HUNL200Adapter,
    policies: Mapping[int, AgentPolicy],
    *, seed: int | None = None, max_decisions: int = 200,
    privileged: bool = False,
) -> GlobalEpisode:
    """Records per-global-step data; policies ALWAYS receive InformationState only."""
    if type(max_decisions) is not int or max_decisions <= 0:
        raise ValueError("max_decisions must be positive integer")
    if privileged and not environment.expose_training_state:
        raise PermissionError("environment must explicitly enable privileged data")
    rng = random.Random(seed)
    outcome = environment.reset(seed=seed)
    history: list[GlobalDecision] = []
    while isinstance(outcome, Turn):
        if len(history) >= max_decisions:
            raise RuntimeError("max_decisions exceeded")
        observation = outcome.observation
        seat = observation.player_id
        if seat not in policies:
            raise KeyError(f"missing policy for seat={seat}")
        before = environment.training_state() if privileged else None
        action = policies[seat].act(observation, rng)
        observation.assert_legal(action)
        next_outcome = environment.step(action)
        after = environment.training_state() if privileged else None
        ended = isinstance(next_outcome, Terminal)
        reward = tuple(next_outcome.rewards) if ended else (0.0, 0.0)
        history.append(GlobalDecision(
            time_index=len(history), player_id=seat, observation=observation,
            action_index=action, critic_state=before,
            next_critic_state=after,
            next_player_id=None if ended else next_outcome.player_id,
            reward_vector=reward, done=ended,
        ))
        outcome = next_outcome
    if not isinstance(outcome, Terminal):
        raise TypeError("environment must return Turn or Terminal")
    return GlobalEpisode(tuple(history), (float(outcome.rewards[0]), float(outcome.rewards[1])))
