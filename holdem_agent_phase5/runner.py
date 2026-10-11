"""Turn-based multiagent self-play, one transition per decision by each player."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from .policy import AgentPolicy
from .schema import InformationState, Terminal, Turn, TurnBasedEnvironment


@dataclass(frozen=True, slots=True)
class Transition:
    decision_index: int
    seat: int
    observation: InformationState
    action_index: int
    next_observation: InformationState | None
    reward: float
    done: bool


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    rewards: tuple[float, ...]
    transitions: tuple[Transition, ...]
    num_decisions: int

    def for_seat(self, seat: int) -> tuple[Transition, ...]:
        return tuple(t for t in self.transitions if t.seat == seat)


def run_self_play(
    environment: TurnBasedEnvironment,
    policies: Mapping[int, AgentPolicy],
    *,
    seed: int | None = None,
    max_decisions: int = 100_000,
) -> EpisodeResult:
    """Collect trajectories; next_obs is the NEXT turn of the SAME seat.

    Intermediate reward is zero by contract; only the pending last transition
    per seat receives its terminal utility. Seat IDs must be stable throughout
    the episode, including after tournament elimination.
    """
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    rng = random.Random(seed)
    outcome = environment.reset(seed=seed)
    pending: dict[int, tuple[int, InformationState, int]] = {}
    completed: list[Transition] = []
    decisions = 0
    while isinstance(outcome, Turn):
        obs = outcome.observation
        if obs.player_id not in policies:
            raise KeyError(f"no policy for seat {obs.player_id}")
        if obs.player_id in pending:
            old_index, old_obs, old_action = pending.pop(obs.player_id)
            completed.append(Transition(old_index, obs.player_id, old_obs, old_action, obs, 0.0, False))
        if decisions >= max_decisions:
            raise RuntimeError("max_decisions exhausted; possible nonterminating game")
        action = policies[obs.player_id].act(obs, rng)
        obs.assert_legal(action)
        pending[obs.player_id] = (decisions, obs, action)
        outcome = environment.step(action)
        decisions += 1
    if not isinstance(outcome, Terminal):
        raise TypeError("environment must return Turn or Terminal")
    if any(seat >= len(outcome.rewards) for seat in pending):
        raise ValueError("terminal rewards missing a seat")
    for seat, (index, obs, action) in pending.items():
        completed.append(Transition(index, seat, obs, action, None, outcome.rewards[seat], True))
    if any(not math.isfinite(r) for r in outcome.rewards):
        raise ValueError("nonfinite terminal reward")
    return EpisodeResult(outcome.rewards, tuple(sorted(completed, key=lambda t: t.decision_index)), decisions)
