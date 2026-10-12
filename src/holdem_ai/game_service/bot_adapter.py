"""Apply any Phase-5 AgentPolicy to a bot turn, including NeuralActorPolicy.

The AgentPolicy receives only HUNL200Adapter's acting-player InformationState,
never an engine reference, privileged CriticState, or human hole cards.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from holdem_ai.core.schema import InformationState


class GUIBotPolicy(Protocol):
    def act(self, observation: InformationState, rng: random.Random) -> int: ...


@dataclass(slots=True)
class BotActionAdapter:
    policy: GUIBotPolicy
    rng: random.Random

    def choose(self, observation: InformationState) -> int:
        action = self.policy.act(observation, self.rng)
        observation.assert_legal(action)
        return action
