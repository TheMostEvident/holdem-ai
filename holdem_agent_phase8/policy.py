"""Inference-only adapter implementing the Phase-5 AgentPolicy interface."""
from __future__ import annotations

import random

import torch

from holdem_agent_phase5.schema import InformationState
from holdem_agent_phase7 import batch_actors, encode_actor
from .network import TransformerActor


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
