"""Seat/button-balanced held-out tests vs Random and a frozen anchor Actor."""
from __future__ import annotations

import math

import torch

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode
from holdem_ai.agents.neural_policy import NeuralActorPolicy
from holdem_ai.models.transformer import TransformerActor


def _summary(payoffs: list[float]) -> dict[str, float | int]:
    n = len(payoffs)
    mu = sum(payoffs) / n
    var = sum((x - mu) ** 2 for x in payoffs) / (n - 1)
    half_width = 1.96 * math.sqrt(var / n)
    return {
        'hands': n, 'mean_bb100': round(100 * mu, 4),
        'ci95_lower_bb100': round(100 * (mu - half_width), 4),
        'ci95_upper_bb100': round(100 * (mu + half_width), 4),
        'positive_hand_fraction': round(sum(x > 0 for x in payoffs) / n, 4),
        'seat0_hands': n // 2, 'seat1_hands': n // 2,
        'neural_button_hands': n // 2, 'neural_big_blind_hands': n // 2,
    }


def evaluate_balanced(
    actor: TransformerActor, *, opponent: TransformerActor | None = None,
    hands: int = 40, seed: int = 1_200_000, device: str = 'cpu',
    big_blind: int = 100,
) -> dict[str, float | int | str]:
    """Greedy actor versus random or frozen actor, full four seat×button layouts.

    The opponent is a fixed snapshot and must not share trainable parameters.
    Seeds and layouts are fixed across runs; this is not a true paired common-
    random-numbers counterfactual and does not estimate GTO exploitability.
    """
    if type(hands) is not int or hands < 4 or hands % 4:
        raise ValueError('evaluation hands must be a positive multiple of four')
    if type(seed) is not int:
        raise TypeError('seed must be integer')
    if opponent is actor:
        raise ValueError('opponent must be a frozen independent snapshot, not the same Actor')
    was_training = actor.training
    opponent_was_training = opponent.training if opponent is not None else None
    agent = NeuralActorPolicy(actor, device=device, big_blind=big_blind, greedy=True)
    other = (NeuralActorPolicy(opponent, device=device, big_blind=big_blind, greedy=True)
             if opponent is not None else RandomLegalPolicy())
    payoffs = []
    try:
        for idx in range(hands):
            actor_seat = idx % 2
            button = (idx // 2) % 2
            env = HUNL200Adapter(button=button, big_blind=big_blind)
            seats = {actor_seat: agent, 1 - actor_seat: other}
            # Changing policy can change future streets, even with equal deck seed.
            episode = collect_global_episode(env, seats, seed=seed + idx,
                                             privileged=False)
            payoffs.append(episode.terminal_rewards[actor_seat])
    finally:
        actor.train(was_training)
        if opponent is not None:
            opponent.train(opponent_was_training)
    return {'opponent': 'frozen_anchor' if opponent is not None else 'random',
            **_summary(payoffs)}
