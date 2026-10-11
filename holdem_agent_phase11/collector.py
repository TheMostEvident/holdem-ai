"""Batched (multi-hand, single-process) on-policy GPU inference for HUNL200.

Every hand remains an independent engine instance; a single frozen Actor
processes the *next* observation of each unfinished hand as a batch.
No opponent cards enter ActorBatch. The same policy version acts on both seats.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from holdem_agent_phase5.schema import Terminal, Turn
from holdem_agent_phase6 import HUNL200Adapter
from holdem_agent_phase7 import batch_actors, encode_actor, encode_critic
from holdem_agent_phase8 import TransformerActor
from holdem_agent_phase9.ppo import PPOConfig, TrajectoryRows


@dataclass
class _Hand:
    index: int
    env: HUNL200Adapter
    rng: random.Random
    outcome: Turn | Terminal
    actors: list = field(default_factory=list)
    critics: list = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    seats: list[int] = field(default_factory=list)
    rewards: tuple[float, float] | None = None


def collect_batched_self_play(
    actor: TransformerActor, config: PPOConfig, *, iteration: int = 0,
    device: str = 'cpu', env_batch_size: int = 8,
) -> TrajectoryRows:
    """Batch inference across independently evolving hands; preserve on-policy sampling.

    At most ``env_batch_size`` hands live in memory simultaneously.  Every
    unfinished hand advances exactly ONE engine decision per forward batch.
    Returns are assigned only after its terminal state is observed. ``index``
    and shuffle seed match Phase9's deterministic per-hand scheduling.
    """
    if type(iteration) is not int or iteration < 0:
        raise ValueError('iteration must be a nonnegative integer')
    if type(env_batch_size) is not int or env_batch_size < 1:
        raise ValueError('env_batch_size must be a positive integer')
    if actor.config.dropout != 0:
        raise ValueError('on-policy collector requires deterministic dropout=0')
    actor = actor.to(device).eval()
    all_actors, all_critics, all_actions, all_seats, all_returns = [], [], [], [], []
    all_hand_rewards = []
    for offset in range(0, config.hands_per_update, env_batch_size):
        hands: list[_Hand] = []
        for k in range(offset, min(offset + env_batch_size, config.hands_per_update)):
            index = iteration * config.hands_per_update + k
            seed = config.seed + index
            env = HUNL200Adapter(
                button=index % 2, big_blind=config.big_blind,
                expose_training_state=True,
            )
            outcome = env.reset(seed=seed)
            hands.append(_Hand(index, env, random.Random(seed), outcome))
        while True:
            active = [h for h in hands if isinstance(h.outcome, Turn)]
            if not active:
                break
            actor_rows = [
                encode_actor(h.outcome.observation, big_blind=config.big_blind,
                             max_tokens=actor.config.max_tokens)
                for h in active
            ]
            # Critic states may contain opponent's hole cards: never part of batch_actors.
            critic_rows = [
                encode_critic(h.env.training_state(), big_blind=config.big_blind,
                              max_tokens=actor.config.max_tokens)
                for h in active
            ]
            with torch.inference_mode():
                batch = batch_actors(actor_rows, device=device)
                probabilities = actor.distribution(batch).probs.detach().cpu().tolist()
            for hand, encoded_actor, encoded_critic, probs in zip(
                active, actor_rows, critic_rows, probabilities
            ):
                assert isinstance(hand.outcome, Turn)
                current = hand.outcome.observation
                action = hand.rng.choices(range(7), weights=probs, k=1)[0]
                current.assert_legal(action)
                if len(hand.actions) >= config.max_decisions:
                    raise RuntimeError('max_decisions exceeded in batched rollout')
                hand.actors.append(encoded_actor)
                hand.critics.append(encoded_critic)
                hand.seats.append(current.player_id)
                hand.actions.append(action)
                hand.outcome = hand.env.step(action)
                if isinstance(hand.outcome, Terminal):
                    rewards = tuple(float(x) for x in hand.outcome.rewards)
                    if len(rewards) != 2 or not math.isclose(sum(rewards), 0.0, abs_tol=1e-8):
                        raise AssertionError('non-zero-sum terminal reward')
                    hand.rewards = rewards
        for hand in hands:
            if hand.rewards is None or not hand.actions:
                raise AssertionError('incomplete hand')
            all_actors.extend(hand.actors)
            all_critics.extend(hand.critics)
            all_actions.extend(hand.actions)
            all_seats.extend(hand.seats)
            all_returns.extend(hand.rewards[seat] for seat in hand.seats)
            all_hand_rewards.append(hand.rewards)
    return TrajectoryRows(
        tuple(all_actors), tuple(all_critics), tuple(all_actions),
        tuple(all_seats), tuple(all_returns), config.hands_per_update,
        tuple(all_hand_rewards),
    )
