"""On-policy PPO baseline for Phase-8 HUNL200 networks (NOT VRPO/Q-Boosting).

Poker is undiscounted, zero-sum, and terminal-reward-only: every decision by
seat i in a finished hand has return G_i equal to that seat's final payoff BB.
A privileged Q-critic is allowed to construct a TRAINING-ONLY, action-independent
state baseline V_i(s) = sum_a pi_old(a|I_i) * Q_i(s,a). No critic state enters
Actor.forward or NeuralActorPolicy.act. Fitted Q targets only observed actions:
there are no counterfactual action labels in a single self-play trajectory.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from holdem_agent_phase5.policy import RandomLegalPolicy
from holdem_agent_phase6 import HUNL200Adapter, collect_global_episode
from holdem_agent_phase7 import batch_actors, batch_critics, encode_actor, encode_critic
from holdem_agent_phase7.encoding import EncodedActor, EncodedCritic
from holdem_agent_phase8 import (CentralizedQCritic, NeuralActorPolicy,
                                 TransformerActor, chosen_q)


@dataclass(frozen=True)
class PPOConfig:
    hands_per_update: int = 16
    epochs: int = 2
    minibatch_size: int = 32
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 0.5
    inference_batch_size: int = 64
    max_decisions: int = 200
    big_blind: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        for field in ('hands_per_update', 'epochs', 'minibatch_size',
                      'inference_batch_size', 'max_decisions', 'big_blind'):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f'{field} must be a positive integer')
        if type(self.seed) is not int:
            raise ValueError('seed must be an integer')
        for field in ('learning_rate', 'clip_epsilon', 'value_coef',
                      'entropy_coef', 'grad_clip'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{field} must be a finite number')
        if self.learning_rate <= 0 or not 0 < self.clip_epsilon < 1 or self.value_coef < 0 or self.entropy_coef < 0 or self.grad_clip <= 0:
            raise ValueError('invalid PPO hyperparameters')


@dataclass(frozen=True)
class TrajectoryRows:
    actors: tuple[EncodedActor, ...]
    critics: tuple[EncodedCritic, ...]
    actions: tuple[int, ...]
    acting_seats: tuple[int, ...]
    returns_bb: tuple[float, ...]
    hands: int
    hand_rewards: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        n = len(self.actions)
        if n == 0 or not all(len(x) == n for x in (
                self.actors, self.critics, self.acting_seats, self.returns_bb)):
            raise ValueError('trajectory rows must be nonempty and aligned')
        if self.hands != len(self.hand_rewards):
            raise ValueError('hand rewards must align')
        if any(i not in (0, 1) for i in self.acting_seats):
            raise ValueError('invalid acting seat')
        for row, action in zip(self.actors, self.actions):
            if type(action) is not int or action not in range(7) or not row.legal_action_mask[action]:
                raise ValueError('illegal trajectory action')
        if not all(math.isfinite(x) for x in self.returns_bb):
            raise ValueError('nonfinite trajectory return')


@dataclass(frozen=True)
class FrozenPPOData:
    rows: TrajectoryRows
    old_log_probs: torch.Tensor  # CPU float32 [N]
    advantages: torch.Tensor    # CPU float32 [N], normalized
    baselines: torch.Tensor     # CPU float32 [N], old policy/critic


def collect_self_play(actor: TransformerActor, config: PPOConfig, *,
                      iteration: int = 0, device: str = 'cpu') -> TrajectoryRows:
    """Collect a single fully-frozen policy version. Seat/button alternate.

    GPU actor inference is one decision at a time for correctness. This path
    is NOT a parallel/batched inference server and should not inherit the
    Phase-6 CPU-only hands/sec benchmark estimate.
    """
    if type(iteration) is not int or iteration < 0:
        raise ValueError('iteration must be nonnegative')
    policy = NeuralActorPolicy(actor, device=device, big_blind=config.big_blind,
                               max_tokens=actor.config.max_tokens, greedy=False)
    policies = {0: policy, 1: policy}
    actors: list[EncodedActor] = []
    critics: list[EncodedCritic] = []
    actions: list[int] = []
    seats: list[int] = []
    targets: list[float] = []
    hand_rewards: list[tuple[float, float]] = []
    # Seed choice is independent of optimizer/random minibatch RNG.
    for hand in range(config.hands_per_update):
        index = iteration * config.hands_per_update + hand
        seed = config.seed + index
        env = HUNL200Adapter(button=index % 2,
                             big_blind=config.big_blind,
                             expose_training_state=True)
        episode = collect_global_episode(
            env, policies, seed=seed, privileged=True,
            max_decisions=config.max_decisions,
        )
        if not math.isclose(sum(episode.terminal_rewards), 0.0, abs_tol=1e-8):
            raise AssertionError('hand is not zero-sum')
        hand_rewards.append(tuple(episode.terminal_rewards))
        for decision in episode.decisions:
            if decision.critic_state is None:
                raise AssertionError('collector did not provide privileged state')
            actors.append(encode_actor(decision.observation,
                                      big_blind=config.big_blind,
                                      max_tokens=actor.config.max_tokens))
            critics.append(encode_critic(decision.critic_state,
                                        big_blind=config.big_blind,
                                        max_tokens=actor.config.max_tokens))
            actions.append(decision.action_index)
            seats.append(decision.player_id)
            targets.append(episode.terminal_rewards[decision.player_id])
    return TrajectoryRows(tuple(actors), tuple(critics), tuple(actions),
                          tuple(seats), tuple(targets), config.hands_per_update,
                          tuple(hand_rewards))


@torch.no_grad()
def freeze_old_policy_data(actor: TransformerActor, critic: CentralizedQCritic,
                           rows: TrajectoryRows, *, device: str = 'cpu',
                           chunk_size: int = 64) -> FrozenPPOData:
    """Freeze log(pi_old(a|I)) and privileged action-INDEPENDENT baseline.

    Baseline is a state function for the observed actor infoset because the
    expectation sums over every legal action. Using Q(s, selected_action)
    itself would leak the sampled action into the advantage baseline.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    actor.eval()
    critic.eval()
    old_logs, baseline_values = [], []
    n = len(rows.actions)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        ab = batch_actors(rows.actors[start:stop], device=device)
        cb = batch_critics(rows.critics[start:stop], device=device)
        actions = torch.tensor(rows.actions[start:stop], dtype=torch.long, device=device)
        seats = torch.tensor(rows.acting_seats[start:stop], dtype=torch.long, device=device)
        dist = actor.distribution(ab)
        q = critic(cb)
        indexes = torch.arange(stop - start, device=device)
        v = (dist.probs * q[indexes, seats]).sum(dim=1)
        old_logs.append(dist.log_prob(actions).cpu())
        baseline_values.append(v.cpu())
    log_probs = torch.cat(old_logs)
    baselines = torch.cat(baseline_values)
    raw_adv = torch.tensor(rows.returns_bb, dtype=torch.float32) - baselines
    # Stabilize updates across the high variance of 200BB stacks.
    advantages = (raw_adv - raw_adv.mean()) / raw_adv.std(unbiased=False).clamp_min(1e-6)
    for name, value in [('log_probs', log_probs), ('baseline', baselines), ('advantages', advantages)]:
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f'nonfinite frozen {name}')
    return FrozenPPOData(rows, log_probs, advantages, baselines)


def clipped_surrogate(log_prob: torch.Tensor, old_log_prob: torch.Tensor,
                      advantage: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Per-row PPO clipped objective (MAXIMIZE)."""
    if log_prob.shape != old_log_prob.shape or log_prob.shape != advantage.shape:
        raise ValueError('PPO logprobs and advantages must have same shape')
    ratio = torch.exp(log_prob - old_log_prob)
    return torch.minimum(ratio * advantage,
                         torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)


def optimize_ppo(actor: TransformerActor, critic: CentralizedQCritic,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer,
                 frozen: FrozenPPOData, config: PPOConfig, *,
                 device: str = 'cpu', shuffle_seed: int = 0) -> dict[str, float | int]:
    """PPO epochs on a single on-policy rollout: both networks updated.

    Critic uses Huber regression Q_i(s, observed_a) -> terminal return_i.
    This does NOT claim counterfactual-Q supervision or VRPO Q-Boosting.
    """
    actor.train()
    critic.train()
    n = len(frozen.rows.actions)
    prng = random.Random(shuffle_seed)
    totals = {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0,
              'approx_kl': 0.0, 'clip_fraction': 0.0, 'grad_norm_actor': 0.0,
              'grad_norm_critic': 0.0}
    minibatches = 0
    for _ in range(config.epochs):
        order = list(range(n))
        prng.shuffle(order)
        for offset in range(0, n, config.minibatch_size):
            indices = order[offset:offset + config.minibatch_size]
            ab = batch_actors([frozen.rows.actors[i] for i in indices], device=device)
            cb = batch_critics([frozen.rows.critics[i] for i in indices], device=device)
            actions = torch.tensor([frozen.rows.actions[i] for i in indices], dtype=torch.long, device=device)
            seats = torch.tensor([frozen.rows.acting_seats[i] for i in indices], dtype=torch.long, device=device)
            old_log = frozen.old_log_probs[indices].to(device)
            advantages = frozen.advantages[indices].to(device)
            targets = torch.tensor([frozen.rows.returns_bb[i] for i in indices],
                                   dtype=torch.float32, device=device)

            dist = actor.distribution(ab)
            new_log = dist.log_prob(actions)
            ratio = (new_log - old_log).exp()
            actor_loss = -clipped_surrogate(new_log, old_log, advantages,
                                            config.clip_epsilon).mean()
            entropy = dist.entropy().mean()
            policy_objective = actor_loss - config.entropy_coef * entropy
            predicted = chosen_q(critic(cb), seats, actions)
            critic_loss = F.smooth_l1_loss(predicted, targets)

            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            policy_objective.backward()
            (config.value_coef * critic_loss).backward()
            actor_grad = nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip)
            critic_grad = nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip)
            if not all(math.isfinite(float(x)) for x in (policy_objective.item(),
                     critic_loss.item(), actor_grad.item(), critic_grad.item())):
                raise FloatingPointError('nonfinite PPO loss or gradient norm')
            actor_optimizer.step()
            critic_optimizer.step()
            with torch.no_grad():
                log_ratio = new_log - old_log
                # Schulman k3 KL estimate, nonnegative up to rounding.
                approximate_kl = (ratio - 1 - log_ratio).mean()
                clipped = ((ratio - 1).abs() > config.clip_epsilon).float().mean()
            values = {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(),
                      'entropy': entropy.item(), 'approx_kl': approximate_kl.item(),
                      'clip_fraction': clipped.item(), 'grad_norm_actor': actor_grad.item(),
                      'grad_norm_critic': critic_grad.item()}
            for key, value in values.items():
                totals[key] += value
            minibatches += 1
    return {**{key: round(value / minibatches, 6) for key, value in totals.items()},
            'minibatches': minibatches, 'decisions': n,
            'terminal_abs_mean_bb': round(sum(abs(x) for x in frozen.rows.returns_bb) / n, 5)}


@torch.no_grad()
def evaluate_vs_random(actor: TransformerActor, *, hands: int = 40,
                       device: str = 'cpu', seed: int = 900000,
                       big_blind: int = 100) -> dict[str, float | int]:
    """Seat-balanced benchmark; evaluation has no optimizer updates.

    Not a GTO/exploitability estimate. Confidence intervals are approximate
    iid normal intervals, with potentially very large poker reward variance.
    """
    if type(hands) is not int or hands < 4 or hands % 4:
        raise ValueError('evaluation hands must be a positive multiple of four')
    actor.eval()
    neural = NeuralActorPolicy(actor, device=device, big_blind=big_blind, greedy=True)
    payoffs = []
    for index in range(hands):
        neural_seat = index % 2
        # Cross all four (neural_seat, button) combinations in each group.
        button = (index // 2) % 2
        env = HUNL200Adapter(button=button, big_blind=big_blind)
        policies = {0: RandomLegalPolicy(), 1: RandomLegalPolicy()}
        policies[neural_seat] = neural
        result = collect_global_episode(env, policies, seed=seed + index,
                                        privileged=False)
        payoffs.append(result.terminal_rewards[neural_seat])
    mean = sum(payoffs) / hands
    variance = sum((x - mean) ** 2 for x in payoffs) / (hands - 1)
    se = math.sqrt(variance / hands)
    return {
        'hands': hands,
        'mean_bb100': round(mean * 100, 4),
        'ci95_lower_bb100': round((mean - 1.96 * se) * 100, 4),
        'ci95_upper_bb100': round((mean + 1.96 * se) * 100, 4),
        'seat0_hands': hands // 2,
        'seat1_hands': hands // 2,
        'neural_button_hands': hands // 2,
        'neural_big_blind_hands': hands // 2,
        'positive_hand_fraction': round(sum(x > 0 for x in payoffs) / hands, 4),
    }
