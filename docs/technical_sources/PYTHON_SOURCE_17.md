# Complete Python source volume 17/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## src/holdem_ai/training/global_rollout.py

Original blob `9e264e857fe9f5c8bbe45f7c8a46b6e76a9df4f6` (3127 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/global_rollout.py" sha="9e264e857fe9f5c8bbe45f7c8a46b6e76a9df4f6" -->

````````python
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

from holdem_ai.agents.policies import AgentPolicy
from holdem_ai.core.schema import InformationState, Terminal, Turn
from holdem_ai.environment.hunl200_adapter import CriticState, HUNL200Adapter


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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/global_rollout.py" -->

## src/holdem_ai/training/ppo_session.py

Original blob `2fc6468a18acd81ef604c12929f00992c43080d9` (5241 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/ppo_session.py" sha="2fc6468a18acd81ef604c12929f00992c43080d9" -->

````````python
"""Phase-9 on-policy PPO orchestration, safe checkpoints, and resume."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch

from holdem_ai.models.transformer import CentralizedQCritic, NetworkConfig, TransformerActor
from holdem_ai.training.ppo import PPOConfig, collect_self_play, evaluate_vs_random, freeze_old_policy_data, optimize_ppo


@dataclass
class TrainingSession:
    actor: TransformerActor
    critic: CentralizedQCritic
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    completed_updates: int
    network_config: NetworkConfig
    ppo_config: PPOConfig
    device: str


def create_session(*, device: str = 'cpu', seed: int = 42,
                   network_config: NetworkConfig | None = None,
                   ppo_config: PPOConfig | None = None) -> TrainingSession:
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or install compatible CUDA torch')
    network_config = network_config or NetworkConfig()
    ppo_config = ppo_config or PPOConfig(seed=seed)
    # Phase 8 inference assumes deterministic masked logits for old log-probs.
    if network_config.dropout != 0:
        raise ValueError('PPO baseline requires dropout=0 for consistent behavior logprobs')
    torch.manual_seed(seed)
    actor = TransformerActor(network_config).to(device)
    critic = CentralizedQCritic(network_config).to(device)
    aopt = torch.optim.Adam(actor.parameters(), lr=ppo_config.learning_rate)
    copt = torch.optim.Adam(critic.parameters(), lr=ppo_config.learning_rate)
    return TrainingSession(actor, critic, aopt, copt, 0,
                           network_config, ppo_config, device)


def save_checkpoint(session: TrainingSession, path: str | Path) -> None:
    """Atomic local checkpoint; .tmp partial files are never loaded as final."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    payload = {
        'phase': 9, 'algorithm': 'ppo_mc_privileged_baseline',
        'completed_updates': session.completed_updates,
        'network_config': asdict(session.network_config),
        'ppo_config': asdict(session.ppo_config),
        'actor_state': session.actor.state_dict(),
        'critic_state': session.critic.state_dict(),
        'actor_optimizer': session.actor_optimizer.state_dict(),
        'critic_optimizer': session.critic_optimizer.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str | Path, *, device: str = 'cpu') -> TrainingSession:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('phase') != 9 or payload.get('algorithm') != 'ppo_mc_privileged_baseline':
        raise ValueError('this is not a Phase 9 PPO checkpoint')
    cfg = NetworkConfig(**payload['network_config'])
    hyperparams = PPOConfig(**payload['ppo_config'])
    session = create_session(device=device, seed=hyperparams.seed,
                             network_config=cfg, ppo_config=hyperparams)
    session.actor.load_state_dict(payload['actor_state'])
    session.critic.load_state_dict(payload['critic_state'])
    session.actor_optimizer.load_state_dict(payload['actor_optimizer'])
    session.critic_optimizer.load_state_dict(payload['critic_optimizer'])
    session.completed_updates = int(payload['completed_updates'])
    torch.set_rng_state(payload['torch_rng_state'].cpu())
    return session


def train_updates(session: TrainingSession, *, updates: int,
                  checkpoint: str | Path | None = None) -> list[dict]:
    """updates are ADDITIONAL iterations when resuming a saved session."""
    if type(updates) is not int or updates < 1:
        raise ValueError('updates must be positive integer')
    reports = []
    for _ in range(updates):
        tick = time.perf_counter()
        step = session.completed_updates
        rows = collect_self_play(session.actor, session.ppo_config,
                                 iteration=step, device=session.device)
        frozen = freeze_old_policy_data(
            session.actor, session.critic, rows,
            device=session.device, chunk_size=session.ppo_config.inference_batch_size)
        stats = optimize_ppo(
            session.actor, session.critic, session.actor_optimizer,
            session.critic_optimizer, frozen, session.ppo_config,
            device=session.device, shuffle_seed=session.ppo_config.seed + 100003 * step)
        session.completed_updates += 1
        elapsed = time.perf_counter() - tick
        stats.update({'update': session.completed_updates, 'hands': rows.hands,
                      'elapsed_seconds': round(elapsed, 3),
                      'hands_per_second_training': round(rows.hands / max(elapsed, 1e-9), 3)})
        reports.append(stats)
        if checkpoint is not None:
            save_checkpoint(session, checkpoint)
    return reports


def format_report(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/ppo_session.py" -->

## src/holdem_ai/training/ppo.py

Original blob `bf6cc749ca20183b507adbc1df8687dc2b41f142` (15577 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/ppo.py" sha="bf6cc749ca20183b507adbc1df8687dc2b41f142" -->

````````python
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

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode
from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.encoding.observation import EncodedActor, EncodedCritic
from holdem_ai.models.transformer import CentralizedQCritic, TransformerActor, chosen_q
from holdem_ai.agents.neural_policy import NeuralActorPolicy


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
    terminal_actions: tuple[int, ...] = ()  # one final action per hand, if collected
    decisions_per_hand: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.actions)
        if n == 0 or not all(len(x) == n for x in (
                self.actors, self.critics, self.acting_seats, self.returns_bb)):
            raise ValueError('trajectory rows must be nonempty and aligned')
        if self.terminal_actions and len(self.terminal_actions) != self.hands:
            raise ValueError('terminal actions must align with hands')
        if self.decisions_per_hand and (len(self.decisions_per_hand) != self.hands
                                        or sum(self.decisions_per_hand) != n):
            raise ValueError('hand decision counts must align')
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/ppo.py" -->

## src/holdem_ai/training/qboost_session.py

Original blob `0cca5ab85493a816374d491c0bd41aa26d3702b4` (5741 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/qboost_session.py" sha="0cca5ab85493a816374d491c0bd41aa26d3702b4" -->

````````python
"""Phase-10 trainer, safe checkpoints and Phase-9 warm start.

A frozen Q estimate and old policy are used throughout each update's PPO epochs.
This prevents leakage from simultaneously changing critic targets into actor control
variates. Comparisons must use separate on-policy rollouts after actors diverge.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig, collect_self_play
from holdem_ai.training.ppo_session import create_session, load_checkpoint
from holdem_ai.training.ppo_session import TrainingSession
from holdem_ai.training.qboost import QBoostConfig, freeze_qboost_data, optimize_qboost


@dataclass
class QBoostSession:
    base: TrainingSession
    qconfig: QBoostConfig

    @property
    def completed_updates(self) -> int:
        return self.base.completed_updates


def create_qboost_session(*, device: str = 'cpu', seed: int = 42,
                          network_config: NetworkConfig | None = None,
                          ppo_config: PPOConfig | None = None,
                          qconfig: QBoostConfig | None = None) -> QBoostSession:
    return QBoostSession(create_session(device=device, seed=seed,
                         network_config=network_config, ppo_config=ppo_config),
                         qconfig or QBoostConfig())


def from_phase9_checkpoint(path: str | Path, *, device: str = 'cpu',
                           qconfig: QBoostConfig | None = None) -> QBoostSession:
    """Explicitly warm-start from user's trusted Phase9 checkpoint.

    Phase9's Adam and update counter are preserved; switching algorithms
    means the training sequence is not an exact uninterrupted PPO replay.
    """
    return QBoostSession(load_checkpoint(path, device=device), qconfig or QBoostConfig())


def save_qboost_checkpoint(session: QBoostSession, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    base = session.base
    payload = {
        'phase': 10,
        'algorithm': 'qboost_ppo_doubly_corrected_lite',
        'completed_updates': base.completed_updates,
        'network_config': asdict(base.network_config),
        'ppo_config': asdict(base.ppo_config),
        'qboost_config': asdict(session.qconfig),
        'actor_state': base.actor.state_dict(),
        'critic_state': base.critic.state_dict(),
        'actor_optimizer': base.actor_optimizer.state_dict(),
        'critic_optimizer': base.critic_optimizer.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_qboost_checkpoint(path: str | Path, *, device: str = 'cpu') -> QBoostSession:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('phase') != 10 or payload.get('algorithm') != 'qboost_ppo_doubly_corrected_lite':
        raise ValueError('this is not a Phase10 QBoost checkpoint')
    network_cfg = NetworkConfig(**payload['network_config'])
    ppo_cfg = PPOConfig(**payload['ppo_config'])
    qcfg = QBoostConfig(**payload['qboost_config'])
    session = create_qboost_session(device=device, seed=ppo_cfg.seed,
                                     network_config=network_cfg, ppo_config=ppo_cfg,
                                     qconfig=qcfg)
    base = session.base
    base.actor.load_state_dict(payload['actor_state'])
    base.critic.load_state_dict(payload['critic_state'])
    base.actor_optimizer.load_state_dict(payload['actor_optimizer'])
    base.critic_optimizer.load_state_dict(payload['critic_optimizer'])
    base.completed_updates = int(payload['completed_updates'])
    if base.completed_updates < 0:
        raise ValueError('negative checkpoint update counter')
    torch.set_rng_state(payload['torch_rng_state'].cpu())
    return session


def train_qboost_updates(session: QBoostSession, *, updates: int,
                         checkpoint: str | Path | None = None) -> list[dict]:
    if type(updates) is not int or updates < 1:
        raise ValueError('updates must be positive integer')
    base = session.base
    reports = []
    for _ in range(updates):
        begin = time.perf_counter()
        iteration = base.completed_updates
        rows = collect_self_play(base.actor, base.ppo_config,
                                 iteration=iteration, device=base.device)
        frozen = freeze_qboost_data(base.actor, base.critic, rows,
                                    device=base.device,
                                    chunk_size=base.ppo_config.inference_batch_size)
        stats = optimize_qboost(base.actor, base.critic,
                                base.actor_optimizer, base.critic_optimizer,
                                frozen, base.ppo_config, session.qconfig,
                                device=base.device,
                                shuffle_seed=base.ppo_config.seed + 100003 * iteration)
        base.completed_updates += 1
        elapsed = time.perf_counter() - begin
        stats.update({'update': base.completed_updates, 'hands': rows.hands,
                      'beta': session.qconfig.beta,
                      'elapsed_seconds': round(elapsed, 3),
                      'hands_per_second_training': round(rows.hands/max(elapsed, 1e-9), 3)})
        if not all(math.isfinite(float(x)) for x in stats.values()):
            raise FloatingPointError('nonfinite training metric')
        reports.append(stats)
        if checkpoint is not None:
            save_qboost_checkpoint(session, checkpoint)
    return reports

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/qboost_session.py" -->

## src/holdem_ai/training/qboost.py

Original blob `be748dd8a609718276646aa2fc2595a3f56ab39a` (11179 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/qboost.py" sha="be748dd8a609718276646aa2fc2595a3f56ab39a" -->

````````python
"""Phase 10: doubly-corrected Q-control-variate PPO research baseline.

At theta_old, E[score*(G - V - beta*(Q(a)-V))]
+ beta * sum_a grad(pi(a))*Q(a) equals the Monte Carlo policy gradient.
PPO clipping subsequently introduces ordinary on-policy surrogate bias.
This is a VRPO-inspired *Lite* experiment, NOT a faithful paper reimplementation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.models.transformer import CentralizedQCritic, TransformerActor, chosen_q
from holdem_ai.training.ppo import FrozenPPOData, PPOConfig, TrajectoryRows, clipped_surrogate


@dataclass(frozen=True)
class QBoostConfig:
    beta: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.beta, bool) or not isinstance(self.beta, (float, int)) or not math.isfinite(self.beta) or not 0.0 <= self.beta <= 1.0:
            raise ValueError('beta must be a finite number in [0,1]')


@dataclass(frozen=True)
class FrozenQBoostData:
    rows: TrajectoryRows
    old_log_probs: torch.Tensor    # CPU [N]
    old_probs: torch.Tensor        # CPU [N,7]
    q_actions: torch.Tensor        # CPU [N,7], acting seat, no gradients
    baselines: torch.Tensor        # CPU [N]
    raw_advantages: torch.Tensor   # CPU [N] = MC - V
    center: float                 # same MC-advantage normalizer for both terms
    scale: float


def freeze_qboost_data(actor: TransformerActor, critic: CentralizedQCritic,
                       rows: TrajectoryRows, *, device: str = 'cpu',
                       chunk_size: int = 64) -> FrozenQBoostData:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size must be positive integer')
    actor.eval()
    critic.eval()
    probs, qs, logs = [], [], []
    with torch.no_grad():
        for start in range(0, len(rows.actions), chunk_size):
            stop = min(start + chunk_size, len(rows.actions))
            ab = batch_actors(rows.actors[start:stop], device=device)
            cb = batch_critics(rows.critics[start:stop], device=device)
            d = actor.distribution(ab)
            seats = torch.tensor(rows.acting_seats[start:stop], device=device, dtype=torch.long)
            acts = torch.tensor(rows.actions[start:stop], device=device, dtype=torch.long)
            q = critic(cb)
            q_seat = q[torch.arange(stop - start, device=device), seats]
            if not bool(torch.isfinite(q_seat).all().item()):
                raise FloatingPointError('nonfinite old action values')
            probs.append(d.probs.detach().cpu())
            logs.append(d.log_prob(acts).detach().cpu())
            qs.append(q_seat.detach().cpu())
    old_p, old_log, q_all = torch.cat(probs), torch.cat(logs), torch.cat(qs)
    baselines = (old_p * q_all).sum(-1)
    raw = torch.tensor(rows.returns_bb, dtype=torch.float32) - baselines
    center = float(raw.mean().item())
    scale = max(float(raw.std(unbiased=False).item()), 1e-6)
    if not all(bool(torch.isfinite(z).all().item()) for z in (old_p, old_log, q_all, baselines, raw)):
        raise FloatingPointError('nonfinite frozen QBoost information')
    return FrozenQBoostData(rows, old_log, old_p, q_all, baselines, raw, center, scale)


def qboost_surrogate(new_log: torch.Tensor, new_probs: torch.Tensor,
                     old_log: torch.Tensor, old_probs: torch.Tensor,
                     q_values: torch.Tensor, actions: torch.Tensor,
                     returns: torch.Tensor, baseline: torch.Tensor,
                     *, beta: float, center: float, scale: float,
                     clip_epsilon: float) -> torch.Tensor:
    """Per-row actor objective to MAXIMIZE, sharing one centering/scaling.

    MC normalized A = (G-V-center)/scale.
    Residual normalized A = (G-V-beta*(Q(a)-V)-center)/scale.
    Analytic = beta * sum_a (pi_new(a)-pi_old(a))*Q(a)/scale.
    beta=0 agrees exactly with Phase9 clipped PPO on identically frozen data.
    """
    n = new_log.numel()
    if not isinstance(beta, (int, float)) or isinstance(beta, bool) or not 0 <= beta <= 1:
        raise ValueError('beta must be in [0,1]')
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(center):
        raise ValueError('invalid advantage normalization')
    if not 0 < clip_epsilon < 1:
        raise ValueError('clip epsilon must be in (0,1)')
    if not (new_log.shape == old_log.shape == actions.shape == returns.shape == baseline.shape == (n,)):
        raise ValueError('scalar row tensors must align')
    if (new_probs.shape != (n, 7) or old_probs.shape != (n, 7) or q_values.shape != (n, 7)):
        raise ValueError('policy and Q must be [N,7]')
    if actions.dtype != torch.long or not bool(((actions >= 0) & (actions < 7)).all().item()):
        raise ValueError('invalid actions')
    if not all(bool(torch.isfinite(t).all().item()) for t in
               (new_log, old_log, new_probs, old_probs, q_values, returns, baseline)):
        raise FloatingPointError('nonfinite surrogate inputs')
    observed_q = q_values.gather(1, actions[:, None]).squeeze(1)
    residual = (returns - baseline - beta * (observed_q - baseline) - center) / scale
    mc_clip = clipped_surrogate(new_log, old_log, residual, clip_epsilon)
    analytic = beta * ((new_probs - old_probs) * q_values).sum(-1) / scale
    return mc_clip + analytic


def qboost_diagnostics(frozen: FrozenQBoostData, beta: float = 1.0) -> dict[str, float]:
    """Variance of empirical logit-gradient proxies, NOT parameter-gradient variance.

    They compare within the same observed sample, using frozen old policy and Q.
    Improvement (ratio <1) is not guaranteed; Q error can make it worse.
    """
    a = torch.tensor(frozen.rows.actions, dtype=torch.long)
    ret = torch.tensor(frozen.rows.returns_bb, dtype=torch.float32)
    qa = frozen.q_actions.gather(1, a[:, None]).squeeze(1)
    score = F.one_hot(a, num_classes=7).float() - frozen.old_probs
    old = frozen.raw_advantages - frozen.center
    residual = old - beta * (qa - frozen.baselines)
    p = frozen.old_probs
    correction = beta * p * (frozen.q_actions - frozen.baselines[:, None])
    g_old = score * (old / frozen.scale)[:, None]
    g_new = score * (residual / frozen.scale)[:, None] + correction / frozen.scale
    def variance(x: torch.Tensor) -> float:
        return float(x.var(unbiased=False).item())
    def logit_variance(x: torch.Tensor) -> float:
        return float(x.var(dim=0, unbiased=False).sum().item())
    old_v, new_v = logit_variance(g_old), logit_variance(g_new)
    return {
        'mc_advantage_variance': round(variance(frozen.raw_advantages), 6),
        'boosted_residual_variance': round(variance(residual), 6),
        'logit_gradient_proxy_variance_ppo': round(old_v, 6),
        'logit_gradient_proxy_variance_qboost': round(new_v, 6),
        'logit_variance_ratio': round(new_v / max(old_v, 1e-12), 6),
    }


def optimize_qboost(actor: TransformerActor, critic: CentralizedQCritic,
                    actor_optimizer: torch.optim.Optimizer,
                    critic_optimizer: torch.optim.Optimizer,
                    frozen: FrozenQBoostData, config: PPOConfig,
                    qconfig: QBoostConfig, *, device: str = 'cpu',
                    shuffle_seed: int = 0) -> dict[str, float | int]:
    actor.train()
    critic.train()
    n = len(frozen.rows.actions)
    rng = random.Random(shuffle_seed)
    keys = ('actor_loss', 'critic_loss', 'entropy', 'approx_kl', 'clip_fraction',
            'grad_norm_actor', 'grad_norm_critic', 'analytical_boost_mean')
    totals = dict.fromkeys(keys, 0.0)
    count = 0
    for _ in range(config.epochs):
        order = list(range(n))
        rng.shuffle(order)
        for offset in range(0, n, config.minibatch_size):
            ind = order[offset:offset + config.minibatch_size]
            ab = batch_actors([frozen.rows.actors[i] for i in ind], device=device)
            cb = batch_critics([frozen.rows.critics[i] for i in ind], device=device)
            acts = torch.tensor([frozen.rows.actions[i] for i in ind], dtype=torch.long, device=device)
            seats = torch.tensor([frozen.rows.acting_seats[i] for i in ind], dtype=torch.long, device=device)
            rewards = torch.tensor([frozen.rows.returns_bb[i] for i in ind], dtype=torch.float32, device=device)
            oldlog = frozen.old_log_probs[ind].to(device)
            oldp = frozen.old_probs[ind].to(device)
            q_old = frozen.q_actions[ind].to(device)
            v_old = frozen.baselines[ind].to(device)
            distribution = actor.distribution(ab)
            log_new = distribution.log_prob(acts)
            surrogate = qboost_surrogate(
                log_new, distribution.probs, oldlog, oldp, q_old, acts,
                rewards, v_old, beta=qconfig.beta, center=frozen.center,
                scale=frozen.scale, clip_epsilon=config.clip_epsilon)
            a_loss = -surrogate.mean()
            entropy = distribution.entropy().mean()
            policy_loss = a_loss - config.entropy_coef * entropy
            q_prediction = chosen_q(critic(cb), seats, acts)
            c_loss = F.smooth_l1_loss(q_prediction, rewards)
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            (config.value_coef * c_loss).backward()
            an = nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip)
            cn = nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip)
            if not all(math.isfinite(float(x)) for x in (policy_loss.item(), c_loss.item(), an.item(), cn.item())):
                raise FloatingPointError('nonfinite Phase10 loss/grad')
            actor_optimizer.step()
            critic_optimizer.step()
            with torch.no_grad():
                ratio = (log_new - oldlog).exp()
                log_ratio = log_new - oldlog
                residual = (rewards - v_old - qconfig.beta *
                            (q_old.gather(1, acts[:, None]).squeeze(1) - v_old) - frozen.center) / frozen.scale
                clip_frac = ((ratio - 1).abs() > config.clip_epsilon).float().mean()
                boost_mean = qconfig.beta * ((distribution.probs - oldp) * q_old).sum(-1).mean() / frozen.scale
                stats = {'actor_loss': a_loss.item(), 'critic_loss': c_loss.item(),
                         'entropy': entropy.item(), 'approx_kl': (ratio - 1 - log_ratio).mean().item(),
                         'clip_fraction': clip_frac.item(), 'grad_norm_actor': an.item(),
                         'grad_norm_critic': cn.item(), 'analytical_boost_mean': boost_mean.item()}
            for k in keys:
                totals[k] += stats[k]
            count += 1
    return {**{k: round(v / count, 6) for k, v in totals.items()},
            **qboost_diagnostics(frozen, beta=qconfig.beta),
            'minibatches': count, 'decisions': n,
            'terminal_abs_mean_bb': round(sum(abs(x) for x in frozen.rows.returns_bb)/n, 5)}

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/qboost.py" -->

## src/holdem_ai/training/reward_models.py

Original blob `7216d7bbce7dc495cc7acc9636350aa5e5cd8917` (1937 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/reward_models.py" sha="7216d7bbce7dc495cc7acc9636350aa5e5cd8917" -->

````````python
"""Terminal reward definitions; preserve distinction between chip and tournament utility."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _vector(values: Sequence[int | float], label: str) -> tuple[float, ...]:
    result = tuple(float(x) for x in values)
    if len(result) < 2 or any(not math.isfinite(x) for x in result):
        raise ValueError(f"{label} requires >=2 finite values")
    return result


@dataclass(frozen=True, slots=True)
class ChipDeltaReward:
    """r_i = (stack_end[i] - stack_start[i]) / scale; scale often BB."""

    scale: float = 1.0
    enforce_conservation: bool = True
    atol: float = 1e-8

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and > 0")

    def calculate(self, starting_stacks: Sequence[float], ending_stacks: Sequence[float]) -> tuple[float, ...]:
        start = _vector(starting_stacks, "starting_stacks")
        end = _vector(ending_stacks, "ending_stacks")
        if len(start) != len(end):
            raise ValueError("starting/ending stack count mismatch")
        if self.enforce_conservation and not math.isclose(
            math.fsum(start), math.fsum(end), rel_tol=0, abs_tol=self.atol
        ):
            raise ValueError("chip conservation violated (or rake/fees not configured)")
        return tuple((e - s) / self.scale for s, e in zip(start, end))


@dataclass(frozen=True, slots=True)
class WinnerTakeAllReward:
    """Centered winner-take-all reward: r_i = 1[i wins] - 1 / N."""

    def calculate(self, winner_id: int, num_players: int) -> tuple[float, ...]:
        if num_players < 2 or not 0 <= winner_id < num_players:
            raise ValueError("invalid tournament winner or player count")
        return tuple(float(i == winner_id) - 1.0 / num_players for i in range(num_players))

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/reward_models.py" -->

## src/holdem_ai/training/self_play.py

Original blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (0 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/self_play.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/self_play.py" -->

## src/holdem_ai/training/trainer.py

Original blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (0 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/trainer.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/trainer.py" -->

## src/holdem_ai/utils/__init__.py

Original blob `c45dbcdd525a3fadb2abb368e5264f09bb51e773` (39 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/utils/__init__.py" sha="c45dbcdd525a3fadb2abb368e5264f09bb51e773" -->

````````python
"""Consolidated functional package."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/utils/__init__.py" -->

## src/holdem_ai/utils/progress.py

Original blob `00dbb59f0a6153bc3acce146fe30eed7c5176775` (9411 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/utils/progress.py" sha="00dbb59f0a6153bc3acce146fe30eed7c5176775" -->

````````python
"""Live, optional-tqdm terminal progress for Phase11 training.

Observes JSON event callbacks, never touches the Actor, optimizers, poker RNG,
checkpoints, or the metrics.jsonl event stream. A heartbeat reports remaining
wall-clock budget even while an update/evaluation is taking a long time.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from typing import Callable, TextIO


def format_duration(seconds: float | None) -> str:
    """Display a human-readable conservative duration, or -- if unknown."""
    if seconds is None or not math.isfinite(seconds):
        return '--:--:--'
    sec = max(0, math.ceil(seconds))
    hours, rem = divmod(sec, 3600)
    minutes, sec = divmod(rem, 60)
    return f'{hours:02d}:{minutes:02d}:{sec:02d}'


class TrainingProgress:
    """Live display fed by the *existing* run_experiment emit callback.

    Progress is measured by additional updates if max_updates is supplied,
    otherwise by the wall-clock budget. `time_left` always means remaining
    invocation budget, NOT guaranteed time until the job ends (evaluations and
    stopping at boundaries can cause overruns). Update ETA is an estimate.
    """

    def __init__(self, *, duration_seconds: float, max_updates: int | None,
                 interval: float = 1.0, stream: TextIO | None = None,
                 clock: Callable[[], float] | None = None,
                 threaded: bool = True, prefer_tqdm: bool = True) -> None:
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError('duration_seconds must be finite and positive')
        if max_updates is not None and (type(max_updates) is not int or max_updates <= 0):
            raise ValueError('max_updates must be None or positive integer')
        if not math.isfinite(interval) or interval < 0.1:
            raise ValueError('interval must be >= 0.1 second')
        self.duration_seconds = float(duration_seconds)
        self.max_updates = max_updates
        self.interval = float(interval)
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock or time.monotonic
        self.threaded = threaded
        self.prefer_tqdm = prefer_tqdm
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._start: float | None = None
        self._initial_updates = 0
        self._updates = 0
        self._total_updates = 0
        self._hands_per_second: float | None = None
        self._last_update_seconds: float | None = None
        self._status = 'starting'
        self._done = False
        self._bar = None
        self._last_plain_chars = 0

    def __enter__(self) -> 'TrainingProgress':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def on_event(self, event: dict) -> None:
        """Consume a runner event, without changing its persisted JSON."""
        kind = event.get('kind')
        with self._lock:
            if kind in ('start', 'resume'):
                if self._start is not None:
                    raise RuntimeError('progress received a second start event')
                self._start = self.clock()
                self._initial_updates = int(event['completed_updates'])
                self._total_updates = self._initial_updates
                self._status = 'running'
                self._open_display()
                if self.threaded:
                    self._worker = threading.Thread(target=self._heartbeat, daemon=True,
                                                    name='phase11-progress')
                    self._worker.start()
            elif kind == 'update' and self._start is not None:
                self._total_updates = int(event['update'])
                self._updates = max(0, self._total_updates - self._initial_updates)
                rate = event.get('hands_per_second_training')
                if isinstance(rate, (float, int)) and math.isfinite(float(rate)):
                    self._hands_per_second = float(rate)
                update_seconds = event.get('elapsed_seconds')
                if isinstance(update_seconds, (float, int)) and not isinstance(update_seconds, bool) and math.isfinite(float(update_seconds)) and update_seconds > 0:
                    self._last_update_seconds = float(update_seconds)
                self._status = 'training'
            elif kind == 'evaluation':
                self._status = 'evaluated'
            elif kind == 'interrupt':
                self._status = 'interrupted'
            elif kind == 'end':
                self._status = str(event.get('stop_reason', 'finished'))
                self._done = True
            self._refresh()

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = max(0.0, (self.clock() - self._start)) if self._start is not None else 0.0
            time_left = max(0.0, self.duration_seconds - elapsed)
            remaining_updates = (max(0, self.max_updates - self._updates)
                                 if self.max_updates is not None else None)
            eta_updates = (elapsed / self._updates * remaining_updates
                           if remaining_updates is not None and self._updates > 0 else None)
            estimated_remaining = (min(time_left, eta_updates)
                                   if eta_updates is not None else time_left)
            fraction = (min(1.0, self._updates / self.max_updates)
                        if self.max_updates is not None else min(1.0, elapsed / self.duration_seconds))
            return dict(elapsed=elapsed, time_left=time_left,
                        update_eta=eta_updates, estimated_remaining=estimated_remaining,
                        remaining_updates=remaining_updates, fraction=fraction,
                        updates=self._updates, total_updates=self._total_updates,
                        hands_per_second=self._hands_per_second,
                        last_update_seconds=self._last_update_seconds, status=self._status)

    def _open_display(self) -> None:
        if not self.prefer_tqdm:
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return
        target = (self.max_updates if self.max_updates is not None
                  else self.duration_seconds)
        unit = 'upd' if self.max_updates is not None else 's'
        self._bar = tqdm(total=target, desc='Phase 11', unit=unit,
                         file=self.stream, dynamic_ncols=True,
                         mininterval=0, leave=True, disable=False)

    def _refresh(self) -> None:
        if self._start is None:
            return
        s = self.snapshot()
        left = format_duration(s['time_left'])
        eta = format_duration(s['estimated_remaining'])
        counter = (f"{s['total_updates']} (+{s['updates']}/{self.max_updates})"
                   if self.max_updates is not None else str(s['total_updates']))
        speed = ('--' if s['hands_per_second'] is None
                 else f"{s['hands_per_second']:.1f}")
        elapsed = format_duration(s['elapsed'])
        postfix = (f"upd={counter} hands/s={speed} "
                   f"elapsed={elapsed} time_left={left} ETA={eta} stage={s['status']}")
        if self._bar is not None:
            self._bar.n = (s['updates'] if self.max_updates is not None
                           else min(self.duration_seconds, s['elapsed']))
            self._bar.set_postfix_str(postfix, refresh=False)
            self._bar.refresh()
        else:
            width = 22
            filled = round(s['fraction'] * width)
            line = '[' + '#' * filled + '-' * (width - filled) + '] '
            line += f"{100*s['fraction']:5.1f}% {postfix}"
            self.stream.write('\r' + line + ' ' * max(0, self._last_plain_chars - len(line)))
            self.stream.flush()
            self._last_plain_chars = len(line)
            if self._done:
                self.stream.write('\n')
                self.stream.flush()
                self._last_plain_chars = 0

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._done:
                    return
                self._refresh()

    def message(self, message: str) -> None:
        """Print a non-progress message without leaving a broken bar."""
        with self._lock:
            if self._bar is not None:
                self._bar.clear()
            elif self._last_plain_chars:
                self.stream.write('\r' + ' ' * self._last_plain_chars + '\r')
                self.stream.flush()
            print(message, flush=True)
            if self._start is not None and not self._done:
                self._refresh()

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=max(0.5, self.interval * 2))
            self._worker = None
        with self._lock:
            if self._start is not None:
                self._refresh()
            if self._bar is not None:
                self._bar.close()
                self._bar = None
            elif self._last_plain_chars:
                self.stream.write('\n')
                self.stream.flush()
                self._last_plain_chars = 0

````````

<!-- END_PY_SOURCE path="src/holdem_ai/utils/progress.py" -->

## src/holdem_ai/utils/snapshot.py

Original blob `9e359decefae168acdc4f76606dbc141f674c20b` (776 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/utils/snapshot.py" sha="9e359decefae168acdc4f76606dbc141f674c20b" -->

````````python
"""Safe, explicit experimental snapshot facility (not a CFR correctness guarantee)."""
from __future__ import annotations

from copy import deepcopy
from typing import TypeVar

T = TypeVar("T")


def snapshot(environment: T) -> T:
    """Deepcopy engine AND RNG state, if its implementation supports deepcopy.

    Do not assume a poker engine is branchable: test reproducibility and
    mutation isolation for each engine's card RNG, seat RNG and state history.
    """
    try:
        replica = deepcopy(environment)
    except Exception as exc:
        raise RuntimeError("engine snapshot failed; provide engine-specific clone API") from exc
    if replica is environment:
        raise RuntimeError("snapshot returned the original environment object")
    return replica

````````

<!-- END_PY_SOURCE path="src/holdem_ai/utils/snapshot.py" -->

## tests/__init__.py

Original blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (0 bytes).

<!-- BEGIN_PY_SOURCE path="tests/__init__.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="tests/__init__.py" -->
