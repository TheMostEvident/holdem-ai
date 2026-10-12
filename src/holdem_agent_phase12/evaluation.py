"""Independent seed-blocked, four-layout HUNL200 evaluation (Phase 12).

Each seed block reuses the SAME shuffled deck for both agent seat assignments
and both button positions.  We average those four hands BEFORE computing
uncertainty; hands within the block are not independent data points.

Neither actor receives CriticState or the engine's hidden cards.  This is a
matched-deal benchmark, NOT exact duplicate poker under policy divergence and
NOT an estimate of GTO exploitability.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from holdem_agent_phase5 import RandomLegalPolicy
from holdem_agent_phase5.schema import InformationState
from holdem_agent_phase6 import HUNL200Adapter, collect_global_episode
from holdem_agent_phase8 import NeuralActorPolicy, TransformerActor


class EvaluationPolicy(Protocol):
    def act(self, observation: InformationState, rng) -> int: ...


@dataclass(frozen=True)
class EvaluationConfig:
    groups: int = 50  # 50 independent deck seeds; 4 hands per seed
    seed: int = 12_000_000
    mode: str = 'greedy'
    device: str = 'cpu'
    big_blind: int = 100
    max_decisions: int = 200

    def __post_init__(self) -> None:
        if type(self.groups) is not int or self.groups < 2:
            raise ValueError('groups must be an integer >=2 for a CI')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if self.mode not in ('greedy', 'sampled'):
            raise ValueError('mode must be greedy or sampled')
        if self.device != 'cpu' and not self.device.startswith('cuda'):
            raise ValueError('device must be cpu or cuda')
        if type(self.big_blind) is not int or self.big_blind < 1:
            raise ValueError('big_blind must be a positive integer')
        if type(self.max_decisions) is not int or self.max_decisions < 1:
            raise ValueError('max_decisions must be a positive integer')


@dataclass(frozen=True)
class DealGroup:
    seed: int
    # Ordered: button 0/actor 0; button 0/actor 1;
    #          button 1/actor 0; button 1/actor 1.
    payoffs_bb: tuple[float, float, float, float]
    actor_actions: tuple[int, ...]
    opponent_actions: tuple[int, ...]

    @property
    def average_bb(self) -> float:
        return math.fsum(self.payoffs_bb) / 4.0


@dataclass(frozen=True)
class EvaluationResult:
    config: EvaluationConfig
    blocks: tuple[DealGroup, ...]
    candidate_id: str
    opponent_id: str

    def __post_init__(self) -> None:
        if len(self.blocks) != self.config.groups:
            raise ValueError('wrong number of deal groups')
        if len({x.seed for x in self.blocks}) != len(self.blocks):
            raise ValueError('duplicate group seeds')

    @property
    def means_bb(self) -> tuple[float, ...]:
        return tuple(g.average_bb for g in self.blocks)

    def summary(self) -> dict:
        block_scores = self.means_bb
        estimate = summarize_block_scores(block_scores)
        candidate_actions = tuple(a for g in self.blocks for a in g.actor_actions)
        opponent_actions = tuple(a for g in self.blocks for a in g.opponent_actions)
        return {
            'protocol': 'phase12_seed_blocked_four_layout',
            'mode': self.config.mode,
            'candidate_id': self.candidate_id,
            'opponent_id': self.opponent_id,
            'seed_start': self.config.seed,
            'seed_end_inclusive': self.config.seed + self.config.groups - 1,
            'groups': self.config.groups,
            'hands': 4 * self.config.groups,
            'candidate_seat0_hands': 2 * self.config.groups,
            'candidate_seat1_hands': 2 * self.config.groups,
            'candidate_button_hands': 2 * self.config.groups,
            'candidate_big_blind_hands': 2 * self.config.groups,
            'actor_actions': action_counts(candidate_actions),
            'opponent_actions': action_counts(opponent_actions),
            **estimate,
        }


def action_counts(actions: tuple[int, ...]) -> list[int]:
    counts = [0] * 7
    for action in actions:
        if type(action) is not int or not 0 <= action < 7:
            raise ValueError('action index out of range')
        counts[action] += 1
    return counts


def summarize_block_scores(scores: tuple[float, ...]) -> dict[str, float | int]:
    """95% approximate normal CI, treating one DECK SEED as one iid unit."""
    n = len(scores)
    if n < 2 or not all(math.isfinite(x) for x in scores):
        raise ValueError('need >=2 finite block scores')
    mean = math.fsum(scores) / n
    var = math.fsum((x - mean)**2 for x in scores) / (n - 1)
    se = math.sqrt(var / n)
    ci = 1.96 * se
    return {
        'mean_bb100': round(mean * 100, 4),
        'ci95_lower_bb100': round(100 * (mean - ci), 4),
        'ci95_upper_bb100': round(100 * (mean + ci), 4),
        'standard_error_bb100': round(100 * se, 4),
        'std_per_group_bb': round(math.sqrt(var), 6),
        'positive_group_fraction': round(sum(x > 0 for x in scores) / n, 4),
    }


class _CountingPolicy:
    def __init__(self, inner: EvaluationPolicy):
        self.inner = inner
        self.actions: list[int] = []

    def act(self, observation: InformationState, rng) -> int:
        result = self.inner.act(observation, rng)
        observation.assert_legal(result)
        self.actions.append(result)
        return result


def model_fingerprint(actor: TransformerActor) -> str:
    """Stable model digest to distinguish a frozen checkpoint from the current one."""
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        digest.update(name.encode('utf-8'))
        array = value.detach().contiguous().cpu()
        digest.update(str(tuple(array.shape)).encode('ascii'))
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()[:16]


def evaluate_seed_blocks(candidate: TransformerActor, *,
                         opponent: TransformerActor | None = None,
                         config: EvaluationConfig | None = None,
                         on_group: Callable[[], None] | None = None) -> EvaluationResult:
    """Evaluate both seat orders for each common deck and each button position.

    For fixed (seed,button) the two games use the same deck, but swap which
    policy controls physical seat 0 and 1.  This *balances* card/seat effects;
    it does NOT guarantee identical subsequent chance outcomes if action
    paths diverge.  Opponent=None => uniform RandomLegalPolicy.
    """
    cfg = config or EvaluationConfig()
    if opponent is candidate:
        raise ValueError('pass a separate frozen Actor instance for self-check')
    if candidate.config.dropout != 0 or (opponent is not None and opponent.config.dropout != 0):
        raise ValueError('dropout must equal zero for repeatable evaluation')
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    modes = [(candidate, candidate.training)]
    if opponent is not None:
        modes.append((opponent, opponent.training))
    candidate_id = model_fingerprint(candidate)
    opponent_id = model_fingerprint(opponent) if opponent is not None else 'random_legal'
    agent = NeuralActorPolicy(candidate, device=cfg.device,
                              big_blind=cfg.big_blind, greedy=cfg.mode == 'greedy')
    other = (NeuralActorPolicy(opponent, device=cfg.device, big_blind=cfg.big_blind,
                               greedy=cfg.mode == 'greedy')
             if opponent is not None else RandomLegalPolicy())
    blocks: list[DealGroup] = []
    try:
        for k in range(cfg.groups):
            seed = cfg.seed + k
            rewards: list[float] = []
            candidate_actions: list[int] = []
            opponent_actions: list[int] = []
            for button in (0, 1):
                for candidate_seat in (0, 1):
                    agent_counter = _CountingPolicy(agent)
                    other_counter = _CountingPolicy(other)
                    policies = {candidate_seat: agent_counter,
                                1 - candidate_seat: other_counter}
                    env = HUNL200Adapter(button=button, big_blind=cfg.big_blind)
                    episode = collect_global_episode(
                        env, policies, seed=seed, max_decisions=cfg.max_decisions,
                        privileged=False,
                    )
                    if len(episode.terminal_rewards) != 2 or not math.isclose(
                        math.fsum(episode.terminal_rewards), 0.0, abs_tol=1e-7,
                    ):
                        raise AssertionError('terminal rewards are not zero-sum')
                    rewards.append(float(episode.terminal_rewards[candidate_seat]))
                    candidate_actions.extend(agent_counter.actions)
                    opponent_actions.extend(other_counter.actions)
            blocks.append(DealGroup(
                seed, tuple(rewards), tuple(candidate_actions), tuple(opponent_actions)
            ))
            if on_group is not None:
                on_group()
    finally:
        for model, train_mode in modes:
            model.train(train_mode)
    return EvaluationResult(cfg, tuple(blocks), candidate_id, opponent_id)


def compare_results(candidate: EvaluationResult, control: EvaluationResult) -> dict:
    """Paired difference of candidate and control against same fixed opponent.

    Valid when each uses identical seed-block schedule and opponent reference.
    Uses paired block-level differences, not subtraction of separate 95% CIs.
    """
    if candidate.config != control.config:
        raise ValueError('comparison requires exactly matched configurations')
    if candidate.opponent_id != control.opponent_id:
        raise ValueError('comparison requires exactly the same opponent')
    if tuple(g.seed for g in candidate.blocks) != tuple(g.seed for g in control.blocks):
        raise ValueError('comparison requires the same shuffled-deck seeds')
    deltas = tuple(a - b for a, b in zip(candidate.means_bb, control.means_bb))
    return {
        'protocol': 'phase12_paired_group_difference',
        'candidate_id': candidate.candidate_id,
        'control_id': control.candidate_id,
        'opponent_id': candidate.opponent_id,
        'groups': len(deltas), 'hands_per_agent': 4 * len(deltas),
        'delta_mean_bb100': summarize_block_scores(deltas)['mean_bb100'],
        'delta_ci95_lower_bb100': summarize_block_scores(deltas)['ci95_lower_bb100'],
        'delta_ci95_upper_bb100': summarize_block_scores(deltas)['ci95_upper_bb100'],
        'delta_se_bb100': summarize_block_scores(deltas)['standard_error_bb100'],
    }
