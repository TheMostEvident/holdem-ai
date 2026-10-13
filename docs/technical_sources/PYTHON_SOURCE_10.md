# Python full source volume 10/28

[Canonical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) — original commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`; exact code with individual blob SHA.

## src/holdem_ai/evaluation/seed_blocked.py

Original SHA `4a30e1e22ee87830e1059176db3a20c31e2a55f4` (10961 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/seed_blocked.py" sha="4a30e1e22ee87830e1059176db3a20c31e2a55f4" -->

````````python
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

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.core.schema import InformationState
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode
from holdem_ai.agents.neural_policy import NeuralActorPolicy
from holdem_ai.models.transformer import TransformerActor


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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/seed_blocked.py" -->

## src/holdem_ai/evaluation/seed_statistics.py

Original SHA `8c650e3e3b0c366a7d91013696f2cf125f98e428` (1889 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/seed_statistics.py" sha="8c650e3e3b0c366a7d91013696f2cf125f98e428" -->

````````python
"""Training-seed-level statistical estimates for fixed independent test-deck banks.

Do not falsely treat correlated deck blocks from different policies/runs as
independent training replicates.  The t interval quantifies between-seed
variation conditional on the specified test deck schedule.
"""
from __future__ import annotations

import math
from statistics import mean, stdev

# Student's t two-tailed 0.975 critical values for df 1..30.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def seed_level_summary(values: list[float] | tuple[float, ...]) -> dict:
    """Mean +/- approximate 95% t CI over *training seeds*, not deal groups.

    With one training seed the interval is unidentified and returned as null.
    Values are already in bb/100 units.
    """
    if not values or not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                             and math.isfinite(v) for v in values):
        raise ValueError('need >=1 finite seed-level bb/100 estimates')
    n = len(values)
    mu = mean(values)
    if n == 1:
        return {'seeds': 1, 'mean_delta_bb100': round(mu, 4), 'se_bb100': None,
                'ci95_lower_bb100': None, 'ci95_upper_bb100': None,
                'positive_seed_fraction': round(float(values[0] > 0), 4)}
    se = stdev(values) / math.sqrt(n)
    # For n>31, use an asymptotic 1.96; slightly optimistic at moderate n.
    tcrit = _T975[n-2] if n <= 31 else 1.96
    return {'seeds': n, 'mean_delta_bb100': round(mu, 4), 'se_bb100': round(se, 4),
            'ci95_lower_bb100': round(mu-tcrit*se, 4),
            'ci95_upper_bb100': round(mu+tcrit*se, 4),
            'positive_seed_fraction': round(sum(v > 0 for v in values)/n, 4)}

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/seed_statistics.py" -->

## src/holdem_ai/evaluation/strategy_diagnostics.py

Original SHA `d098e73ff576b48388927920fdbb1829df0b5203` (5641 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/strategy_diagnostics.py" sha="d098e73ff576b48388927920fdbb1829df0b5203" -->

````````python
"""Actor-observation-only behavioral diagnostics for evaluated HUNL200 hands."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from holdem_ai.core.schema import InformationState
from holdem_ai.encoding.tensors import batch_actors
from holdem_ai.encoding.observation import encode_actor
from holdem_ai.evaluation.action_diagnostics import ACTION_LABELS, STREETS

SPR_BUCKETS = ('spr_lt_1', 'spr_1_to_4', 'spr_4_to_10', 'spr_ge_10')
POSITIONS = ('button', 'big_blind')


def spr_bucket(observation: InformationState) -> str:
    """Effective stack-to-pot ratio using ONLY actor-visible public features.

    At an action decision, effective stack = min(hero stack, opponent stack).
    Feature indices are supplied by the Phase6 HUNL200 adapter.
    """
    if len(observation.features) < 8:
        raise ValueError('HUNL200 observation requires at least 8 features')
    pot = float(observation.features[3])
    hero = float(observation.features[6])
    villain = float(observation.features[7])
    if not all(math.isfinite(x) and x >= 0 for x in (pot, hero, villain)):
        raise ValueError('invalid observable pot or stacks')
    ratio = min(hero, villain) / max(pot, 1e-9)
    if ratio < 1:
        return SPR_BUCKETS[0]
    if ratio < 4:
        return SPR_BUCKETS[1]
    if ratio < 10:
        return SPR_BUCKETS[2]
    return SPR_BUCKETS[3]


def _empty_bucket() -> dict:
    return {'decisions': 0, 'actions': [0] * 7, 'legal_opportunities': [0] * 7}


def _summary_bucket(bucket: dict) -> dict:
    count, legal = bucket['actions'], bucket['legal_opportunities']
    return {'decisions': bucket['decisions'],
            'actions': dict(zip(ACTION_LABELS, count)),
            'legal_opportunities': dict(zip(ACTION_LABELS, legal)),
            'conditional_action_rate': {
                label: round(count[i] / legal[i], 6) if legal[i] else None
                for i, label in enumerate(ACTION_LABELS)
            }}


@dataclass
class ActionProfile:
    """Only public/own-player observations. No CriticState or engine access."""
    samples: list[InformationState] = field(default_factory=list, repr=False)
    actions: list[int] = field(default_factory=list, repr=False)

    def record(self, obs: InformationState, action: int) -> None:
        obs.assert_legal(action)
        if len(obs.legal_action_mask) != len(ACTION_LABELS):
            raise ValueError('wrong HUNL200 action mask size')
        if obs.stage not in STREETS or obs.features[1] not in (0.0, 1.0):
            raise ValueError('unrecognized HUNL200 street or button feature')
        self.samples.append(obs)
        self.actions.append(action)

    def summary(self, actor=None, *, device: str = 'cpu', big_blind: int = 100,
                entropy_batch_size: int = 64) -> dict:
        if type(entropy_batch_size) is not int or entropy_batch_size < 1:
            raise ValueError('entropy_batch_size must be positive int')
        total = _empty_bucket()
        by_street = {name: _empty_bucket() for name in STREETS}
        by_position = {name: _empty_bucket() for name in POSITIONS}
        by_spr = {name: _empty_bucket() for name in SPR_BUCKETS}
        for obs, action in zip(self.samples, self.actions):
            pos = 'button' if obs.features[1] == 1.0 else 'big_blind'
            bucket_name = spr_bucket(obs)
            for bucket in (total, by_street[obs.stage], by_position[pos], by_spr[bucket_name]):
                bucket['decisions'] += 1
                bucket['actions'][action] += 1
                for i, valid in enumerate(obs.legal_action_mask):
                    if valid:
                        bucket['legal_opportunities'][i] += 1
        out = {**_summary_bucket(total),
               'by_street': {k: _summary_bucket(v) for k, v in by_street.items()},
               'by_position': {k: _summary_bucket(v) for k, v in by_position.items()},
               'by_spr': {k: _summary_bucket(v) for k, v in by_spr.items()},
               'mean_policy_entropy': None,
               'mean_normalized_entropy': None}
        if actor is not None and self.samples:
            old_mode = actor.training
            entropy = normalized = 0.0
            try:
                actor.eval()
                with torch.inference_mode():
                    for start in range(0, len(self.samples), entropy_batch_size):
                        part = self.samples[start:start + entropy_batch_size]
                        enc = [encode_actor(obs, big_blind=big_blind,
                                            max_tokens=actor.config.max_tokens) for obs in part]
                        batch = batch_actors(enc, device=device)
                        distribution = actor.distribution(batch)
                        ent = distribution.entropy()
                        num_legal = batch.legal_action_mask.sum(dim=-1)
                        nent = torch.where(num_legal > 1,
                                           ent / num_legal.clamp_min(2).log(),
                                           torch.zeros_like(ent))
                        entropy += float(ent.sum().item())
                        normalized += float(nent.sum().item())
            finally:
                actor.train(old_mode)
            if not all(math.isfinite(x) for x in (entropy, normalized)):
                raise FloatingPointError('non-finite policy entropy')
            out['mean_policy_entropy'] = round(entropy / len(self.samples), 6)
            out['mean_normalized_entropy'] = round(normalized / len(self.samples), 6)
        return out

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/strategy_diagnostics.py" -->

## src/holdem_ai/evaluation/strategy_report.py

Original SHA `ead06dce6233b21f3c469e9f8f32a72e28a9a342` (4153 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/strategy_report.py" sha="ead06dce6233b21f3c469e9f8f32a72e28a9a342" -->

````````python
"""Obsidian-compatible Phase16 benchmark report from JSON-safe results."""
from __future__ import annotations

from holdem_ai.evaluation.action_diagnostics import ACTION_LABELS


def markdown_report(result: dict) -> str:
    cfg = result['config']
    summary = result['pool_summary']
    lines = [
        '# Phase 16 — HUNL200 Opponent Pool / Strategy Report',
        '',
        f"- Actor fingerprint: `{result['candidate_id']}`",
        f"- Mode: `{cfg['mode']}`; device: `{cfg['device']}`",
        f"- Deal seed range: `{cfg['seed']}` ... `{cfg['seed']+cfg['groups']-1}`",
        f"- Each opponent: $K={cfg['groups']}$ independent seed blocks, $4K={4*cfg['groups']}$ hands.",
        '',
        '## Opponent-pool performance', '',
        '| Opponent | bb/100 | CI 95% | Hands |',
        '|---|---:|---:|---:|',
    ]
    for row in result['opponents']:
        ci = f"[{row['ci95_lower_bb100']:+,.2f}, {row['ci95_upper_bb100']:+,.2f}]"
        lines.append(f"| {row['opponent']} | {row['mean_bb100']:+,.2f} | {ci} | {row['hands']} |")
    lines += [
        '',
        f"Equal-weight mean across these fixed opponents: **{summary['equal_weight_mean_bb100']:+,.2f} bb/100**.",
        f"Worst *observed* pool mean: **{summary['worst_observed_mean_bb100']:+,.2f} bb/100** "
        f"against **{summary['worst_opponent']}**.",
        '',
        '## Action diagnostics', '',
        'Choice frequency is conditional on the corresponding action being legal:',
        '',
        '$$\\widehat p(a\\mid a\\in\\mathcal A(I))=N_{\\mathrm{chosen}}(a)/N_{\\mathrm{legal}}(a).$$',
        '',
        'Policy entropy: $H(\\pi)=-\\sum_{a\\in\\mathcal A(I)}\\pi(a\\mid I)\\log\\pi(a\\mid I)$.',
        '',
    ]
    for row in result['opponents']:
        profile = row['candidate_action_profile']
        lines += [f"### vs {row['opponent']}", '',
                  f"- Actor decisions: **{profile['decisions']}**",
                  f"- Mean policy entropy: **{profile['mean_policy_entropy']}**",
                  f"- Mean normalized entropy: **{profile['mean_normalized_entropy']}**",
                  '', '| Action | Chosen | Legal opportunities | Chosen when legal |',
                  '|---|---:|---:|---:|']
        for a in ACTION_LABELS:
            freq = profile['conditional_action_rate'][a]
            rate = 'N/A' if freq is None else f'{100*freq:.2f}%'
            lines.append(f"| {a} | {profile['actions'][a]} | {profile['legal_opportunities'][a]} | {rate} |")
        for dimension in ('by_street', 'by_position', 'by_spr'):
            lines += ['', f'**{dimension}**', '',
                      '| Group | Decisions | Fold when legal | Check when legal | Call when legal | All-in when legal |',
                      '|---|---:|---:|---:|---:|---:|']
            for key, part in profile[dimension].items():
                rates = part['conditional_action_rate']
                fmt = lambda x: 'N/A' if x is None else f'{100*x:.1f}%'
                lines.append(f"| {key} | {part['decisions']} | {fmt(rates['fold'])} | "
                             f"{fmt(rates['check'])} | {fmt(rates['call'])} | {fmt(rates['all_in'])} |")
        lines.append('')
    lines += [
        '## Statistical protocol and limits', '',
        '$$X_k=\\frac14\\sum_{b=0}^1\\sum_{s=0}^1R_{k,b,s},\\qquad '
        '\\widehat J_{\\mathrm{bb/100}}=100\\cdot\\frac1K\\sum_k X_k.$$',
        '',
        '$$\\mathrm{CI}_{95\\%}\\approx 100\\left(\\overline X\\pm1.96\\frac{s_X}{\\sqrt K}\\right).$$',
        '',
        '- The unit for uncertainty is a whole deck-seed block, not an individual hand.',
        '- The pool mean and worst-opponent value are *descriptive*. They are not exploitability or guarantees.',
        '- Baseline opponents are simple heuristic policies, not poker professionals or GTO agents.',
        '- Different agents can induce different post-action paths even with the same deck seed.',
        '- This evaluation does not train or modify the PPO/Q-Boost model.',
        '- Avoid tuning repeatedly against a held-out test bank; reserve fresh seeds for final tests.',
        '',
    ]
    return '\n'.join(lines)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/strategy_report.py" -->

## src/holdem_ai/evaluation/terminal_reward_audit.py

Original SHA `d9d2152728ebbe93f8091dc4c9c27c7058185d50` (12130 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/terminal_reward_audit.py" sha="d9d2152728ebbe93f8091dc4c9c27c7058185d50" -->

````````python
"""Read-only per-HAND payoff and preflop-exit audit for HUNL PPO.

Uses the existing agent trajectory boundary (decisions_per_hand) and actor
observations. Does not access hidden cards or change the game engine.

In heads-up engine play a terminal Fold action ends a hand without showdown;
all other terminations go to pot settlement with two non-folded contenders.
'last_decision_street=preflop' does not mean the BOARD stayed at preflop:
an all-in can run out the entire board without further actor decisions.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.training.ppo import PPOConfig, TrajectoryRows

STREETS = ("preflop", "flop", "turn", "river")
ACTION_NAMES = ("fold", "check", "call", "raise_half_pot", "raise_pot",
                "raise_double_pot", "all_in")
# Absolute seat-0 rewards, in BB. Every HUNL200 hand starts with 200BB stacks.
ABS_BINS = ((0, 1), (1, 10), (10, 50), (50, 100),
            (100, 150), (150, 190), (190, 199), (199, float("inf")))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    low = int(math.floor(at))
    high = int(math.ceil(at))
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def _distribution(values: list[float], *, stack_bb: float) -> dict:
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("finite, nonempty terminal rewards required")
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((x - mean) ** 2 for x in values) / n
    av = [abs(x) for x in values]
    bins = {}
    for left, right in ABS_BINS:
        name = f"[{left},{right})" if math.isfinite(right) else f"[{left},inf)"
        bins[name] = sum(left <= x < right for x in av)
    return {
        "count": n,
        "mean_bb": round(mean, 6),
        "std_bb": round(math.sqrt(var), 6),
        "min_bb": round(min(values), 6),
        "max_bb": round(max(values), 6),
        "mean_absolute_bb": round(math.fsum(av) / n, 6),
        "abs_quantiles_bb": {
            f"p{int(q*100)}": round(_percentile(av, q), 6)
            for q in (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
        },
        "abs_histogram_counts": bins,
        "abs_ge_190bb_fraction": round(sum(x >= 190 for x in av) / n, 6),
        "abs_ge_199bb_fraction": round(sum(x >= 199 for x in av) / n, 6),
        "near_full_stack_fraction": round(
            sum(x >= stack_bb - 1e-5 for x in av) / n, 6),
        "signed_full_stack": {
            "win": sum(x >= stack_bb - 1e-5 for x in values),
            "loss": sum(x <= -(stack_bb - 1e-5) for x in values),
        },
    }


def summarize_terminal_distribution(rows: TrajectoryRows, *,
                                    stack_bb: float = 200.0) -> dict:
    """Read terminal HAND rewards with their exact sampled action histories.

    This is deliberately hand-weighted. The prior Critic diagnosis was
    decision-weighted; multiple decisions within a hand share the same reward.
    """
    if not math.isfinite(stack_bb) or stack_bb <= 0:
        raise ValueError("stack_bb must be positive finite")
    if (not rows.decisions_per_hand or not rows.terminal_actions
            or len(rows.decisions_per_hand) != rows.hands
            or len(rows.terminal_actions) != rows.hands):
        raise ValueError("collector must preserve per-hand boundaries and terminal actions")
    hands = []
    offset = 0
    for index, (length, last_action, reward) in enumerate(
            zip(rows.decisions_per_hand, rows.terminal_actions, rows.hand_rewards)):
        if length <= 0:
            raise ValueError("empty hand decisions")
        end = offset + length
        if end > len(rows.actions):
            raise ValueError("hand boundary exceeds rows")
        actions = rows.actions[offset:end]
        actors = rows.actors[offset:end]
        if actions[-1] != last_action:
            raise ValueError("terminal action does not match final action")
        streets = []
        for actor in actors:
            st = int(actor.features[2])
            if st not in range(4) or float(st) != actor.features[2]:
                raise ValueError("invalid actor street feature")
            streets.append(st)
        if any(b < a for a, b in zip(streets, streets[1:])):
            raise ValueError("hand street regressed")
        last_street = STREETS[streets[-1]]
        fold = last_action == 0
        preflop_all_in = any(a == 6 and st == 0 for a, st in zip(actions, streets))
        any_all_in = 6 in actions
        r0, r1 = map(float, reward)
        if not math.isfinite(r0) or not math.isfinite(r1) or not math.isclose(r0 + r1, 0, abs_tol=1e-7):
            raise ValueError("non-zero-sum or nonfinite terminal reward")
        # All-in via CALL is not counted as an explicit all-in action; report it
        # separately as a classification limitation.
        hands.append({
            "hand_index": index,
            "seat0_payoff_bb": r0,
            "abs_payoff_bb": abs(r0),
            "decisions": length,
            "last_decision_street": last_street,
            "terminal_action": ACTION_NAMES[last_action],
            "termination": "fold" if fold else "nonfold_showdown",
            "explicit_preflop_all_in": preflop_all_in,
            "explicit_all_in_any_street": any_all_in,
            "has_flop_decision": any(s >= 1 for s in streets),
            "has_turn_decision": any(s >= 2 for s in streets),
            "has_river_decision": any(s >= 3 for s in streets),
        })
        offset = end
    if offset != len(rows.actions):
        raise ValueError("hand boundary does not consume every decision")

    n = len(hands)
    payoffs = [h["seat0_payoff_bb"] for h in hands]
    decisions_weighted = [rows.returns_bb[i] for i in range(len(rows.returns_bb))]
    term = Counter(h["termination"] for h in hands)
    stage = Counter(h["last_decision_street"] for h in hands)
    stage_term = Counter(
        f"{h['last_decision_street']}:{h['termination']}" for h in hands)
    preflop = [h for h in hands if h["last_decision_street"] == "preflop"]
    preflop_nonfold = [h for h in preflop if h["termination"] == "nonfold_showdown"]
    full_stack = [h for h in hands if h["abs_payoff_bb"] >= stack_bb - 1e-5]

    payoff_by_exit_group = {}
    for group, subset in (
        ("preflop:fold", [h for h in preflop if h["termination"] == "fold"]),
        ("preflop:showdown", preflop_nonfold),
        ("postflop:fold", [h for h in hands if h["last_decision_street"] != "preflop"
                           and h["termination"] == "fold"]),
        ("postflop:showdown", [h for h in hands if h["last_decision_street"] != "preflop"
                               and h["termination"] == "nonfold_showdown"]),
    ):
        payoff_by_exit_group[group] = (
            _distribution([h["seat0_payoff_bb"] for h in subset], stack_bb=stack_bb)
            if subset else None
        )

    return {
        "protocol": "phase15_on_policy_terminal_hand_audit",
        "hands": n,
        "decisions": len(rows.actions),
        "seat0_hand_weighted_reward": _distribution(payoffs, stack_bb=stack_bb),
        "actor_decision_weighted_return": _distribution(
            decisions_weighted, stack_bb=stack_bb),
        "termination_counts": dict(sorted(term.items())),
        "last_decision_street_counts": {
            key: stage[key] for key in STREETS
        },
        "payoff_by_exit_group": payoff_by_exit_group,
        "last_decision_street_x_termination": {
            f"{street}:{reason}": stage_term[f"{street}:{reason}"]
            for street in STREETS for reason in ("fold", "nonfold_showdown")
        },
        "preflop": {
            "last_decision_hand_count": len(preflop),
            "last_decision_hand_fraction": round(len(preflop) / n, 6),
            "fold_ended_count": sum(h["termination"] == "fold" for h in preflop),
            "nonfold_showdown_count": len(preflop_nonfold),
            "last_action_counts": {
                name: sum(h["terminal_action"] == name for h in preflop)
                for name in ACTION_NAMES
            },
            "explicit_all_in_hand_count": sum(h["explicit_preflop_all_in"] for h in hands),
            "explicit_all_in_and_nonfold_last_preflop": sum(
                h["explicit_preflop_all_in"] for h in preflop_nonfold),
            "nonfold_last_preflop_without_explicit_allin_count": sum(
                not h["explicit_preflop_all_in"] for h in preflop_nonfold),
            "payoff_abs_mean_bb": (
                round(math.fsum(h["abs_payoff_bb"] for h in preflop) / len(preflop), 6)
                if preflop else None
            ),
        },
        "decision_coverage": {
            "flop_hands": sum(h["has_flop_decision"] for h in hands),
            "turn_hands": sum(h["has_turn_decision"] for h in hands),
            "river_hands": sum(h["has_river_decision"] for h in hands),
        },
        "full_stack_settlements": {
            "hand_count": len(full_stack),
            "preflop_last_decision_count": sum(
                h["last_decision_street"] == "preflop" for h in full_stack),
            "explicit_preflop_all_in_count": sum(
                h["explicit_preflop_all_in"] for h in full_stack),
        },
        "mean_decisions_per_hand": round(
            sum(h["decisions"] for h in hands) / n, 6),
        "notes": [
            "Last decision street does not equal showdown board street: "
            "preflop all-ins can auto-run out all board cards.",
            "In HUNL heads-up, a final fold yields fold termination; otherwise "
            "pot settlement is by showdown. An explicit all-in refers ONLY "
            "to the Actor head index 6; a CALL may also exhaust a stack.",
            "Distributions are hand-weighted unless labeled actor-decision-weighted.",
            "This is a behavioral analysis, not EV, Nash exploitability, or "
            "evidence of a causal impact of a particular action.",
        ],
    }


def run_terminal_reward_audit(*, checkpoints: dict[str, str | Path],
                              device: str = "cpu", hands: int = 512,
                              seed: int = 580_000_000, env_batch_size: int = 32,
                              stack_bb: float = 200.0) -> dict:
    """Sample each frozen checkpoint on distinct, unused deck seeds.

    Uses the SAME deck seed block for each candidate model but each model
    samples its own actions; hence the played states and hand outcomes differ.
    """
    if not checkpoints or len(checkpoints) != len(set(checkpoints)):
        raise ValueError("nonempty unique checkpoint labels required")
    if type(hands) is not int or hands < 1 or type(seed) is not int or seed < 0:
        raise ValueError("hands and seed must be positive/nonnegative integers")
    if type(env_batch_size) is not int or not 1 <= env_batch_size <= hands:
        raise ValueError("invalid env_batch_size")
    report = {}
    for label, path in checkpoints.items():
        actor = load_actor(path, device=device)
        config = PPOConfig(seed=seed, hands_per_update=hands)
        rows = collect_batched_self_play(
            actor, config, iteration=0, device=device,
            env_batch_size=env_batch_size)
        report[label] = summarize_terminal_distribution(rows, stack_bb=stack_bb)
    return {
        "protocol": "phase15_separate_on_policy_reward_distributions",
        "seed": seed, "hands_per_checkpoint": hands, "device": device,
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "models": report,
        "warning": "Same deck seed block but policies sample different actions "
                   "and visit different states; descriptive comparison only. "
                   "A large absolute payoff does not by itself imply profitable "
                   "or irrational aggression.",
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/terminal_reward_audit.py" -->

## src/holdem_ai/evaluation/tournament_eval.py

Original SHA `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (0 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/tournament_eval.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/tournament_eval.py" -->

## src/holdem_ai/expand/__init__.py

Original SHA `ef54dfeb01d3748ef313eed9e4f2a4222ab73c5c` (77 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/__init__.py" sha="ef54dfeb01d3748ef313eed9e4f2a4222ab73c5c" -->

````````python
"""6-Max expansion modules. Existing Base Phase imports remain unchanged."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/__init__.py" -->

## src/holdem_ai/expand/environment/__init__.py

Original SHA `ef5a2d4a73f8bfe28fbc50e2ce9d10fb74419cda` (561 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/environment/__init__.py" sha="ef5a2d4a73f8bfe28fbc50e2ce9d10fb74419cda" -->

````````python
"""Expand Phase 1 public table API."""
from .table_state import (TableConfig, PlayerSeat, TableSnapshot, Action,
                          HandHandle, StepEvent, Settlement)
from .cash_rules import CashRuleSet
from .tournament_rules import TournamentRuleSet
from .table_manager import DynamicTable, replay_journal
from .event_ledger import EventLedger

__all__ = [
    'TableConfig', 'PlayerSeat', 'TableSnapshot', 'Action', 'HandHandle',
    'StepEvent', 'Settlement', 'CashRuleSet', 'TournamentRuleSet',
    'DynamicTable', 'replay_journal', 'EventLedger',
]

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/environment/__init__.py" -->

## src/holdem_ai/expand/environment/cash_rules.py

Original SHA `209cdbebeba28c20636ccf191deb1597dd572e1b` (1073 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/environment/cash_rules.py" sha="209cdbebeba28c20636ccf191deb1597dd572e1b" -->

````````python
"""E1 cash table admission rules: fixed blinds, no rake, wait for BB."""
from __future__ import annotations
from dataclasses import dataclass
from .table_state import TableConfig


@dataclass(frozen=True)
class CashRuleSet:
    config: TableConfig

    def __post_init__(self):
        if self.config.mode != 'cash':
            raise ValueError('CashRuleSet requires cash mode')

    @property
    def min_buyin(self) -> int:
        return self.config.big_blind * self.config.min_buyin_bb

    @property
    def max_buyin(self) -> int:
        return self.config.big_blind * self.config.max_buyin_bb

    def validate_buyin(self, chips: int) -> None:
        if type(chips) is not int or not self.min_buyin <= chips <= self.max_buyin:
            raise ValueError(f'buyin must be integer chips in [{self.min_buyin}, {self.max_buyin}]')

    def validate_rebuy(self, stack: int, chips: int) -> None:
        if type(chips) is not int or chips <= 0 or stack + chips > self.max_buyin:
            raise ValueError('top-up must be positive and resulting stack <= max buyin')

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/environment/cash_rules.py" -->

## src/holdem_ai/expand/environment/event_ledger.py

Original SHA `a46f71f4ab874a700fdd5b6e31c5b755f7441e23` (1623 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/environment/event_ledger.py" sha="a46f71f4ab874a700fdd5b6e31c5b755f7441e23" -->

````````python
"""Append-only deterministic journal; privileged seed log, not an actor observation."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any


class EventLedger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self.records: list[dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                raise FileExistsError('journal exists; use a fresh journal path')
            # File created at construction, never clobber an existing journal.
            self.path.open('x', encoding='utf-8').close()

    def append(self, event: str, **data: Any) -> dict[str, Any]:
        record = {'seq': len(self.records) + 1, 'event': event, **data}
        # Reject nonportable/nonserializable records up front.
        wire = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if self.path is not None:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(wire + '\n')
                f.flush()
                os.fsync(f.fileno())
        self.records.append(record)
        return record

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open(encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if any(row.get('seq') != i for i, row in enumerate(rows, 1)):
            raise ValueError('ledger sequence is discontinuous')
        return rows

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/environment/event_ledger.py" -->

## src/holdem_ai/expand/environment/multiway_adapter.py

Original SHA `08b59672f57dae15c709595b2d82731da4680043` (943 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/environment/multiway_adapter.py" sha="08b59672f57dae15c709595b2d82731da4680043" -->

````````python
"""Late-bound adapter to the unchanged Base Phase 3 6-seat hand engine."""
from __future__ import annotations


def make_hand_engine(**kwargs):
    from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine
    return MultiPlayerHandEngine(**kwargs)


def moving_button_after(previous: int | None, active: tuple[int, ...]) -> int:
    if not active:
        raise ValueError('no active seat')
    if previous is None:
        return active[0]
    for offset in range(1, 7):
        candidate = (previous + offset) % 6
        if candidate in active:
            return candidate
    raise AssertionError('no next button')


def blind_seats(button: int, active: tuple[int, ...]) -> tuple[int, int]:
    from holdem_ai.environment.multi_player_seats import multi_player_seats
    stacks = [100 if i in active else 0 for i in range(6)]
    seats = multi_player_seats(stacks, button)
    return seats.small_blind, seats.big_blind

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/environment/multiway_adapter.py" -->

## src/holdem_ai/expand/environment/table_manager.py

Original SHA `934e587e718047ef8c8e361cd7609a640c8dfd06` (22864 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/environment/table_manager.py" sha="934e587e718047ef8c8e361cd7609a640c8dfd06" -->

````````python
"""Expand Phase 1: dynamic 6-seat cash sessions on the unchanged Phase 3 hand engine.

Events change *eligible rosters* only between hands. The hand engine owns
chips committed to pots; the table manager owns external buyin/cashout flow.
This module intentionally does not implement RL, PPO, or MCCFR.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Any
import json

from .cash_rules import CashRuleSet
from .event_ledger import EventLedger
from .multiway_adapter import blind_seats, make_hand_engine, moving_button_after
from .table_state import (Action, HandHandle, PlayerSeat, Settlement,
                          StepEvent, TableConfig, TableSnapshot)


@dataclass
class _Seat:
    player_id: str | None = None
    stack: int = 0
    status: str = 'vacant'


@dataclass(frozen=True)
class _QueueEntry:
    event_id: str
    kind: str
    player_id: str
    seat: int
    chips: int = 0


class DynamicTable:
    """Cash-only E1 manager. Tournament mode is a hard failure until E7.

    `engine_factory` is injectable for isolated tests; default uses the actual
    Base Phase 3 engine, without patching it. Not thread-safe: serialize table
    actions through one owner process, as with a poker table event loop.
    """

    def __init__(self, config: TableConfig | None = None, *, session_id: str = 'expand-e01',
                 journal_path: str | Path | None = None,
                 engine_factory: Callable[..., Any] | None = None):
        self.config = config if config is not None else TableConfig()
        if self.config.mode != 'cash':
            raise NotImplementedError('Tournament play is Expand Phase 7; E1 is cash only')
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('nonempty session_id is required')
        self.rules = CashRuleSet(self.config)
        self.session_id = session_id
        self.ledger = EventLedger(journal_path)
        self._engine_factory = engine_factory or make_hand_engine
        self._seats = [_Seat() for _ in range(6)]
        self._pending: list[_QueueEntry] = []
        self._event_id = 0
        self._hand_number = 0
        self._button: int | None = None
        self._hand: Any | None = None
        self._hand_seed: int | None = None
        self._hand_id: str | None = None
        self._hand_roster: dict[int, str] = {}
        self._hand_initial: tuple[int, ...] | None = None
        self.total_buyins = 0
        self.total_cashouts = 0
        self.ledger.append('table_created', config=asdict(self.config), session_id=self.session_id)

    def _next_event_id(self) -> str:
        self._event_id += 1
        return f'{self.session_id}:event:{self._event_id:08d}'

    def _require_player(self, player_id: str) -> int:
        if not isinstance(player_id, str) or not player_id:
            raise ValueError('player_id must be nonempty string')
        for i, seat in enumerate(self._seats):
            if seat.player_id == player_id:
                return i
        raise ValueError('unknown player_id')

    def _open_hand(self) -> bool:
        return self._hand is not None

    def _validate_seat(self, seat_id: int) -> None:
        if type(seat_id) is not int or not 0 <= seat_id < 6:
            raise ValueError('seat must be integer in 0..5')

    def _money_check(self) -> None:
        if self._open_hand():
            live = sum(self._hand.stacks) + sum(self._hand.contributions) + sum(
                seat.stack for i, seat in enumerate(self._seats) if i not in self._hand_roster)
        else:
            live = sum(seat.stack for seat in self._seats)
        if live != self.total_buyins - self.total_cashouts:
            raise AssertionError(f'cash conservation: {live} != {self.total_buyins}-{self.total_cashouts}')
        if any(s.stack < 0 for s in self._seats):
            raise AssertionError('negative seat stack')
        ids = [s.player_id for s in self._seats if s.player_id is not None]
        if len(ids) != len(set(ids)) or len(self._seats) != 6:
            raise AssertionError('invalid seat registry')

    def request_join(self, player_id: str, buyin_chips: int,
                     preferred_seat: int | None = None) -> str:
        if not isinstance(player_id, str) or not player_id:
            raise ValueError('nonempty player_id is required')
        if any(seat.player_id == player_id for seat in self._seats):
            raise ValueError('player already occupies or reserved a seat')
        self.rules.validate_buyin(buyin_chips)
        if preferred_seat is None:
            empty = [i for i, s in enumerate(self._seats) if s.status == 'vacant']
            if not empty:
                raise ValueError('table full')
            preferred_seat = empty[0]
        self._validate_seat(preferred_seat)
        seat = self._seats[preferred_seat]
        if seat.status != 'vacant':
            raise ValueError('seat already occupied or reserved')
        eid = self._next_event_id()
        self._seats[preferred_seat] = _Seat(player_id=player_id, stack=0, status='reserved')
        self._pending.append(_QueueEntry(eid, 'join', player_id, preferred_seat, buyin_chips))
        self.ledger.append('request', event_id=eid, kind='join', player_id=player_id,
                           seat=preferred_seat, chips=buyin_chips)
        return eid

    def request_leave(self, player_id: str) -> str:
        i = self._require_player(player_id)
        if any(e.player_id == player_id and e.kind == 'leave' for e in self._pending):
            raise ValueError('leave already pending')
        eid = self._next_event_id()
        self._pending.append(_QueueEntry(eid, 'leave', player_id, i))
        # Status-only change; live hand roster remains immutable.
        if self._seats[i].status != 'reserved':
            self._seats[i].status = 'pending_leave'
        self.ledger.append('request', event_id=eid, kind='leave', player_id=player_id, seat=i, chips=0)
        return eid

    def request_rebuy(self, player_id: str, amount_chips: int) -> str:
        i = self._require_player(player_id)
        seat = self._seats[i]
        if seat.status in ('reserved', 'pending_leave') or any(
                e.player_id == player_id and e.kind == 'leave' for e in self._pending):
            raise ValueError('rebuy is unavailable for reserved/leaving player')
        pending_rebuys = sum(e.chips for e in self._pending if e.player_id == player_id and e.kind == 'rebuy')
        # During a hand, the live stack is the correct basis, but buyin cap is
        # evaluated against the original hand-start stack to avoid pot-dependent
        # mid-hand top-up qualification.
        basis = self._hand_initial[i] if self._open_hand() and i in self._hand_roster else seat.stack
        self.rules.validate_rebuy(basis + pending_rebuys, amount_chips)
        eid = self._next_event_id()
        self._pending.append(_QueueEntry(eid, 'rebuy', player_id, i, amount_chips))
        self.ledger.append('request', event_id=eid, kind='rebuy', player_id=player_id,
                           seat=i, chips=amount_chips)
        return eid

    def request_sitout(self, player_id: str) -> str:
        return self._queue_status(player_id, 'sitout')

    def request_resume(self, player_id: str) -> str:
        return self._queue_status(player_id, 'resume')

    def _queue_status(self, player_id: str, kind: str) -> str:
        i = self._require_player(player_id)
        seat = self._seats[i]
        if seat.status in ('reserved', 'pending_leave') or any(
                e.player_id == player_id and e.kind in ('sitout', 'resume', 'leave') for e in self._pending):
            raise ValueError('conflicting pending status request')
        if kind == 'sitout' and seat.status not in ('seated', 'waiting_bb'):
            raise ValueError('player cannot sit out')
        if kind == 'resume' and seat.status != 'sitting_out':
            raise ValueError('player is not sitting out')
        eid = self._next_event_id()
        self._pending.append(_QueueEntry(eid, kind, player_id, i))
        self.ledger.append('request', event_id=eid, kind=kind, player_id=player_id, seat=i, chips=0)
        return eid

    def apply_boundary_events(self) -> None:
        if self._open_hand():
            raise RuntimeError('cannot apply pending changes during a hand')
        self.ledger.append('boundary_call')
        self._apply_pending()

    def _apply_pending(self) -> None:
        if self._open_hand():
            raise RuntimeError('cannot modify roster during an open hand')
        pending, self._pending = self._pending, []
        for entry in pending:
            seat = self._seats[entry.seat]
            if seat.player_id != entry.player_id:
                # A prior leave in this same boundary removes later commands.
                self.ledger.append('boundary_skip', event_id=entry.event_id, reason='player_departed')
                continue
            if entry.kind == 'join':
                if seat.status != 'reserved':
                    raise AssertionError('reservation mismatch')
                seat.stack = entry.chips
                seat.status = 'waiting_bb'
                self.total_buyins += entry.chips
                self.ledger.append('boundary_join', event_id=entry.event_id,
                                   player_id=entry.player_id, seat=entry.seat, buyin=entry.chips)
            elif entry.kind == 'leave':
                chips = seat.stack
                self.total_cashouts += chips
                self._seats[entry.seat] = _Seat()
                self.ledger.append('boundary_leave', event_id=entry.event_id,
                                   player_id=entry.player_id, seat=entry.seat, cashout=chips)
            elif entry.kind == 'rebuy':
                seat.stack += entry.chips
                self.total_buyins += entry.chips
                if seat.status in ('busted', 'waiting_bb') or seat.stack == entry.chips:
                    seat.status = 'waiting_bb'
                self.ledger.append('boundary_rebuy', event_id=entry.event_id,
                                   player_id=entry.player_id, seat=entry.seat, chips=entry.chips)
            elif entry.kind == 'sitout':
                seat.status = 'sitting_out'
                self.ledger.append('boundary_sitout', event_id=entry.event_id, seat=entry.seat)
            elif entry.kind == 'resume':
                seat.status = 'waiting_bb'
                self.ledger.append('boundary_resume', event_id=entry.event_id, seat=entry.seat)
            else:
                raise AssertionError('invalid boundary command')
        # Final stacks of 0 imply actual cash bust only AFTER the hand settles.
        # A queued rebuy can save a busted player's original seat.
        for i, seat in enumerate(self._seats):
            if seat.player_id is not None and seat.stack == 0 and seat.status != 'reserved':
                player_id = seat.player_id
                self._seats[i] = _Seat()
                self.ledger.append('boundary_bust', seat=i, player_id=player_id)
        self._money_check()

    def _choose_roster(self) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
        active = tuple(i for i, s in enumerate(self._seats)
                       if s.status == 'seated' and s.stack > 0)
        waiting = tuple(i for i, s in enumerate(self._seats)
                        if s.status == 'waiting_bb' and s.stack > 0)
        activated: list[int] = []
        if len(active) < 2:
            # Bootstrap a stalled table. A two-person table must start even
            # though there was no prior BB rotation to wait for.
            need = 2 - len(active)
            if len(waiting) < need:
                return None
            activated.extend(waiting[:need])
            active = tuple(sorted(active + tuple(activated)))
            waiting = tuple(i for i in waiting if i not in activated)
            button = moving_button_after(self._button, active)
            return button, active, tuple(activated)

        button = moving_button_after(self._button, active)
        # Admit at most one additional waiting player, and only if they will
        # post the big blind in THIS hand. Never grant free entry at BTN/CO.
        for candidate in waiting:
            roster = tuple(sorted(active + (candidate,)))
            _, bb = blind_seats(button, roster)
            if bb == candidate:
                activated.append(candidate)
                active = roster
                break
        return button, active, tuple(activated)

    def start_hand(self, seed: int) -> HandHandle | None:
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be nonnegative integer')
        if self._open_hand():
            raise RuntimeError('previous hand must be settled first')
        # Log the external command BEFORE boundary effects for strict replay.
        self.ledger.append('command_start_hand', seed=seed)
        # Commit queued table operations at the boundary, before selecting
        # participants; joins never occur mid-hand.
        self._apply_pending()
        selection = self._choose_roster()
        if selection is None:
            self.ledger.append('waiting_for_players', seated=sum(s.status == 'seated' for s in self._seats),
                               waiting=sum(s.status == 'waiting_bb' for s in self._seats))
            return None
        button, roster, admitted = selection
        for i in admitted:
            self._seats[i].status = 'seated'
        for i in admitted:
            self.ledger.append('big_blind_admission', seat=i, player_id=self._seats[i].player_id,
                               bootstrap=len(roster) == 2)
        stacks = tuple(self._seats[i].stack if i in roster else 0 for i in range(6))
        engine = self._engine_factory(stacks=stacks, button=button,
                                      small_blind=self.config.small_blind,
                                      big_blind=self.config.big_blind, seed=seed)
        engine.reset()
        self._hand = engine
        self._hand_seed = seed
        self._hand_number += 1
        self._hand_id = f'{self.session_id}:hand:{self._hand_number:08d}'
        self._hand_roster = {i: self._seats[i].player_id for i in roster}
        self._hand_initial = stacks
        self._button = button
        self._money_check()
        self.ledger.append('hand_started', hand_id=self._hand_id, hand_number=self._hand_number,
                           button=button, sb=engine.seats.small_blind, bb=engine.seats.big_blind,
                           participants=[[i, self._seats[i].player_id] for i in roster],
                           starting_stacks=list(stacks), seed=seed)
        return HandHandle(hand_id=self._hand_id, hand_number=self._hand_number,
                          button_seat=button, small_blind_seat=engine.seats.small_blind,
                          big_blind_seat=engine.seats.big_blind,
                          participants=tuple((i, self._seats[i].player_id) for i in roster),
                          seed=seed, terminal=engine.is_over())

    @property
    def current_player_id(self) -> str | None:
        if not self._open_hand() or self._hand.is_over():
            return None
        return self._hand_roster[self._hand.current_player]

    def observe_current(self) -> dict:
        if self.current_player_id is None:
            raise RuntimeError('no pending decision')
        # Base engine returns only the acting player's hole cards. This raw
        # representation is temporary: E2 adds a formal InformationState.
        return dict(self._hand.observe(self._hand.current_player))

    def act(self, player_id: str, action: Action) -> StepEvent:
        if not self._open_hand() or self._hand.is_over():
            raise RuntimeError('no current hand action')
        if player_id != self.current_player_id:
            raise ValueError('not the acting player')
        if not isinstance(action, Action):
            raise TypeError('action must be Action')
        actor = self._hand.current_player
        observation = self._hand.observe(actor)['raw_obs']
        to_call = observation['to_call']
        if action.kind == 'check' and to_call != 0:
            raise ValueError('cannot check when facing a bet')
        if action.kind == 'call' and to_call == 0:
            raise ValueError('cannot call when check is free')
        kind_action = {
            'fold': 0, 'check': 1, 'call': 1,
            'raise_half_pot': 2, 'raise_pot': 3,
        }
        if action.kind == 'raise_to':
            self._hand.step_raise_to(action.raise_to_chips)
        elif action.kind == 'all_in':
            # In the Base engine, short all-in calls are represented by action
            # 1 (check/call), while action 4 is an aggressive all-in raise.
            chosen = 1 if self._hand.stacks[actor] <= to_call else 4
            self._hand.step(chosen)
        else:
            self._hand.step(kind_action[action.kind])
        self._money_check()
        self.ledger.append('command_action', hand_id=self._hand_id, player_id=player_id,
                           seat=actor, action=asdict(action))
        nxt = None if self._hand.is_over() else self._hand.current_player
        return StepEvent(hand_id=self._hand_id, actor_seat=actor, player_id=player_id,
                         action=action, terminal=self._hand.is_over(), next_actor_seat=nxt)

    def settle_hand(self) -> Settlement:
        if not self._open_hand() or not self._hand.is_over():
            raise RuntimeError('settle requires a terminal, unsettled hand')
        result = self._hand.get_result()
        initial = self._hand_initial
        if (len(result.final_stacks) != 6 or len(result.payoffs_chips) != 6
                or sum(result.final_stacks) != sum(initial)
                or sum(result.payoffs_chips) != 0
                or any(x < 0 for x in result.final_stacks)):
            raise AssertionError('invalid or non-zero-sum hand settlement')
        busts = tuple(self._hand_roster[i] for i in self._hand_roster if result.final_stacks[i] == 0)
        settled = Settlement(hand_id=self._hand_id, hand_number=self._hand_number,
                             starting_stacks=initial, final_stacks=tuple(result.final_stacks),
                             payoffs_chips=tuple(result.payoffs_chips),
                             payoffs_bb=tuple(x / self.config.big_blind for x in result.payoffs_chips),
                             busted_player_ids=busts, button_seat=self._button)
        self.ledger.append('command_settle', hand_id=self._hand_id)
        self.ledger.append('hand_settled', hand_id=self._hand_id,
                           final_stacks=list(result.final_stacks), payoffs=list(result.payoffs_chips),
                           busts=list(busts), board=list(result.board),
                           action_history=[list(x) for x in result.action_history],
                           action_raise_to=list(result.action_raise_to))
        for i in self._hand_roster:
            self._seats[i].stack = result.final_stacks[i]
            if self._seats[i].status == 'seated' and result.final_stacks[i] == 0:
                self._seats[i].status = 'busted'
        self._hand = None
        self._hand_id = None
        self._hand_roster = {}
        self._hand_initial = None
        self._hand_seed = None
        # Hand boundary is automatic: commit queued joins, departures, rebuys.
        self._apply_pending()
        return settled

    def snapshot(self) -> TableSnapshot:
        rows = []
        for i, seat in enumerate(self._seats):
            stack = self._hand.stacks[i] if self._open_hand() and i in self._hand_roster else seat.stack
            rows.append(PlayerSeat(seat_id=i, player_id=seat.player_id,
                                   stack_chips=stack, seat_status=seat.status))
        return TableSnapshot(config=self.config, seats=tuple(rows), button_seat=self._button,
                             hand_number=self._hand_number, session_id=self.session_id,
                             hand_in_progress=self._open_hand(),
                             current_actor_seat=self._hand.current_player if self.current_player_id is not None else None,
                             pending_event_count=len(self._pending),
                             total_chips=self.total_buyins - self.total_cashouts)


def replay_journal(path: str | Path, *, engine_factory: Callable[..., Any] | None = None) -> DynamicTable:
    """Re-execute external commands and verify all emitted effects exactly.

    NOTE: Journal contains shuffle seeds; it is privileged experiment data and
    must never be supplied to a learning actor as an observation.
    """
    events = EventLedger.read(path)
    if not events or events[0]['event'] != 'table_created':
        raise ValueError('missing journal header')
    original = events[0]
    table = DynamicTable(TableConfig(**original['config']),
                         session_id=original['session_id'], engine_factory=engine_factory)
    # Effects are regenerated, not blindly trusted. Compare as we advance.
    cursor = 1
    while cursor < len(events):
        record = events[cursor]
        tag = record['event']
        if tag == 'request':
            kind, pid = record['kind'], record['player_id']
            if kind == 'join':
                eid = table.request_join(pid, record['chips'], record['seat'])
            elif kind == 'leave':
                eid = table.request_leave(pid)
            elif kind == 'rebuy':
                eid = table.request_rebuy(pid, record['chips'])
            elif kind == 'sitout':
                eid = table.request_sitout(pid)
            elif kind == 'resume':
                eid = table.request_resume(pid)
            else:
                raise ValueError(f'unknown recorded request: {kind}')
            if eid != record['event_id']:
                raise ValueError('event ID mismatch')
        elif tag == 'boundary_call':
            table.apply_boundary_events()
        elif tag == 'command_start_hand':
            table.start_hand(record['seed'])
        elif tag == 'command_action':
            table.act(record['player_id'], Action(**record['action']))
        elif tag == 'command_settle':
            table.settle_hand()
        else:
            raise ValueError(f'unexpected journal command at seq {record["seq"]}: {tag}')
        produced = table.ledger.records
        if produced[:len(produced)] != events[:len(produced)]:
            raise ValueError(f'journal replay divergence near seq {cursor + 1}')
        cursor = len(produced)
    if table.ledger.records != events:
        raise ValueError('replayed event stream differs')
    return table

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/environment/table_manager.py" -->
