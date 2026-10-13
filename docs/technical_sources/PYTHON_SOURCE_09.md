# Python full source volume 9/28

[Canonical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) — original commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`; exact code with individual blob SHA.

## src/holdem_ai/evaluation/action_diagnostics.py

Original SHA `33af7d36ca10b9e7e4a88de17ec9001bd24b703f` (4003 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/action_diagnostics.py" sha="33af7d36ca10b9e7e4a88de17ec9001bd24b703f" -->

````````python
"""Action-rate and legal-opportunity diagnostics on an on-policy trajectory.

Rows contain only actor-visible encodings, actual action, and a training-only
critic row; never derive actor statistics from privileged critic features.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch

from holdem_ai.encoding.tensors import batch_actors
from holdem_ai.training.ppo import TrajectoryRows

STREETS = ('preflop', 'flop', 'turn', 'river')
ACTION_LABELS = ('fold', 'check', 'call', 'raise_half_pot',
                 'raise_pot', 'raise_double_pot', 'all_in')


def action_diagnostics(rows: TrajectoryRows, actor, *, device: str = 'cpu',
                       chunk_size: int = 64) -> dict:
    """Report actual frequencies and *conditional-on-legal* choice frequencies.

    A choice probability of zero for Check cannot be interpreted without
    knowing if Check was ever legal. Legal opportunities are tracked per action.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    counts = [0] * 7
    opportunities = [0] * 7
    by_street = {name: {'actions': [0]*7, 'legal_opportunities': [0]*7,
                        'decisions': 0} for name in STREETS}
    by_position = {name: {'actions': [0]*7, 'legal_opportunities': [0]*7,
                          'decisions': 0} for name in ('button', 'big_blind')}
    for obs, action in zip(rows.actors, rows.actions):
        street_index = int(obs.features[2])
        if street_index not in range(4) or float(street_index) != obs.features[2]:
            raise ValueError('invalid stage in actor features')
        street = STREETS[street_index]
        position = 'button' if obs.features[1] == 1.0 else 'big_blind'
        if obs.features[1] not in (0.0, 1.0):
            raise ValueError('invalid button indicator')
        counts[action] += 1
        for key in ('actions',):
            by_street[street][key][action] += 1
            by_position[position][key][action] += 1
        by_street[street]['decisions'] += 1
        by_position[position]['decisions'] += 1
        for i, legal in enumerate(obs.legal_action_mask):
            if legal:
                opportunities[i] += 1
                by_street[street]['legal_opportunities'][i] += 1
                by_position[position]['legal_opportunities'][i] += 1
    total = len(rows.actions)
    old_mode = actor.training
    entropy_total = 0.0
    entropy_norm_total = 0.0
    try:
        actor.eval()
        with torch.inference_mode():
            for start in range(0, total, chunk_size):
                batch = batch_actors(rows.actors[start:start + chunk_size], device=device)
                dist = actor.distribution(batch)
                h = dist.entropy()
                nlegal = batch.legal_action_mask.sum(dim=1)
                # Normalized entropy = H/log(num legal); 1-legal => 0.
                h_norm = torch.where(nlegal > 1, h / nlegal.clamp_min(2).log(),
                                     torch.zeros_like(h))
                entropy_total += float(h.sum().item())
                entropy_norm_total += float(h_norm.sum().item())
    finally:
        actor.train(old_mode)
    if not all(math.isfinite(x) for x in (entropy_total, entropy_norm_total)):
        raise FloatingPointError('nonfinite actor entropy')
    return {
        'decisions': total,
        'actions': {name: counts[i] for i, name in enumerate(ACTION_LABELS)},
        'legal_opportunities': {name: opportunities[i] for i, name in enumerate(ACTION_LABELS)},
        'conditional_action_rate': {
            name: (round(counts[i] / opportunities[i], 6)
                   if opportunities[i] else None)
            for i, name in enumerate(ACTION_LABELS)
        },
        'mean_policy_entropy': round(entropy_total / total, 6),
        'mean_normalized_entropy': round(entropy_norm_total / total, 6),
        'by_street': by_street,
        'by_position': by_position,
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/action_diagnostics.py" -->

## src/holdem_ai/evaluation/balanced.py

Original SHA `09e9e3b70f5f3d74e143f0a7156ef51480a61ca3` (3092 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/balanced.py" sha="09e9e3b70f5f3d74e143f0a7156ef51480a61ca3" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/balanced.py" -->

## src/holdem_ai/evaluation/benchmark.py

Original SHA `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (0 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/benchmark.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/benchmark.py" -->

## src/holdem_ai/evaluation/checkpoint_loader.py

Original SHA `19053de0286dacb391d1e42eff1d1c5234f20c37` (950 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/checkpoint_loader.py" sha="19053de0286dacb391d1e42eff1d1c5234f20c37" -->

````````python
"""Read trusted Phase-9/10/11 checkpoint or Phase-11 anchor, actor ONLY."""
from __future__ import annotations

from pathlib import Path
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor


def load_actor(path: str | Path, *, device: str = 'cpu') -> TransformerActor:
    # Load only from a trusted local training run; torch weights_only=True.
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'network_config' not in payload or 'actor_state' not in payload:
        raise ValueError('not a Phase9/10/11 checkpoint or Phase11 anchor.pt')
    if payload.get('phase') not in (None, 9, 10):
        raise ValueError('unrecognized training checkpoint phase')
    actor = TransformerActor(NetworkConfig(**payload['network_config']))
    actor.load_state_dict(payload['actor_state'], strict=True)
    return actor.to(device).eval().requires_grad_(False)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/checkpoint_loader.py" -->

## src/holdem_ai/evaluation/continuous_stack.py

Original SHA `cfa61781c80c9acbf0694acd0e716166fd2e0bf7` (14714 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/continuous_stack.py" sha="cfa61781c80c9acbf0694acd0e716166fd2e0bf7" -->

````````python
"""Phase 17: matched-seed, variable-stack HUNL continuous-match evaluation.

This is an EVALUATION-ONLY harness. Phase 6 HUNL200Adapter.reset() and PPO/
Q-Boost training are never modified. A match carries stacks between hands,
rotates the button, and stops at bankruptcy or the hand horizon.

Only each player's InformationState reaches .act(); all other information is
used for post-action accounting, not policy decisions. No unrevealed cards are
included in reports. Seed matching does not create exact duplicate poker.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Protocol

from holdem_ai.core.schema import InformationState, Terminal, Turn
from holdem_ai.game_service.interactive_adapter import InteractiveHUNLAdapter

ACTION_NAMES = (
    "fold", "check", "call", "raise_half_pot", "raise_pot",
    "raise_double_pot", "all_in",
)
STACK_BANDS = ("lt_40", "40_to_99", "100_to_199", "ge_200")


class Policy(Protocol):
    def act(self, observation: InformationState, rng: random.Random) -> int: ...


@dataclass(frozen=True)
class ContinuousEvaluationConfig:
    groups: int = 8
    hands_per_match: int = 30
    candidate_stacks_bb: tuple[int, ...] = (50, 100, 200, 300, 350)
    total_stack_bb: int = 400
    big_blind: int = 100
    small_blind: int = 50
    seed: int = 17_000_000
    max_decisions_per_hand: int = 200

    def __post_init__(self) -> None:
        for name in ("groups", "hands_per_match", "total_stack_bb", "big_blind",
                     "small_blind", "seed", "max_decisions_per_hand"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a valid integer")
        if self.groups < 2:
            raise ValueError("groups >=2 are needed for an uncertainty interval")
        if not 0 < self.small_blind < self.big_blind:
            raise ValueError("require 0 < small_blind < big_blind")
        if self.total_stack_bb != 400:
            # GUI-1.6 adapter fixes the match bank to 2 * 200BB. Reject rather
            # than silently vary an experimental constraint.
            raise ValueError("GUI-1.6 continuous adapter requires total_stack_bb=400")
        if not isinstance(self.candidate_stacks_bb, tuple) or not self.candidate_stacks_bb:
            raise ValueError("candidate_stacks_bb must be a nonempty tuple")
        if any(type(x) is not int or not 0 < x < self.total_stack_bb
               for x in self.candidate_stacks_bb):
            raise ValueError("each candidate stack must be 1..399 BB")
        if len(set(self.candidate_stacks_bb)) != len(self.candidate_stacks_bb):
            raise ValueError("duplicate stack scenarios are not permitted")


def _stack_band(effective_remaining_bb: float) -> str:
    if effective_remaining_bb < 40:
        return STACK_BANDS[0]
    if effective_remaining_bb < 100:
        return STACK_BANDS[1]
    if effective_remaining_bb < 200:
        return STACK_BANDS[2]
    return STACK_BANDS[3]


class _ActionTally:
    """Conditional action rates: numerator and legal-opportunity denominator."""

    def __init__(self) -> None:
        self.legal = {b: [0] * 7 for b in STACK_BANDS}
        self.taken = {b: [0] * 7 for b in STACK_BANDS}

    def record(self, observation: InformationState, index: int, *,
               remaining_stacks: tuple[int, int], big_blind: int) -> None:
        # The hand engine is inspected by the EVALUATOR for analytics only.
        # Never forward it (or hidden cards) to the acting policy.
        stacks = remaining_stacks
        if len(stacks) != 2:
            raise AssertionError("Phase 17 requires heads-up public observations")
        band = _stack_band(min(stacks) / big_blind)
        for action, allowed in enumerate(observation.legal_action_mask):
            if allowed:
                self.legal[band][action] += 1
        self.taken[band][index] += 1

    def merge(self, other: _ActionTally) -> None:
        for band in STACK_BANDS:
            for action in range(7):
                self.legal[band][action] += other.legal[band][action]
                self.taken[band][action] += other.taken[band][action]

    def report(self) -> dict:
        return {
            band: {
                "actions_taken": sum(self.taken[band]),
                "conditional_action_rate": {
                    name: (round(self.taken[band][i] / self.legal[band][i], 6)
                           if self.legal[band][i] else None)
                    for i, name in enumerate(ACTION_NAMES)
                },
                "legal_opportunities": dict(zip(ACTION_NAMES, self.legal[band])),
                "chosen_counts": dict(zip(ACTION_NAMES, self.taken[band])),
            }
            for band in STACK_BANDS
        }


def _group_interval(samples: list[float]) -> dict[str, float]:
    """One independent SHUFFLE SEED GROUP is the statistical replicate."""
    n = len(samples)
    if n < 2 or not all(math.isfinite(x) for x in samples):
        raise ValueError("need two or more finite seed-group estimates")
    mean = math.fsum(samples) / n
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in samples) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    return {
        "mean_bb100": round(mean, 6),
        "ci95_lower_bb100": round(mean - half, 6),
        "ci95_upper_bb100": round(mean + half, 6),
        "std_seed_group_bb100": round(sd, 6),
        "standard_error_bb100": round(sd / math.sqrt(n), 6),
    }


def _play_match(
    candidate: Policy, opponent: Policy, *, cfg: ContinuousEvaluationConfig,
    candidate_seat: int, starting_button: int, candidate_stack_bb: int,
    group_seed: int, tally: _ActionTally,
) -> dict:
    """One match from a fixed endowment; no top-ups, no stack reset."""
    if candidate_seat not in (0, 1) or starting_button not in (0, 1):
        raise ValueError("seat/button must be 0 or 1")
    other = 1 - candidate_seat
    start = [0, 0]
    start[candidate_seat] = candidate_stack_bb * cfg.big_blind
    start[other] = (cfg.total_stack_bb - candidate_stack_bb) * cfg.big_blind
    stacks = tuple(start)
    rng_deck = random.Random(group_seed)
    # Independent streams per policy/seat/layout; *deck seeds*, not action
    # histories, are matched across the four seat/button layouts.
    rng_candidate = random.Random((group_seed << 4) + candidate_seat * 2 + starting_button + 1)
    rng_opponent = random.Random((group_seed << 4) + candidate_seat * 2 + starting_button + 100)
    hands = 0
    showdowns = 0
    hand_wins = 0
    hand_losses = 0
    hand_ties = 0
    decisions = 0
    for hand in range(cfg.hands_per_match):
        if min(stacks) <= 0:
            break
        adapter = InteractiveHUNLAdapter(
            big_blind=cfg.big_blind, small_blind=cfg.small_blind,
            stack_bb=cfg.total_stack_bb // 2,
            button=(starting_button + hand) % 2,
        )
        transition = adapter.reset_with_stacks(stacks, seed=rng_deck.getrandbits(64))
        count = 0
        while isinstance(transition, Turn):
            count += 1
            if count > cfg.max_decisions_per_hand:
                raise RuntimeError("max_decisions_per_hand reached; no truncated payoff was recorded")
            observation = transition.observation
            acting_seat = observation.player_id
            policy = candidate if acting_seat == candidate_seat else opponent
            rng = rng_candidate if acting_seat == candidate_seat else rng_opponent
            action = policy.act(observation, rng)
            # Strictly reject illegal actions rather than auto-repairing.
            observation.assert_legal(action)
            if acting_seat == candidate_seat:
                tally.record(observation, action,
                             remaining_stacks=tuple(adapter.engine.stacks),
                             big_blind=cfg.big_blind)
            transition = adapter.step(action)
            decisions += 1
        if not isinstance(transition, Terminal):
            raise AssertionError("expected a terminal transition")
        result = adapter.engine.get_result()
        updated = tuple(int(x) for x in result.final_stacks)
        if sum(updated) != cfg.total_stack_bb * cfg.big_blind:
            raise AssertionError("chip conservation violated")
        if any(updated[i] - stacks[i] != result.payoffs_chips[i] for i in (0, 1)):
            raise AssertionError("payout does not match start/end stacks")
        payoff = int(result.payoffs_chips[candidate_seat])
        hand_wins += payoff > 0
        hand_losses += payoff < 0
        hand_ties += payoff == 0
        stacks = updated
        hands += 1
        if len(result.board) == 5 and not any(result.folded):
            showdowns += 1
    if not hands:
        raise AssertionError("positive starting stacks must play at least one hand")
    net_bb = (stacks[candidate_seat] - start[candidate_seat]) / cfg.big_blind
    return {
        "hands": hands,
        "net_bb": net_bb,
        "final_candidate_bb": stacks[candidate_seat] / cfg.big_blind,
        "candidate_bust": stacks[candidate_seat] == 0,
        "opponent_bust": stacks[other] == 0,
        "candidate_ahead": net_bb > 0,
        "candidate_behind": net_bb < 0,
        "showdowns": showdowns,
        "hand_wins": hand_wins,
        "hand_losses": hand_losses,
        "hand_ties": hand_ties,
        "decisions": decisions,
    }


def evaluate_continuous_stacks(
    candidate: Policy, opponent: Policy, *,
    config: ContinuousEvaluationConfig | None = None,
    control: Policy | None = None,
    on_group=None,
) -> dict:
    """Compare a candidate (and optional control) under paired shuffle seeds.

    Groups are independent, each four layouts (seat 0/1, BTN 0/1).
    Within one group the bb/100 estimator pools chips and hands across layouts
    before the group-level CI; early-bust matches have fewer hands.
    """
    cfg = config or ContinuousEvaluationConfig()
    if candidate is opponent or (control is not None and control is opponent):
        raise ValueError("use separate policy objects for candidate/control and opponent")
    results = []
    for start_bb in cfg.candidate_stacks_bb:
        primary_tally = _ActionTally()
        control_tally = _ActionTally() if control is not None else None
        primary_blocks: list[dict] = []
        control_blocks: list[dict] = []
        primary_matches: list[dict] = []
        control_matches: list[dict] = []
        for group_idx in range(cfg.groups):
            group_seed = cfg.seed + group_idx
            for who, tally, matches, blocks in (
                (candidate, primary_tally, primary_matches, primary_blocks),
                (control, control_tally, control_matches, control_blocks),
            ):
                if who is None:
                    continue
                group_matches = [
                    _play_match(
                        who, opponent, cfg=cfg, candidate_seat=seat,
                        starting_button=button, candidate_stack_bb=start_bb,
                        group_seed=group_seed, tally=tally,
                    )
                    for button in (0, 1) for seat in (0, 1)
                ]
                matches.extend(group_matches)
                total_hands = sum(m["hands"] for m in group_matches)
                net = math.fsum(m["net_bb"] for m in group_matches)
                blocks.append({
                    "seed": group_seed,
                    "hands": total_hands,
                    "net_bb": round(net, 6),
                    "bb100": round(100.0 * net / total_hands, 12),
                })
            if on_group is not None:
                on_group(start_bb, group_idx + 1, cfg.groups)

        def summarize(blocks: list[dict], matches: list[dict], tally: _ActionTally) -> dict:
            return {
                **_group_interval([b["bb100"] for b in blocks]),
                "groups": cfg.groups,
                "matches": len(matches),
                "hands": sum(x["hands"] for x in matches),
                "mean_net_bb_per_match": round(math.fsum(x["net_bb"] for x in matches) / len(matches), 6),
                "mean_hands_per_match": round(math.fsum(x["hands"] for x in matches) / len(matches), 6),
                "match_profit_fraction": round(sum(x["candidate_ahead"] for x in matches) / len(matches), 6),
                "candidate_bust_fraction": round(sum(x["candidate_bust"] for x in matches) / len(matches), 6),
                "opponent_bust_fraction": round(sum(x["opponent_bust"] for x in matches) / len(matches), 6),
                "hand_win_fraction": round(sum(x["hand_wins"] for x in matches) / sum(x["hands"] for x in matches), 6),
                "hand_loss_fraction": round(sum(x["hand_losses"] for x in matches) / sum(x["hands"] for x in matches), 6),
                "hand_tie_fraction": round(sum(x["hand_ties"] for x in matches) / sum(x["hands"] for x in matches), 6),
                "showdown_fraction": round(sum(x["showdowns"] for x in matches) / sum(x["hands"] for x in matches), 6),
                "candidate_action_profile_by_effective_remaining_stack": tally.report(),
                "seed_groups": blocks,
            }

        scenario = {
            "candidate_start_bb": start_bb,
            "opponent_start_bb": cfg.total_stack_bb - start_bb,
            "candidate": summarize(primary_blocks, primary_matches, primary_tally),
        }
        if control is not None:
            assert control_tally is not None
            scenario["control"] = summarize(control_blocks, control_matches, control_tally)
            paired_deltas = [a["bb100"] - b["bb100"]
                             for a, b in zip(primary_blocks, control_blocks)]
            scenario["paired_delta"] = {
                **_group_interval(paired_deltas),
                "group_deltas_bb100": [round(v, 6) for v in paired_deltas],
            }
        results.append(scenario)
    return {
        "phase": 17,
        "protocol": "continuous_400bb_four_layout_seed_group",
        "config": {**asdict(cfg), "candidate_stacks_bb": list(cfg.candidate_stacks_bb)},
        "scenarios": results,
        "interpretation": (
            "Per-seed four-layout paired design; within-group deck reuse does NOT "
            "guarantee exact duplicate poker. Approximate normal 95% intervals on "
            "seed-group bb/100, not GTO exploitability. Early bankruptcies "
            "shorten match lengths. This is continuous match evaluation, NOT "
            "Phase16 independent 200BB-hand evaluation."
        ),
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/continuous_stack.py" -->

## src/holdem_ai/evaluation/critic_diagnostics.py

Original SHA `1edc6be08bc4d8849ff149c363c0c075df8d330d` (11696 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/critic_diagnostics.py" sha="1edc6be08bc4d8849ff149c363c0c075df8d330d" -->

````````python
"""Read-only HUNL PPO critic calibration diagnostics.

Q_i(s,a_taken) is supervised on realized terminal returns, while PPO uses
V_i(s) = sum_a pi(a|I_i) Q_i(s,a) as an action-independent baseline.

On-policy rollout metrics are the main calibration signal. The *shared*
reference-rollout audit controls the observations/actions/targets, but its
targets belong to the reference continuation policy, not necessarily to a
candidate critic's continuation policy: interpret those numbers as
cross-policy generalization, NEVER as direct proof of critic failure.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.models.transformer import (CentralizedQCritic, NetworkConfig,
                                           TransformerActor, chosen_q)
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.training.ppo import PPOConfig, TrajectoryRows

STREETS = ("preflop", "flop", "turn", "river")
ACTIONS = ("fold", "check", "call", "raise_half_pot", "raise_pot",
           "raise_double_pot", "all_in")


@dataclass(frozen=True)
class CriticCheckpoint:
    actor: TransformerActor
    critic: CentralizedQCritic
    network_config: NetworkConfig
    ppo_config: PPOConfig
    completed_updates: int


def load_ppo_critic_checkpoint(path: str | Path, *, device: str = "cpu") -> CriticCheckpoint:
    """Read trusted Phase-9 PPO format without loading optimizer or modifying files."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict)
            or payload.get("phase") != 9
            or payload.get("algorithm") != "ppo_mc_privileged_baseline"):
        raise ValueError("expected a trusted Phase-9 PPO checkpoint")
    net = NetworkConfig(**payload["network_config"])
    conf = PPOConfig(**payload["ppo_config"])
    actor = TransformerActor(net)
    critic = CentralizedQCritic(net)
    actor.load_state_dict(payload["actor_state"], strict=True)
    critic.load_state_dict(payload["critic_state"], strict=True)
    actor.to(device).eval().requires_grad_(False)
    critic.to(device).eval().requires_grad_(False)
    return CriticCheckpoint(actor, critic, net, conf, int(payload["completed_updates"]))


def regression_metrics(targets: list[float], predictions: list[float]) -> dict:
    """Population explained variance, bias (pred - target), MAE and RMSE.

    EV is undefined if the target variance is zero. Repeated decisions from
    one hand are correlated: no iid significance intervals are claimed.
    """
    if not targets or len(targets) != len(predictions):
        raise ValueError("expected equal nonempty vectors")
    if not all(math.isfinite(x) for x in (*targets, *predictions)):
        raise ValueError("nonfinite critic targets/predictions")
    n = len(targets)
    errors = [p - y for y, p in zip(targets, predictions)]
    bias = math.fsum(errors) / n
    mae = math.fsum(abs(e) for e in errors) / n
    mse = math.fsum(e * e for e in errors) / n
    ymean = math.fsum(targets) / n
    yvar = math.fsum((y - ymean) ** 2 for y in targets) / n
    evar = math.fsum((e - bias) ** 2 for e in errors) / n
    pmean = math.fsum(predictions) / n
    pvar = math.fsum((p - pmean) ** 2 for p in predictions) / n
    return {
        "decisions": n,
        "bias_bb": round(bias, 6),
        "mae_bb": round(mae, 6),
        "rmse_bb": round(math.sqrt(mse), 6),
        "explained_variance": round(1 - evar / yvar, 6) if yvar > 1e-12 else None,
        "target_mean_bb": round(ymean, 6),
        "target_std_bb": round(math.sqrt(yvar), 6),
        "prediction_mean_bb": round(pmean, 6),
        "prediction_std_bb": round(math.sqrt(pvar), 6),
    }


def _group_metrics(rows: TrajectoryRows, targets: list[float],
                   predictions: list[float]) -> dict:
    indices = defaultdict(list)
    for i, (obs, action, seat) in enumerate(
            zip(rows.actors, rows.actions, rows.acting_seats)):
        street = int(obs.features[2])
        if street not in range(4) or float(street) != obs.features[2]:
            raise ValueError("invalid street encoding")
        if obs.features[1] not in (0.0, 1.0):
            raise ValueError("invalid position encoding")
        keys = ("overall", f"street:{STREETS[street]}", f"seat:{seat}",
                f"position:{'button' if obs.features[1] == 1 else 'big_blind'}",
                f"action:{ACTIONS[action]}")
        for key in keys:
            indices[key].append(i)
    return {
        key: regression_metrics([targets[i] for i in sel],
                                [predictions[i] for i in sel])
        for key, sel in sorted(indices.items())
    }


def diagnose_critic_on_rows(rows: TrajectoryRows, reference_actor: TransformerActor,
                            critic: CentralizedQCritic, *, device: str = "cpu",
                            chunk_size: int = 64) -> dict:
    """Return Q(chosen action) and reference-policy V calibration on given rows.

    The reference actor MUST be the behavior policy that generated the rows
    for the V baseline's return error to have the intended interpretation.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    old_actor, old_critic = reference_actor.training, critic.training
    q_selected, v_ref = [], []
    try:
        reference_actor.eval()
        critic.eval()
        with torch.inference_mode():
            for start in range(0, len(rows.actions), chunk_size):
                stop = min(start + chunk_size, len(rows.actions))
                ab = batch_actors(rows.actors[start:stop], device=device)
                cb = batch_critics(rows.critics[start:stop], device=device)
                actions = torch.tensor(rows.actions[start:stop], device=device, dtype=torch.long)
                seats = torch.tensor(rows.acting_seats[start:stop], device=device, dtype=torch.long)
                q = critic(cb)
                if q.shape != (stop - start, 2, 7):
                    raise ValueError("critic must output [batch,2,7]")
                dist = reference_actor.distribution(ab)
                if dist.probs.shape != (stop - start, 7):
                    raise ValueError("actor policy must output [batch,7]")
                q_taken = chosen_q(q, seats, actions)
                selected = q[torch.arange(stop - start, device=device), seats, :]
                baseline = (selected * dist.probs).sum(dim=1)
                q_selected.extend(q_taken.cpu().tolist())
                v_ref.extend(baseline.cpu().tolist())
    finally:
        reference_actor.train(old_actor)
        critic.train(old_critic)
    targets = list(rows.returns_bb)
    if any(not math.isfinite(x) for x in (*q_selected, *v_ref)):
        raise FloatingPointError("nonfinite critic predictions")
    adv = [y - b for y, b in zip(targets, v_ref)]
    adv_mean = math.fsum(adv) / len(adv)
    adv_std = math.sqrt(math.fsum((a - adv_mean) ** 2 for a in adv) / len(adv))
    return {
        "hands": rows.hands,
        "decisions": len(rows.actions),
        "q_taken": _group_metrics(rows, targets, q_selected),
        "v_reference_policy": _group_metrics(rows, targets, v_ref),
        "raw_advantage": {
            "mean_bb": round(adv_mean, 6),
            "std_bb": round(adv_std, 6),
            "mean_abs_bb": round(math.fsum(abs(a) for a in adv) / len(adv), 6),
        },
        "note": "Q selected-action errors and action-independent V baseline "
                "errors are distinct; observations within a hand are correlated.",
    }


def run_critic_audit(*, reference_checkpoint: str | Path,
                     before_checkpoint: str | Path, after_checkpoint: str | Path,
                     device: str = "cpu", hands: int = 256, on_policy_hands: int = 128,
                     seed: int = 570_000_000, env_batch_size: int = 32,
                     chunk_size: int = 64) -> dict:
    """One shared independent reference rollout plus two own-policy rollouts.

    Shared errors are affected by continuation-policy mismatch. Each own-policy
    audit uses actual sampled terminal returns from the respective policy.
    """
    if (type(hands) is not int or hands < 1 or type(on_policy_hands) is not int
            or on_policy_hands < 1 or type(seed) is not int or seed < 0
            or type(env_batch_size) is not int or env_batch_size < 1):
        raise ValueError("invalid rollout configuration")
    source = load_ppo_critic_checkpoint(reference_checkpoint, device=device)
    models = {
        "before": load_ppo_critic_checkpoint(before_checkpoint, device=device),
        "after": load_ppo_critic_checkpoint(after_checkpoint, device=device),
    }
    for label, model in models.items():
        if (model.network_config != source.network_config
                or model.ppo_config.big_blind != source.ppo_config.big_blind):
            raise ValueError(f"{label} checkpoint network/big blind differs from reference")
    shared_cfg = replace(source.ppo_config, seed=seed, hands_per_update=hands)
    shared = collect_batched_self_play(source.actor, shared_cfg,
                                      iteration=0, device=device,
                                      env_batch_size=min(env_batch_size, hands))
    common = {
        label: diagnose_critic_on_rows(shared, source.actor, model.critic,
                                       device=device, chunk_size=chunk_size)
        for label, model in models.items()
    }
    own = {}
    for label, model in models.items():
        own_cfg = replace(model.ppo_config, seed=seed + 1_000_000,
                          hands_per_update=on_policy_hands)
        own_rows = collect_batched_self_play(model.actor, own_cfg,
                                            iteration=0, device=device,
                                            env_batch_size=min(env_batch_size, on_policy_hands))
        own[label] = diagnose_critic_on_rows(own_rows, model.actor, model.critic,
                                             device=device, chunk_size=chunk_size)
    return {
        "protocol": "phase15_critic_calibration_read_only",
        "checkpoints": {
            "reference": str(reference_checkpoint),
            "before": str(before_checkpoint),
            "after": str(after_checkpoint),
        },
        "completed_updates": {label: model.completed_updates
                              for label, model in models.items()},
        "shared_reference": {
            "seed": seed, "hands": hands, "decisions": len(shared.actions),
            "results": common,
            "limitation": "The sampled action/return labels come from the reference "
                          "continuation policy; candidate Q(s,a) can legitimately "
                          "differ if its continuation strategy differs. This is "
                          "a paired cross-policy probe, NOT isolated critic accuracy.",
        },
        "on_policy": {
            "seed": seed + 1_000_000,
            "hands_per_model": on_policy_hands,
            "results": own,
            "limitation": "On-policy targets suit each critic's policy, but model "
                          "rollout distributions differ: compare with caution.",
        },
        "limitations": "This does not measure exploitability or establish causal "
                       "impact of critic errors on validation scores. Sparse "
                       "postflop state counts may limit per-street conclusions.",
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/critic_diagnostics.py" -->

## src/holdem_ai/evaluation/fixed_state_policy_audit.py

Original SHA `78429cbfde96d5995f7e3d71f369dde1e03069ed` (6773 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/fixed_state_policy_audit.py" sha="78429cbfde96d5995f7e3d71f369dde1e03069ed" -->

````````python
"""Evaluate frozen HUNL actor checkpoints on exactly the same actor-visible states.

Rows are sampled ONCE from a reference checkpoint; candidate checkpoints never
generate their own evaluation trajectories. No CriticState or hidden cards are
used in the policy comparisons.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from holdem_ai.encoding.tensors import batch_actors
from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.training.ppo import PPOConfig

ACTION_NAMES = ("fold", "check", "call", "raise_half_pot", "raise_pot",
                "raise_double_pot", "all_in")
STREETS = ("preflop", "flop", "turn", "river")


def _new_bucket():
    return {"states": 0, "fold_legal": 0, "fold_prob_sum": 0.0,
            "entropy_sum": 0.0, "normalized_entropy_sum": 0.0,
            "multi_legal": 0, "action_prob_sums": [0.0] * 7,
            "action_legal_counts": [0] * 7, "legal_action_counts": defaultdict(int)}


def _summarize(bucket):
    n = bucket["states"]
    f = bucket["fold_legal"]
    m = bucket["multi_legal"]
    return {
        "states": n,
        "fold_legal_states": f,
        "mean_fold_probability_given_legal": round(bucket["fold_prob_sum"] / f, 6) if f else None,
        "mean_entropy": round(bucket["entropy_sum"] / n, 6) if n else None,
        "mean_normalized_entropy_for_multi_legal": round(bucket["normalized_entropy_sum"] / m, 6) if m else None,
        "multi_legal_states": m,
        "mean_action_probability_given_legal": {
            name: (round(bucket["action_prob_sums"][i] / bucket["action_legal_counts"][i], 6)
                   if bucket["action_legal_counts"][i] else None)
            for i, name in enumerate(ACTION_NAMES)
        },
        "legal_action_counts": dict(sorted(bucket["legal_action_counts"].items())),
    }


def compare_actors_on_states(actors: dict, actor_rows: tuple, *,
                             device: str = "cpu", chunk_size: int = 64) -> dict:
    """Compare policies on a shared collection of EncodedActor observations.

    Each model gets the same samples/denominators (conditional on action
    legality); action rates are *expected probabilities*, not sampled choices.
    """
    if not actor_rows:
        raise ValueError("actor_rows must not be empty")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    output = {}
    for label, actor in actors.items():
        stats = defaultdict(_new_bucket)
        was_training = actor.training
        try:
            actor.eval()
            with torch.inference_mode():
                for start in range(0, len(actor_rows), chunk_size):
                    part = actor_rows[start:start + chunk_size]
                    batch = batch_actors(part, device=device)
                    dist = actor.distribution(batch)
                    probs = dist.probs.detach().cpu()
                    entropies = dist.entropy().detach().cpu()
                    legal = batch.legal_action_mask.detach().cpu()
                    for i, row in enumerate(part):
                        st = STREETS[int(row.features[2])]
                        pos = "button" if row.features[1] == 1.0 else "big_blind"
                        k = int(legal[i].sum().item())
                        if not math.isfinite(float(entropies[i])):
                            raise FloatingPointError("nonfinite entropy")
                        h = float(entropies[i])
                        norm = h / math.log(k) if k > 1 else None
                        for key in ("overall", f"street:{st}", f"position:{pos}", f"street_position:{st}:{pos}"):
                            bucket = stats[key]
                            bucket["states"] += 1
                            bucket["entropy_sum"] += h
                            bucket["legal_action_counts"][k] += 1
                            if norm is not None:
                                bucket["multi_legal"] += 1
                                bucket["normalized_entropy_sum"] += norm
                            if bool(legal[i, 0]):
                                bucket["fold_legal"] += 1
                                bucket["fold_prob_sum"] += float(probs[i, 0])
                            for j in range(7):
                                if bool(legal[i, j]):
                                    bucket["action_legal_counts"][j] += 1
                                    bucket["action_prob_sums"][j] += float(probs[i, j])
        finally:
            actor.train(was_training)
        output[label] = {key: _summarize(bucket) for key, bucket in sorted(stats.items())}
    return output


def compare_checkpoints(checkpoints: dict[str, str | Path], *,
                        reference_checkpoint: str | Path,
                        device: str = "cpu", hands: int = 128,
                        seed: int = 550_000_000, env_batch_size: int = 32) -> dict:
    """Produce an identical-observation policy audit from trusted local files."""
    if len(checkpoints) < 2:
        raise ValueError("at least two checkpoints required")
    if hands < 1 or env_batch_size < 1 or env_batch_size > hands or seed < 0:
        raise ValueError("invalid reference sampling settings")
    reference = load_actor(reference_checkpoint, device=device)
    # This probe doesn't optimize or mutate any checkpoint. Both seats play
    # the same reference actor to produce fixed evaluation observations.
    cfg = PPOConfig(hands_per_update=hands, seed=seed)
    rows = collect_batched_self_play(reference, cfg, iteration=0, device=device,
                                     env_batch_size=env_batch_size)
    models = {label: load_actor(path, device=device) for label, path in checkpoints.items()}
    summaries = compare_actors_on_states(models, rows.actors, device=device)
    names = list(checkpoints)
    a = summaries[names[0]]["overall"]["mean_fold_probability_given_legal"]
    b = summaries[names[1]]["overall"]["mean_fold_probability_given_legal"]
    return {
        "protocol": "fixed_reference_actor_observations",
        "reference_checkpoint": str(reference_checkpoint),
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "hands": hands, "states": len(rows.actors), "seed": seed,
        "device": device, "models": summaries,
        "comparison": {
            "from": names[0], "to": names[1],
            "delta_conditional_fold_probability": round(b - a, 6) if a is not None and b is not None else None,
        },
        "warning": "On-reference-policy states only; not an EV or exploitability test.",
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/fixed_state_policy_audit.py" -->

## src/holdem_ai/evaluation/initial_policy_comparison.py

Original SHA `00479ac04ed08062e09bdb3f2bf3d58e902f2cd9` (9620 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/initial_policy_comparison.py" sha="00479ac04ed08062e09bdb3f2bf3d58e902f2cd9" -->

````````python
"""Read-only Agent comparison: uniform legal, random-init, trained Phase14.

Two complementary diagnostics:
  (1) on-policy hand and *per-legal-decision* action frequencies;
  (2) same reference-visited observations for all three policy probabilities.

Both seats play the SAME policy in each rollout, as in current PPO self-play.
This probes an action-abstraction / initial-policy / training hypothesis;
it is not a poker strength or causal exploitability comparison.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch import nn

from holdem_ai.encoding.tensors import ActorBatch
from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.evaluation.fixed_state_policy_audit import compare_actors_on_states
from holdem_ai.evaluation.terminal_reward_audit import (
    ACTION_NAMES, STREETS, summarize_terminal_distribution,
)
from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.training.ppo import PPOConfig, TrajectoryRows


class UniformLegalActor(nn.Module):
    """Uniform probability over CURRENTLY LEGAL policy-head actions.

    Adapter implements the existing batched collector Actor API but never
    consumes hidden/privileged critic observations and has no trainable weights.
    """
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config

    def distribution(self, batch: ActorBatch) -> torch.distributions.Categorical:
        if not isinstance(batch, ActorBatch):
            raise TypeError("uniform actor accepts ActorBatch only")
        mask = batch.legal_action_mask
        if mask.ndim != 2 or mask.shape[1] != len(ACTION_NAMES):
            raise ValueError("legal action mask must be [batch,7]")
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError("every decision requires at least one legal action")
        logits = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
        return torch.distributions.Categorical(
            logits=logits.masked_fill(~mask, float("-inf"))
        )


def build_untrained_actor(config: NetworkConfig, *, seed: int,
                          device: str = "cpu") -> TransformerActor:
    """Fresh deterministic random initialization, NOT an actual initial ckpt.

    If a verified initial checkpoint exists, pass --initial-checkpoint instead.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("initialization seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TransformerActor(config)
    return model.to(device).eval().requires_grad_(False)


def _action_counts_on_rows(rows: TrajectoryRows) -> dict:
    """Count decisions conditional on an action being LEGAL, not all decisions.

    Explicit all-in means sampled action head index 6; call-to-all-in is not
    classified here. Frequency estimates are sampled, not fixed-state probs.
    """
    if not rows.decisions_per_hand or sum(rows.decisions_per_hand) != len(rows.actions):
        raise ValueError("must preserve complete per-hand decision boundaries")
    street_data = {}
    for street in ("overall", *STREETS):
        selected = [
            i for i, obs in enumerate(rows.actors)
            if street == "overall" or int(obs.features[2]) == STREETS.index(street)
        ]
        action_count = Counter(rows.actions[i] for i in selected)
        legal_counts = {
            name: sum(bool(rows.actors[i].legal_action_mask[j]) for i in selected)
            for j, name in enumerate(ACTION_NAMES)
        }
        street_data[street] = {
            "decisions": len(selected),
            "action_counts": {
                name: action_count[j] for j, name in enumerate(ACTION_NAMES)
            },
            "legal_opportunities": legal_counts,
            "all_in_legal_decisions": legal_counts["all_in"],
            "all_in_selected_decisions": action_count[6],
            "all_in_rate_given_legal": (
                round(action_count[6] / legal_counts["all_in"], 6)
                if legal_counts["all_in"] else None
            ),
            "all_in_rate_over_all_decisions": (
                round(action_count[6] / len(selected), 6) if selected else None
            ),
        }
    # Hand-level presence and first-decision rate are different from action rates.
    first = []
    offset = 0
    explicit_all_in_hands = 0
    for length in rows.decisions_per_hand:
        if length <= 0:
            raise ValueError("a finished hand must have at least one decision")
        obs = rows.actors[offset]
        first.append((rows.actions[offset], bool(obs.legal_action_mask[6])))
        if 6 in rows.actions[offset:offset + length]:
            explicit_all_in_hands += 1
        offset += length
    if offset != len(rows.actions):
        raise ValueError("hand counts do not align with decisions")
    first_legal = sum(allowed for _, allowed in first)
    first_all_in = sum(action == 6 for action, _ in first)
    street_data["per_hand"] = {
        "hands": rows.hands,
        "hands_with_explicit_all_in": explicit_all_in_hands,
        "hands_with_explicit_all_in_fraction": round(explicit_all_in_hands / rows.hands, 6),
        "first_decision_all_in": first_all_in,
        "first_decision_all_in_legal": first_legal,
        "first_decision_all_in_given_legal": (
            round(first_all_in / first_legal, 6) if first_legal else None
        ),
    }
    return street_data


def compare_three_agent_policies(*, trained_checkpoint: str | Path,
                                  initial_checkpoint: str | Path | None = None,
                                  init_seed: int = 52,
                                  device: str = "cpu", hands: int = 512,
                                  seed: int = 590_000_000,
                                  env_batch_size: int = 32) -> dict:
    """Collect on-policy rollouts once per model with shared deck-seed block.

    The trained Phase14 actor is the reference for the fixed-observation audit.
    Distinct decision sequences mean same card seeds do NOT ensure same games.
    """
    if (type(hands) is not int or hands < 1 or type(seed) is not int
            or seed < 0 or type(env_batch_size) is not int
            or not 1 <= env_batch_size <= hands):
        raise ValueError("invalid hands/seed/env_batch_size")
    trained = load_actor(trained_checkpoint, device=device)
    net = trained.config
    if net.dropout != 0:
        raise ValueError("diagnostic requires dropout=0")
    if initial_checkpoint is not None:
        init = load_actor(initial_checkpoint, device=device)
        if init.config != net:
            raise ValueError("initial checkpoint architecture differs from trained checkpoint")
        initial_type = "user_supplied_checkpoint_not_verified_untrained"
    else:
        init = build_untrained_actor(net, seed=init_seed, device=device)
        initial_type = "synthetic_random_initialization_NOT_verified_phase14_start"
    policies = {
        "uniform_legal": UniformLegalActor(net).to(device).eval(),
        "random_init": init,
        "trained_phase14": trained,
    }
    rollout_config = PPOConfig(
        seed=seed, hands_per_update=hands, max_decisions=200,
    )
    summaries = {}
    reference_actor_states = None
    for label, actor in policies.items():
        rows = collect_batched_self_play(
            actor, rollout_config, device=device, iteration=0,
            env_batch_size=env_batch_size,
        )
        summaries[label] = {
            "terminal": summarize_terminal_distribution(rows),
            "actions": _action_counts_on_rows(rows),
        }
        if label == "trained_phase14":
            reference_actor_states = rows.actors
    assert reference_actor_states is not None
    fixed = compare_actors_on_states(
        policies, reference_actor_states, device=device,
    )
    return {
        "protocol": "phase15_uniform_vs_init_vs_phase14_read_only",
        "trained_checkpoint": str(trained_checkpoint),
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint is not None else None,
        "initial_policy_provenance": initial_type,
        "init_seed": init_seed if initial_checkpoint is None else None,
        "hands_per_policy": hands,
        "deck_seed_start": seed,
        "device": device,
        "on_policy": summaries,
        "fixed_phase14_states": {
            "states": len(reference_actor_states),
            "results": fixed,
        },
        "limits": [
            "Uniform policy selects equally among currently LEGAL action indices; "
            "the betting abstraction can supply different numbers of legal actions.",
            "Random-init Actor is freshly initialized using a known seed and "
            "is NOT verified to equal the real Phase14 training start unless "
            "an exact initial checkpoint is supplied.",
            "Each policy plays both seats against itself. A probability "
            "of any all-in per hand is NOT the probability of all-in per decision.",
            "Same deck seed block does not force equal action histories. "
            "The fixed-state section compares all three policies on identical "
            "Phase14-visited actor-visible observations (no hidden cards).",
            "No poker EV, exploitability, statistical significance or causal "
            "effect of optimization is established here.",
        ],
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/initial_policy_comparison.py" -->

## src/holdem_ai/evaluation/opponent_pool.py

Original SHA `78369a6dfad4b548a397dcf6ec9e6e6e8a890ec5` (6852 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/opponent_pool.py" sha="78369a6dfad4b548a397dcf6ec9e6e6e8a890ec5" -->

````````python
"""Phase16 diagnostic opponent-pool benchmark, independent of Phase15 training.

For every shuffle seed, the four layouts are replayed for every opponent.
One seed group (NOT one hand) is the statistical replicate.
The fixed opponent pool is a benchmark, not GTO exploitability.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from holdem_ai.core.schema import InformationState
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode
from holdem_ai.agents.neural_policy import NeuralActorPolicy
from holdem_ai.models.transformer import TransformerActor
from holdem_ai.evaluation.seed_blocked import model_fingerprint, summarize_block_scores
from holdem_ai.evaluation.checkpoint_loader import load_actor

from holdem_ai.evaluation.strategy_diagnostics import ActionProfile
from holdem_ai.agents.diagnostic_opponents import OpponentSpec, build_builtin


@dataclass(frozen=True)
class PoolEvaluationConfig:
    groups: int = 40
    seed: int = 16_000_000
    mode: str = 'sampled'
    device: str = 'cpu'
    big_blind: int = 100
    max_decisions: int = 200
    entropy_batch_size: int = 64

    def __post_init__(self):
        if type(self.groups) is not int or self.groups < 2:
            raise ValueError('groups must be >=2')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be nonnegative int')
        if self.mode not in ('sampled', 'greedy'):
            raise ValueError('mode must be sampled or greedy')
        if self.device != 'cpu' and not self.device.startswith('cuda'):
            raise ValueError('invalid device')
        for name in ('big_blind', 'max_decisions', 'entropy_batch_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive int')
        if self.big_blind < 2:
            raise ValueError('HUNL200 requires big_blind >= 2 to set a positive smaller small blind')


class _TrackedCandidate:
    def __init__(self, neural_policy: NeuralActorPolicy, profile: ActionProfile):
        self.policy = neural_policy
        self.profile = profile

    def act(self, obs: InformationState, rng) -> int:
        action = self.policy.act(obs, rng)
        self.profile.record(obs, action)
        return action


def _opponent_policy(spec: OpponentSpec, config: PoolEvaluationConfig):
    if spec.checkpoint is None:
        return build_builtin(spec.name), 'builtin:' + spec.name, None
    model = load_actor(spec.checkpoint, device=config.device)
    if model.config.dropout:
        raise ValueError('opponent dropout must be zero during evaluation')
    policy = NeuralActorPolicy(model, device=config.device,
                               big_blind=config.big_blind, greedy=config.mode == 'greedy')
    return policy, model_fingerprint(model), model


def evaluate_opponent_pool(candidate: TransformerActor, opponents: tuple[OpponentSpec, ...],
                           *, config: PoolEvaluationConfig | None = None,
                           on_group: Callable[[str], None] | None = None) -> dict:
    """Returns JSON-safe summary without any hole cards, opponent cards or samples."""
    cfg = config or PoolEvaluationConfig()
    if not opponents or len({x.name for x in opponents}) != len(opponents):
        raise ValueError('provide a nonempty pool with unique opponent names')
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if candidate.config.dropout:
        raise ValueError('candidate dropout must be zero during evaluation')
    identity = model_fingerprint(candidate)
    initial_mode = candidate.training
    results = []
    try:
        actor_policy = NeuralActorPolicy(candidate, device=cfg.device,
                                        big_blind=cfg.big_blind, greedy=cfg.mode == 'greedy')
        for spec in opponents:
            other, other_id, other_model = _opponent_policy(spec, cfg)
            action_profile = ActionProfile()
            wrapper = _TrackedCandidate(actor_policy, action_profile)
            blocks: list[float] = []
            for group in range(cfg.groups):
                seed = cfg.seed + group
                payoffs = []
                for button in (0, 1):
                    for candidate_seat in (0, 1):
                        adapter = HUNL200Adapter(button=button, big_blind=cfg.big_blind,
                                                 small_blind=max(1, cfg.big_blind // 2))
                        episode = collect_global_episode(
                            adapter, {candidate_seat: wrapper, 1-candidate_seat: other},
                            seed=seed, privileged=False, max_decisions=cfg.max_decisions,
                        )
                        if len(episode.terminal_rewards) != 2 or not math.isclose(
                                math.fsum(episode.terminal_rewards), 0.0, abs_tol=1e-7):
                            raise AssertionError('nonzero-sum HUNL game payoff')
                        payoffs.append(float(episode.terminal_rewards[candidate_seat]))
                blocks.append(math.fsum(payoffs) / 4)
                if on_group is not None:
                    on_group(spec.name)
            result = {
                'opponent': spec.name, 'opponent_id': other_id,
                **summarize_block_scores(tuple(blocks)),
                'groups': cfg.groups, 'hands': cfg.groups * 4,
                'candidate_action_profile': action_profile.summary(
                    candidate, device=cfg.device, big_blind=cfg.big_blind,
                    entropy_batch_size=cfg.entropy_batch_size,
                ),
            }
            results.append(result)
            del other_model
        if model_fingerprint(candidate) != identity:
            raise AssertionError('evaluation unexpectedly mutated candidate weights')
    finally:
        candidate.train(initial_mode)
    scores = [float(x['mean_bb100']) for x in results]
    return {
        'phase': 16,
        'protocol': 'phase16_fixed_opponent_pool_four_layout',
        'candidate_id': identity,
        'config': asdict(cfg),
        'opponents': results,
        'pool_summary': {
            'opponent_count': len(results),
            'total_hands': len(results) * cfg.groups * 4,
            'equal_weight_mean_bb100': round(math.fsum(scores) / len(scores), 4),
            'worst_opponent': results[min(range(len(scores)), key=scores.__getitem__)]['opponent'],
            'worst_observed_mean_bb100': round(min(scores), 4),
        },
        'interpretation': 'Fixed-pool empirical benchmark, not GTO exploitability or robust game-theoretic guarantee.',
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/opponent_pool.py" -->

## src/holdem_ai/evaluation/ppo_metric_audit.py

Original SHA `52428f68b0ad1d10917836daca2af57032c8ac9c` (1906 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/ppo_metric_audit.py" sha="52428f68b0ad1d10917836daca2af57032c8ac9c" -->

````````python
"""Read-only Phase 15 PPO optimizer diagnostics from existing JSONL logs."""
from __future__ import annotations

import json
import math
from pathlib import Path

FIELDS = ("entropy", "approx_kl", "clip_fraction", "grad_norm_actor",
          "grad_norm_critic", "actor_loss", "critic_loss")


def audit_metrics(path: str | Path) -> dict:
    updates = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") != "update" or record.get("algorithm") != "ppo":
            continue
        values = {}
        for name in FIELDS:
            value = record.get(name)
            if value is not None:
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"nonfinite {name} at line {number}")
                values[name] = float(value)
        updates.append({"update": record["update"], **values})
    if not updates:
        raise ValueError("no PPO update metrics found")
    out = {}
    for field in FIELDS:
        values = [row[field] for row in updates if field in row]
        out[field] = ({
            "observations": len(values),
            "first": values[0], "last": values[-1],
            "min": min(values), "max": max(values),
            "mean": round(sum(values) / len(values), 7),
        } if values else None)
    return {
        "updates": len(updates), "first_update": updates[0]["update"],
        "last_update": updates[-1]["update"], "fields": out,
        "last_20_updates": updates[-20:],
        "note": "Metrics are optimizer-minibatch aggregates, not fixed-state statistics. "
                "Learning rate/entropy coefficient come from run.json ppo_config; "
                "these summaries alone cannot identify a causal collapse mechanism.",
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/ppo_metric_audit.py" -->

## src/holdem_ai/evaluation/preflop_sequence_audit.py

Original SHA `ff7685384d1d7d88afdb4237011316e474fe8274` (11220 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/preflop_sequence_audit.py" sha="ff7685384d1d7d88afdb4237011316e474fe8274" -->

````````python
"""Read-only HUNL agent preflop decision-sequence diagnostics.

Uses the existing per-hand TrajectoryRows; never inspects cards, critic
privilege, future deck, or mutable environment internals. Classifies sampled
policy HEAD actions (not exact wager amounts or formal bet/3-bet sizing).
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.evaluation.initial_policy_comparison import (
    UniformLegalActor, build_untrained_actor,
)
from holdem_ai.evaluation.terminal_reward_audit import ACTION_NAMES
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.training.ppo import PPOConfig, TrajectoryRows

RAISE_INDICES = frozenset((3, 4, 5, 6))


def _top_counts(counter: Counter, *, limit: int = 15) -> list[dict]:
    return [{"sequence": key, "count": value} for key, value
            in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]


def _summary(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values), "mean": round(math.fsum(values) / len(values), 6),
        "min": min(values), "max": max(values),
    }


def summarize_preflop_sequences(rows: TrajectoryRows) -> dict:
    """Reconstruct ordered sampled actions and opportunity-based rates.

    Decisions are hand-aligned; within each hand actor.features[2] encodes
    the street, actor.features[1] encodes button position, actor.features[4]
    is the *public* amount to call (BB). No showdown cards are accessed.
    """
    if (not rows.decisions_per_hand or not rows.terminal_actions
            or len(rows.decisions_per_hand) != rows.hands
            or sum(rows.decisions_per_hand) != len(rows.actions)):
        raise ValueError("complete hand action boundaries required")
    if not rows.actions:
        raise ValueError("no actor decisions")

    first_action = Counter()
    final_preflop_action = Counter()
    complete_preflop = Counter()
    prefix_to_allin = Counter()
    before_allin = Counter()
    first_allin_step = Counter()
    preflop_decision_lengths = []
    allin_preflop_decision_lengths = []
    fold_preflop_decision_lengths = []
    action_by_depth: dict[int, Counter] = {}
    opportunity_by_depth: dict[int, Counter] = {}
    by_context: dict[str, Counter] = {}
    opportunities_by_context: dict[str, Counter] = {}
    preflop_allin_hands = 0
    preflop_fold_ended_hands = 0
    preflop_final_call_hands = 0
    no_postflop_decision_hands = 0
    explicit_allin_after_raised = 0
    allin_selected_total = 0

    offset = 0
    for length, final_action in zip(rows.decisions_per_hand, rows.terminal_actions):
        if length < 1:
            raise ValueError("invalid empty hand")
        end = offset + length
        actions = rows.actions[offset:end]
        actors = rows.actors[offset:end]
        offset = end
        if actions[-1] != final_action:
            raise ValueError("last action/terminal action mismatch")
        pre = []
        seen_later_street = False
        for action, actor in zip(actions, actors):
            street = int(actor.features[2])
            if float(street) != actor.features[2] or street not in range(4):
                raise ValueError("invalid actor street index")
            if street > 0:
                seen_later_street = True
                continue
            if seen_later_street:
                raise ValueError("non-monotone street order")
            pre.append((action, actor))
        if not pre:
            raise ValueError("each hand must begin with a preflop decision")
        seq = [ACTION_NAMES[a] for a, _ in pre]
        preflop_decision_lengths.append(len(pre))
        complete_preflop[" > ".join(seq)] += 1
        first_action[seq[0]] += 1
        if len(pre) == len(actions):
            no_postflop_decision_hands += 1
            final_preflop_action[seq[-1]] += 1
            if final_action == 0:
                preflop_fold_ended_hands += 1
                fold_preflop_decision_lengths.append(len(pre))
            elif final_action == 2:
                preflop_final_call_hands += 1
        prior_raise = False
        for depth, (action, actor) in enumerate(pre, start=1):
            action_by_depth.setdefault(depth, Counter())[ACTION_NAMES[action]] += 1
            for j, name in enumerate(ACTION_NAMES):
                if actor.legal_action_mask[j]:
                    opportunity_by_depth.setdefault(depth, Counter())[name] += 1
            facing = "facing_bet" if float(actor.features[4]) > 0 else "not_facing_bet"
            position = "button" if float(actor.features[1]) == 1 else "big_blind"
            for context in (f"facing:{facing}", f"position:{position}",
                            f"facing_position:{facing}:{position}"):
                by_context.setdefault(context, Counter())[ACTION_NAMES[action]] += 1
                for j, name in enumerate(ACTION_NAMES):
                    if actor.legal_action_mask[j]:
                        opportunities_by_context.setdefault(context, Counter())[name] += 1
            if action == 6:
                allin_selected_total += 1
            if action in RAISE_INDICES:
                prior_raise = True
        first_allin_indices = [j for j, (a, _) in enumerate(pre) if a == 6]
        if first_allin_indices:
            preflop_allin_hands += 1
            allin_preflop_decision_lengths.append(len(pre))
            j = first_allin_indices[0]
            first_allin_step[str(j + 1)] += 1
            prefix_to_allin[" > ".join(seq[:j + 1])] += 1
            before_allin["START" if j == 0 else seq[j - 1]] += 1
            if any(a in RAISE_INDICES for a, _ in pre[:j]):
                explicit_allin_after_raised += 1

    if offset != len(rows.actions):
        raise ValueError("hand boundary mismatch")

    def _breakdown(counts: Counter, legal: Counter) -> dict:
        out = {}
        for name in ACTION_NAMES:
            chosen = counts[name]
            opportunities = legal[name]
            if chosen > opportunities:
                raise AssertionError("illegal sampled action detected")
            out[name] = {
                "selected": chosen,
                "legal_decisions": opportunities,
                "selected_given_legal": (
                    round(chosen / opportunities, 6) if opportunities else None
                ),
            }
        return out

    return {
        "hands": rows.hands,
        "preflop_explicit_allin_hands": preflop_allin_hands,
        "preflop_explicit_allin_fraction": round(preflop_allin_hands / rows.hands, 6),
        "explicit_allin_action_count": allin_selected_total,
        "preflop_last_decision_hands": no_postflop_decision_hands,
        "preflop_terminal_fold_hands": preflop_fold_ended_hands,
        "preflop_terminal_call_hands": preflop_final_call_hands,
        "first_actions": dict(sorted(first_action.items())),
        "final_preflop_actions": dict(sorted(final_preflop_action.items())),
        "preflop_decisions_per_hand": _summary(preflop_decision_lengths),
        "allin_hands_preflop_decisions": _summary(allin_preflop_decision_lengths),
        "fold_terminal_hands_preflop_decisions": _summary(fold_preflop_decision_lengths),
        "first_allin_decision_number_counts": dict(sorted(first_allin_step.items(),
                                                           key=lambda x: int(x[0]))),
        "immediately_before_first_allin": dict(sorted(before_allin.items())),
        "allin_with_prior_raise_head_count": explicit_allin_after_raised,
        "most_common_complete_preflop_sequences": _top_counts(complete_preflop),
        "most_common_prefixes_through_first_allin": _top_counts(prefix_to_allin),
        "action_choice_by_preflop_decision_number": {
            str(k): _breakdown(action_by_depth[k], opportunity_by_depth[k])
            for k in sorted(action_by_depth)
        },
        "action_choice_by_context": {
            k: _breakdown(by_context[k], opportunities_by_context[k])
            for k in sorted(by_context)
        },
        "notes": [
            "One preflop action-sequence entry per HAND; no private cards or engine internals.",
            "A raise-head includes half-pot, pot, double-pot and explicit all-in.",
            "Action-head sequences do not establish exact effective wager amounts or "
            "whether a formal 3-bet, 4-bet or call-to-all-in occurred.",
            "Preflop terminal means LAST ACTOR decision was preflop; an "
            "all-in showdown can run out all remaining board cards automatically.",
            "An all-in can occur through calling off the remaining stack "
            "without choosing explicit action head 6.",
        ],
    }


def compare_preflop_sequences(*, trained_checkpoint: str | Path,
                              initial_checkpoint: str | Path | None = None,
                              init_seed: int = 52, device: str = "cpu",
                              hands: int = 512, seed: int = 600_000_000,
                              env_batch_size: int = 32) -> dict:
    if (type(hands) is not int or hands < 1 or type(seed) is not int
            or seed < 0 or type(env_batch_size) is not int
            or not 1 <= env_batch_size <= hands):
        raise ValueError("invalid hands/seed/env_batch_size")
    trained = load_actor(trained_checkpoint, device=device)
    if trained.config.dropout != 0:
        raise ValueError("actor requires dropout=0")
    if initial_checkpoint is not None:
        initial = load_actor(initial_checkpoint, device=device)
        if initial.config != trained.config:
            raise ValueError("initial network config differs from trained network")
        provenance = "user_supplied_checkpoint_not_verified_untrained"
    else:
        initial = build_untrained_actor(trained.config, seed=init_seed, device=device)
        provenance = "synthetic_random_initialization_NOT_verified_phase14_start"
    models = {
        "uniform_legal": UniformLegalActor(trained.config).to(device).eval(),
        "random_init": initial,
        "trained_phase14": trained,
    }
    cfg = PPOConfig(hands_per_update=hands, seed=seed)
    reports = {}
    for label, actor in models.items():
        rows = collect_batched_self_play(
            actor, cfg, iteration=0, device=device, env_batch_size=env_batch_size)
        reports[label] = summarize_preflop_sequences(rows)
    return {
        "protocol": "phase15_three_agent_preflop_sequences",
        "trained_checkpoint": str(trained_checkpoint),
        "initial_checkpoint": (str(initial_checkpoint)
                               if initial_checkpoint is not None else None),
        "initial_policy_provenance": provenance,
        "hands_per_policy": hands, "seed": seed, "device": device,
        "models": reports,
        "limitation": "Policies self-play on the same deck seeds but visit "
                      "different betting histories. These are sampled trajectories, "
                      "not GTO, exploitability, EV, or formal 3-bet labels.",
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/preflop_sequence_audit.py" -->

## src/holdem_ai/evaluation/reward_diagnostics.py

Original SHA `8a2f27abc14d6e25d462385efae15cb2e05c64f6` (1717 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/reward_diagnostics.py" sha="8a2f27abc14d6e25d462385efae15cb2e05c64f6" -->

````````python
"""Summarize on-policy HUNL terminal rewards and termination actions per update.

No opponent private cards or full engine state is used. Showdown is NOT
inferred solely from a non-fold ending; report non-fold explicitly.
"""
from __future__ import annotations

import math
from collections import Counter

from holdem_ai.training.ppo import TrajectoryRows


def summarize_rollout_rewards(rows: TrajectoryRows) -> dict:
    payoffs = [float(r[0]) for r in rows.hand_rewards]
    if len(payoffs) != rows.hands or not payoffs or any(not math.isfinite(x) for x in payoffs):
        raise ValueError("invalid hand rewards")
    n = len(payoffs)
    mean = math.fsum(payoffs) / n
    var = math.fsum((v - mean) ** 2 for v in payoffs) / n
    out = {
        "hands": n,
        "seat0_mean_reward_bb": round(mean, 6),
        "seat0_reward_std_bb": round(math.sqrt(var), 6),
        "mean_absolute_terminal_reward_bb": round(math.fsum(abs(v) for v in payoffs) / n, 6),
        "min_seat0_reward_bb": round(min(payoffs), 6),
        "max_seat0_reward_bb": round(max(payoffs), 6),
        "decisions_per_hand_mean": (round(sum(rows.decisions_per_hand) / n, 6)
                                     if rows.decisions_per_hand else None),
        "fold_terminated_hands": None,
        "fold_termination_fraction": None,
        "non_fold_terminated_hands": None,
        "showdown_fraction": None,  # Not observable from terminal action alone.
    }
    if rows.terminal_actions:
        counts = Counter(rows.terminal_actions)
        fold = counts[0]
        out["fold_terminated_hands"] = fold
        out["fold_termination_fraction"] = round(fold / n, 6)
        out["non_fold_terminated_hands"] = n - fold
    return out

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/reward_diagnostics.py" -->
