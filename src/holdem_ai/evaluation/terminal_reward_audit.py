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
from dataclasses import replace
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
        "last_decision_street_x_termination": {
            f"{street}:{reason}": stage_term[f"{street}:{reason}"]
            for street in STREETS for reason in ("fold", "nonfold_showdown")
        },
        "preflop": {
            "last_decision_hand_count": len(preflop),
            "last_decision_hand_fraction": round(len(preflop) / n, 6),
            "fold_ended_count": sum(h["termination"] == "fold" for h in preflop),
            "nonfold_showdown_count": len(preflop_nonfold),
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
