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
