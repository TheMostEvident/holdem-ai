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
