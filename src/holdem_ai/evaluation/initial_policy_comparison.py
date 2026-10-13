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
