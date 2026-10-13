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
