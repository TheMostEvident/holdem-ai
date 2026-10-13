# Complete Python source volume 15/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## src/holdem_ai/experiments/ppo_ablation.py

Original blob `dd2b4f6bbbb962d9d6cec5df09d6114ab46cc547` (25669 bytes).

<a name="py-169"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/experiments/ppo_ablation.py" sha="dd2b4f6bbbb962d9d6cec5df09d6114ab46cc547" -->

````````python
"""Phase16 isolated PPO optimizer/critic ablations (AGENT ONLY).

A0: original PPO Q critic + original PPO optimizer.
A1: Q-head target and baseline scaled to stack units (G/200), Huber beta=1.
A2: Direct privileged state V(s) baseline with identical seat zero-sum symmetry.
A3: original PPO + EXACT post-update KL gate with full update rollback.

This is an intentionally short NEW-RUN experiment. No modification of
environment, GUI, Phase15 optimizer implementation, source checkpoints or
prior training reports.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.evaluation.terminal_reward_audit import summarize_terminal_distribution
from holdem_ai.evaluation.initial_policy_comparison import _action_counts_on_rows
from holdem_ai.evaluation.seed_blocked import (
    EvaluationConfig, compare_results, evaluate_seed_blocks,
)
from holdem_ai.models.transformer import CentralizedQCritic, TransformerActor, chosen_q
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.training.ppo import (
    FrozenPPOData, PPOConfig, TrajectoryRows, clipped_surrogate,
    freeze_old_policy_data, optimize_ppo,
)
from holdem_ai.training.ppo_session import load_checkpoint

VARIANTS = ("a0", "a1", "a2", "a3")
LABELS = {
    "a0": "unmodified_PPO",
    "a1": "Q_target_scaled_200BB",
    "a2": "direct_privileged_V",
    "a3": "post_update_exact_KL_rollback",
}


class DirectVCritic(nn.Module):
    """Train-only centralized V [B, seat]; zero-sum via +/- same value.

    Reuses Q Critic's token encoder and hidden layer; final projection is the
    arithmetic mean of Q-action head rows at warm-start. It is a newly shaped
    critic, not a valid existing Q optimizer state.
    """
    def __init__(self, source_q: CentralizedQCritic):
        super().__init__()
        self.tokens = copy.deepcopy(source_q.tokens)
        self.v_head = nn.Sequential(
            copy.deepcopy(source_q.q_head[0]),
            copy.deepcopy(source_q.q_head[1]),
            nn.Linear(source_q.q_head[2].in_features, 1),
        )
        with torch.no_grad():
            self.v_head[2].weight.copy_(source_q.q_head[2].weight.mean(dim=0, keepdim=True))
            self.v_head[2].bias.copy_(source_q.q_head[2].bias.mean().reshape(1))

    def forward(self, critic_batch):
        state = self.tokens(critic_batch.token_ids, critic_batch.token_values,
                            critic_batch.attention_mask)
        v0 = self.v_head(torch.cat((state, torch.asinh(critic_batch.features)), dim=-1))
        v0 = v0.masked_fill(critic_batch.features[:, -1:].to(dtype=torch.bool), 0.0)
        return torch.cat((v0, -v0), dim=1)


def scale_q_head_(critic: CentralizedQCritic, scale: float) -> None:
    """Reparameterize final Q head without altering initial Q in BB units."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite scale required")
    with torch.no_grad():
        critic.q_head[2].weight.div_(scale)
        critic.q_head[2].bias.div_(scale)


def freeze_variant(actor: TransformerActor, critic: nn.Module,
                   rows: TrajectoryRows, *, variant: str,
                   target_scale: float, device: str, chunk_size: int = 64) -> FrozenPPOData:
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    if variant in ("a0", "a3"):
        return freeze_old_policy_data(
            actor, critic, rows, device=device, chunk_size=chunk_size)
    old_logs, values = [], []
    actor.eval()
    critic.eval()
    with torch.inference_mode():
        for start in range(0, len(rows.actions), chunk_size):
            stop = min(start + chunk_size, len(rows.actions))
            ab = batch_actors(rows.actors[start:stop], device=device)
            cb = batch_critics(rows.critics[start:stop], device=device)
            seats = torch.tensor(rows.acting_seats[start:stop], dtype=torch.long, device=device)
            actions = torch.tensor(rows.actions[start:stop], dtype=torch.long, device=device)
            dist = actor.distribution(ab)
            old_logs.append(dist.log_prob(actions).detach().cpu())
            value = (dist.probs * critic(cb)[torch.arange(len(seats), device=device), seats]).sum(dim=1) if variant == "a1" else (
                critic(cb)[torch.arange(len(seats), device=device), seats]
            )
            values.append(value.detach().cpu())
    old_logs_cpu = torch.cat(old_logs)
    baselines = torch.cat(values)
    targets = torch.tensor(rows.returns_bb, dtype=torch.float32)
    if variant == "a1":
        targets = targets / target_scale
    raw_advantage = targets - baselines
    advantage = (raw_advantage - raw_advantage.mean()) / raw_advantage.std(unbiased=False).clamp_min(1e-6)
    if not all(bool(torch.isfinite(t).all()) for t in (old_logs_cpu, baselines, advantage)):
        raise FloatingPointError("nonfinite frozen PPO quantities")
    return FrozenPPOData(rows, old_logs_cpu, advantage, baselines)


def optimize_variant(actor: TransformerActor, critic: nn.Module,
                     actor_optimizer: torch.optim.Optimizer,
                     critic_optimizer: torch.optim.Optimizer,
                     frozen: FrozenPPOData, config: PPOConfig, *,
                     variant: str, target_scale: float, device: str,
                     shuffle_seed: int) -> dict:
    """A1/A2 differ from original PPO only in target and baseline critic path."""
    if variant not in ("a1", "a2"):
        raise ValueError("custom variant must be a1 or a2")
    actor.train()
    critic.train()
    prng = random.Random(shuffle_seed)
    totals = {k: 0. for k in ("actor_loss", "critic_loss", "entropy",
                              "grad_norm_actor", "grad_norm_critic")}
    batches = 0
    for _ in range(config.epochs):
        order = list(range(len(frozen.rows.actions)))
        prng.shuffle(order)
        for offset in range(0, len(order), config.minibatch_size):
            indices = order[offset:offset + config.minibatch_size]
            ab = batch_actors([frozen.rows.actors[i] for i in indices], device=device)
            cb = batch_critics([frozen.rows.critics[i] for i in indices], device=device)
            actions = torch.tensor([frozen.rows.actions[i] for i in indices], dtype=torch.long, device=device)
            seats = torch.tensor([frozen.rows.acting_seats[i] for i in indices], dtype=torch.long, device=device)
            targets = torch.tensor([frozen.rows.returns_bb[i] for i in indices],
                                   dtype=torch.float32, device=device)
            if variant == "a1":
                targets = targets / target_scale
            old_logs = frozen.old_log_probs[indices].to(device)
            advantages = frozen.advantages[indices].to(device)
            dist = actor.distribution(ab)
            surrogate = clipped_surrogate(dist.log_prob(actions), old_logs,
                                          advantages, config.clip_epsilon)
            policy_loss = -surrogate.mean() - config.entropy_coef * dist.entropy().mean()
            estimates = critic(cb)
            batch_indices = torch.arange(len(indices), device=device)
            predictions = (chosen_q(estimates, seats, actions)
                           if variant == "a1" else estimates[batch_indices, seats])
            critic_loss = F.smooth_l1_loss(predictions, targets)
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            (config.value_coef * critic_loss).backward()
            ga = nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip)
            gc = nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip)
            if not all(math.isfinite(x) for x in (
                float(policy_loss.item()), float(critic_loss.item()),
                float(ga.item()), float(gc.item()))):
                raise FloatingPointError("nonfinite loss or gradients")
            actor_optimizer.step()
            critic_optimizer.step()
            totals["actor_loss"] += float(policy_loss.item())
            totals["critic_loss"] += float(critic_loss.item())
            totals["entropy"] += float(dist.entropy().mean().item())
            totals["grad_norm_actor"] += float(ga.item())
            totals["grad_norm_critic"] += float(gc.item())
            batches += 1
    return {**{k: round(v / batches, 6) for k, v in totals.items()},
            "minibatches": batches, "decisions": len(frozen.rows.actions)}


@torch.no_grad()
def exact_policy_kl(actor: TransformerActor, rows: TrajectoryRows, *,
                    old_probs: torch.Tensor, device: str,
                    chunk_size: int = 64) -> float:
    """D_KL(pi_old || pi_new) measured AFTER updating, on frozen rollout states."""
    if old_probs.shape != (len(rows.actions), 7):
        raise ValueError("old_probs must be [decisions,7]")
    actor.eval()
    numerator = 0.
    for start in range(0, len(rows.actions), chunk_size):
        end = min(len(rows.actions), start + chunk_size)
        distribution = actor.distribution(batch_actors(rows.actors[start:end], device=device))
        p = old_probs[start:end].to(device=device)
        q = distribution.probs
        legal = torch.tensor([a.legal_action_mask for a in rows.actors[start:end]],
                             dtype=torch.bool, device=device)
        # Only legal probabilities can contribute to categorical KL.
        local = torch.where(
            legal & (p > 0),
            p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log()),
            torch.zeros_like(p),
        ).sum(dim=1)
        numerator += float(local.sum().item())
    value = numerator / len(rows.actions)
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite exact post-update policy KL")
    return max(0., value)


@torch.no_grad()
def snapshot_probs(actor: TransformerActor, rows: TrajectoryRows, *,
                   device: str, chunk_size: int = 64) -> torch.Tensor:
    actor.eval()
    old = []
    for start in range(0, len(rows.actions), chunk_size):
        end = min(len(rows.actions), start + chunk_size)
        batch = batch_actors(rows.actors[start:end], device=device)
        old.append(actor.distribution(batch).probs.detach().cpu())
    return torch.cat(old)


@torch.no_grad()
def critic_on_frozen_rows(actor: TransformerActor, critic: nn.Module,
                          rows: TrajectoryRows, *, variant: str,
                          scale: float, device: str) -> dict:
    """Descriptive OUT-OF-TRAINING-ROLLOUT calibration, decision weighted."""
    actor.eval()
    critic.eval()
    predictions = []
    for start in range(0, len(rows.actions), 64):
        end = min(start + 64, len(rows.actions))
        cb = batch_critics(rows.critics[start:end], device=device)
        ab = batch_actors(rows.actors[start:end], device=device)
        seats = torch.tensor(rows.acting_seats[start:end], dtype=torch.long, device=device)
        actions = torch.tensor(rows.actions[start:end], dtype=torch.long, device=device)
        batch_idx = torch.arange(len(seats), device=device)
        if variant == "a2":
            values = critic(cb)[batch_idx, seats]
        else:
            values = (actor.distribution(ab).probs * critic(cb)[batch_idx, seats]).sum(dim=1)
            if variant == "a1":
                values = values * scale
        predictions.extend(values.cpu().tolist())
    target = list(rows.returns_bb)
    err = [p - t for p, t in zip(predictions, target)]
    pred_mean = math.fsum(predictions) / len(predictions)
    target_mean = math.fsum(target) / len(target)
    target_variance = math.fsum((v - target_mean) ** 2 for v in target) / len(target)
    err_mean = math.fsum(err) / len(err)
    err_var = math.fsum((v - err_mean) ** 2 for v in err) / len(err)
    return {
        "decisions": len(target),
        "rmse_bb": round(math.sqrt(math.fsum(v*v for v in err) / len(err)), 6),
        "mae_bb": round(math.fsum(abs(v) for v in err) / len(err), 6),
        "explained_variance": round(1. - err_var / target_variance, 6) if target_variance > 0 else None,
        "prediction_mean_bb": round(pred_mean, 6),
        "target_mean_bb": round(target_mean, 6),
        "prediction_std_bb": round(math.sqrt(math.fsum((p-pred_mean)**2 for p in predictions) / len(predictions)), 6),
        "note": "V(s) baseline calibration on fresh on-policy decision rows; "
                "hand-correlated observations, not independent samples.",
    }


@dataclass(frozen=True)
class AblationConfig:
    source_checkpoint: str
    root: str
    device: str = "cpu"
    variants: tuple[str, ...] = VARIANTS
    hands_per_update: int = 64
    updates: int = 8
    minibatch_size: int = 64
    epochs: int = 1
    env_batch_size: int = 32
    eval_groups: int = 12
    rollout_seed: int = 620_000_000
    validation_seed: int = 75_000_000
    critic_seed: int = 660_000_000
    critic_eval_hands: int = 64
    target_scale: float = 200.0
    max_post_kl: float = 0.02

    def __post_init__(self):
        if not self.source_checkpoint or not self.root:
            raise ValueError("source_checkpoint and root required")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("unsupported device")
        if not self.variants or len(set(self.variants)) != len(self.variants) or any(
            v not in VARIANTS for v in self.variants
        ):
            raise ValueError("variants must be a unique subset of a0,a1,a2,a3")
        for key in ("hands_per_update", "updates", "minibatch_size", "epochs",
                    "env_batch_size", "eval_groups", "critic_eval_hands"):
            x = getattr(self, key)
            if type(x) is not int or x < (2 if key == "eval_groups" else 1):
                raise ValueError(f"{key} invalid")
        if self.env_batch_size > self.hands_per_update:
            raise ValueError("batch of environments exceeds hands/update")
        for key in ("rollout_seed", "validation_seed", "critic_seed"):
            x = getattr(self, key)
            if type(x) is not int or x < 0:
                raise ValueError(f"{key} must be nonnegative int")
        ranges = (
            (self.rollout_seed,
             self.rollout_seed + self.hands_per_update * self.updates),
            (self.validation_seed, self.validation_seed + self.eval_groups),
            (self.critic_seed, self.critic_seed + self.critic_eval_hands),
        )
        if any(max(a0, b0) < min(a1, b1)
               for i, (a0, a1) in enumerate(ranges)
               for b0, b1 in ranges[i + 1:]):
            raise ValueError("diagnostic/evaluation seed ranges overlap")
        if not (math.isfinite(self.target_scale) and self.target_scale > 0
                and math.isfinite(self.max_post_kl) and self.max_post_kl > 0):
            raise ValueError("invalid critic scaling/KL gate")


def _put_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _actor_save(actor: TransformerActor, path: Path) -> None:
    # Evaluation-only load_actor-compatible output; intentionally no Adam state.
    payload = {
        "phase": 9,
        "network_config": asdict(actor.config),
        "actor_state": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
        "usage": "phase16_evaluation_only_not_training_resume",
    }
    with path.open("xb") as f:
        torch.save(payload, f)


def _eval(actor: TransformerActor, source: TransformerActor, *,
          config: EvaluationConfig, control: dict) -> dict:
    result = {}
    for opponent, name in ((None, "uniform_random_legal"), (source, "frozen_phase14")):
        report = evaluate_seed_blocks(actor, opponent=opponent, config=config)
        # Compare on paired deck blocks against exactly the same frozen opponent.
        result[name] = {
            **report.summary(),
            "paired_change_from_source": compare_results(report, control[name]),
        }
    return result


def run_ablation(config: AblationConfig) -> dict:
    """Exclusive fresh-run short experiment: all variants use common data seeds.

    A0/A3 have original source Q and matching Actor/Adam moments; A1 transforms
    Q output head from BB to scaled-BB; A2 initializes V from Q mean action head.
    ALL variants reset critic Adam moments to avoid incomparable reparameterized
    optimizer states. Source checkpoint is never modified.
    """
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source = Path(config.source_checkpoint)
    if not source.is_file():
        raise FileNotFoundError(f"missing trusted source checkpoint: {source}")
    root = Path(config.root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {root}")
    if root.resolve() == source.parent.resolve() or root.resolve() in source.resolve().parents:
        raise ValueError("run root must not be source checkpoint parent")
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _put_json(root / "run.json", {
        "protocol": "phase16_new_run_ppo_ablation", "config": asdict(config),
        "source_sha256": digest,
        "restrictions": "No engine/GUI changes or writing to source/old runs. "
                        "No ablation is presumed better before paired validation.",
        "baseline_fairness": "All arms start with source Actor weights and Actor Adam "
                             "state; all arms reset Critic Adam from same LR. "
                             "A1/A2 are new Critic parametrizations initialized from source Q.",
    })
    source_actor = load_actor(source, device=config.device)
    eval_cfg = EvaluationConfig(
        groups=config.eval_groups, seed=config.validation_seed,
        mode="sampled", device=config.device,
    )
    control = {
        name: evaluate_seed_blocks(source_actor, opponent=opponent, config=eval_cfg)
        for opponent, name in ((None, "uniform_random_legal"),
                               (load_actor(source, device=config.device), "frozen_phase14"))
    }
    report = {"source": {key: entry.summary() for key, entry in control.items()},
              "variants": {}, "protocol": "phase16_new_run_ppo_ablation"}
    for variant in config.variants:
        arm_root = root / variant
        arm_root.mkdir(exist_ok=False)
        session = load_checkpoint(source, device=config.device)
        session.ppo_config = replace(
            session.ppo_config,
            hands_per_update=config.hands_per_update,
            minibatch_size=config.minibatch_size,
            epochs=config.epochs,
            seed=config.rollout_seed - session.completed_updates * config.hands_per_update,
        )
        critic: nn.Module = session.critic
        if variant == "a1":
            scale_q_head_(session.critic, config.target_scale)
        if variant == "a2":
            critic = DirectVCritic(session.critic).to(config.device)
        # Same fresh Critic Adam state for every arm. Actor optimizer retained
        # from the source checkpoint for every arm.
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=session.ppo_config.learning_rate,
        )
        updates = []
        for t in range(config.updates):
            step = session.completed_updates + t
            rows = collect_batched_self_play(
                session.actor, session.ppo_config, iteration=step,
                device=config.device, env_batch_size=config.env_batch_size,
            )
            frozen = freeze_variant(
                session.actor, critic, rows, variant=variant,
                target_scale=config.target_scale, device=config.device,
                chunk_size=session.ppo_config.inference_batch_size,
            )
            old_probs = snapshot_probs(session.actor, rows, device=config.device)
            if variant == "a3":
                saved_actor = copy.deepcopy(session.actor.state_dict())
                saved_critic = copy.deepcopy(critic.state_dict())
                saved_aopt = copy.deepcopy(session.actor_optimizer.state_dict())
                saved_copt = copy.deepcopy(critic_optimizer.state_dict())
            if variant in ("a0", "a3"):
                metrics = optimize_ppo(
                    session.actor, critic, session.actor_optimizer,
                    critic_optimizer, frozen, session.ppo_config,
                    device=config.device,
                    shuffle_seed=session.ppo_config.seed + 100003 * step,
                )
            else:
                metrics = optimize_variant(
                    session.actor, critic, session.actor_optimizer,
                    critic_optimizer, frozen, session.ppo_config,
                    variant=variant, target_scale=config.target_scale,
                    device=config.device,
                    shuffle_seed=session.ppo_config.seed + 100003 * step,
                )
            kl = exact_policy_kl(
                session.actor, rows, old_probs=old_probs, device=config.device)
            rejected = variant == "a3" and kl > config.max_post_kl
            if rejected:
                session.actor.load_state_dict(saved_actor)
                critic.load_state_dict(saved_critic)
                session.actor_optimizer.load_state_dict(saved_aopt)
                critic_optimizer.load_state_dict(saved_copt)
            state = {
                "new_update_index": t + 1,
                "global_source_update": step,
                "hands": rows.hands,
                "decisions": len(rows.actions),
                "post_update_exact_kl": round(kl, 7),
                "step_rejected": rejected,
                "accepted_updates_cumulative": sum(not row["step_rejected"] for row in updates) + int(not rejected),
                **metrics,
            }
            updates.append(state)
            print(f"{variant} {t+1}/{config.updates} "
                  f"KL={kl:.5f} rejected={rejected} "
                  f"critic_loss={metrics['critic_loss']:.4f}", flush=True)
        # Evaluate after fitting on a new, independent on-policy hand block.
        hold_cfg = PPOConfig(
            seed=config.critic_seed, hands_per_update=config.critic_eval_hands,
            big_blind=session.ppo_config.big_blind,
        )
        holdout = collect_batched_self_play(
            session.actor, hold_cfg, iteration=0,
            device=config.device,
            env_batch_size=min(config.env_batch_size, config.critic_eval_hands),
        )
        critic_stats = critic_on_frozen_rows(
            session.actor, critic, holdout, variant=variant,
            scale=config.target_scale, device=config.device,
        )
        terminal_stats = summarize_terminal_distribution(holdout)
        action_stats = _action_counts_on_rows(holdout)
        on_policy_behavior = {
            "hands": holdout.hands,
            "preflop_explicit_allin_fraction": round(
                terminal_stats["preflop"]["explicit_all_in_hand_count"] / holdout.hands, 6),
            "preflop_allin_selected_given_legal": action_stats["preflop"]["all_in_rate_given_legal"],
            "preflop_fold_ended_fraction": round(
                terminal_stats["preflop"]["fold_ended_count"] / holdout.hands, 6),
            "postflop_decision_hand_fraction": round(
                terminal_stats["decision_coverage"]["flop_hands"] / holdout.hands, 6),
            "near_full_stack_fraction": terminal_stats["seat0_hand_weighted_reward"]["near_full_stack_fraction"],
            "reward_std_bb": terminal_stats["seat0_hand_weighted_reward"]["std_bb"],
        }
        evaluation = _eval(session.actor, source_actor, config=eval_cfg, control=control)
        _actor_save(session.actor, arm_root / "final_actor.pt")
        data = {
            "variant": variant, "label": LABELS[variant],
            "updates": updates, "critic_holdout": critic_stats,
            "on_policy_behavior": on_policy_behavior,
            "evaluation": evaluation,
            "rejected_updates": sum(row["step_rejected"] for row in updates),
            "post_kl_max": round(max(row["post_update_exact_kl"] for row in updates), 7),
            "checkpoint": str(arm_root / "final_actor.pt"),
            "limitations": "Final actor file is EVALUATION-ONLY; no resume optimizer "
                           "or Critic state. Paired deck-group CIs remain very "
                           "wide at low groups; not proof of stability/convergence.",
        }
        _put_json(arm_root / "result.json", data)
        report["variants"][variant] = {
            "label": data["label"], "critic_holdout": data["critic_holdout"],
            "on_policy_behavior": data["on_policy_behavior"],
            "evaluation": data["evaluation"],
            "rejected_updates": data["rejected_updates"],
            "post_kl_max": data["post_kl_max"], "checkpoint": data["checkpoint"],
        }
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("source checkpoint changed during ablation")
    _put_json(root / "summary.json", report)
    return report

````````

<!-- END_PY_SOURCE path="src/holdem_ai/experiments/ppo_ablation.py" -->

## src/holdem_ai/experiments/ppo_five_hour.py

Original blob `429308c67e536476a05f4d10c817e0fcbb09b5d7` (24123 bytes).

<a name="py-170"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/experiments/ppo_five_hour.py" sha="429308c67e536476a05f4d10c817e0fcbb09b5d7" -->

````````python
"""Phase17: resumable 5-hour PPO retraining (AGENT ONLY, new run).

Reuses validated Phase16 on-policy A0/A3 update, exact post-update KL and
matched-deck frozen-opponent evaluation. Existing engine, GUI, PPO math,
Phase14 source and Phase15/16 reports remain untouched.

The wall-clock budget accumulates ACTIVE invocation time, excluding downtime
between sessions. Training stops at an update boundary; validation at the
last update may extend past the requested wall-clock budget.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.experiments.longrun import _exclusive_lock
from holdem_ai.experiments.ppo_stress import (
    StressConfig, _baseline, _one_update, _validation,
    _json_atomic, _sha,
)
from holdem_ai.training.ppo_session import load_checkpoint


PROTOCOL = "phase17_five_hour_ppo_v1"


@dataclass(frozen=True)
class FiveHourConfig:
    source_checkpoint: str
    root: str
    device: str = "cpu"
    duration_hours: float = 5.0
    variant: str = "a0"       # A0 monitoring by default; A3 opt-in gate
    hands_per_update: int = 128
    minibatch_size: int = 128
    epochs: int = 1
    env_batch_size: int = 32
    eval_every: int = 25
    eval_groups: int = 80
    critic_eval_hands: int = 128
    save_every: int = 10
    max_updates: int = 5000
    rollout_seed: int = 700_000_000
    validation_seed: int = 80_000_000
    critic_seed: int = 710_000_000
    max_post_kl: float = 0.02
    hard_stop_kl: float = 0.10
    resume: bool = False
    max_updates_this_call: int | None = None

    def __post_init__(self) -> None:
        if not self.source_checkpoint or not self.root:
            raise ValueError("source and new run root required")
        if self.variant not in ("a0", "a3") or self.device not in ("cpu", "cuda"):
            raise ValueError("variant=a0|a3 and device=cpu|cuda required")
        if (isinstance(self.duration_hours, bool)
                or not isinstance(self.duration_hours, (int, float))
                or not math.isfinite(self.duration_hours)
                or self.duration_hours <= 0):
            raise ValueError("duration_hours must be positive finite")
        for key in ("hands_per_update", "minibatch_size", "epochs",
                    "env_batch_size", "eval_every", "eval_groups",
                    "critic_eval_hands", "save_every", "max_updates"):
            value = getattr(self, key)
            if type(value) is not int or value < (2 if key == "eval_groups" else 1):
                raise ValueError(f"{key} invalid")
        if self.env_batch_size > self.hands_per_update:
            raise ValueError("env batch cannot exceed rollout batch")
        if self.max_updates_this_call is not None and (
            type(self.max_updates_this_call) is not int
            or self.max_updates_this_call < 1
        ):
            raise ValueError("max_updates_this_call must be positive or omitted")
        for key in ("rollout_seed", "validation_seed", "critic_seed"):
            value = getattr(self, key)
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be nonnegative integer")
        for key in ("max_post_kl", "hard_stop_kl"):
            value = getattr(self, key)
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be positive finite")
        if self.hard_stop_kl <= self.max_post_kl:
            raise ValueError("hard_stop_kl must exceed max_post_kl")
        ranges = (
            (self.rollout_seed,
             self.rollout_seed + self.max_updates * self.hands_per_update),
            (self.validation_seed, self.validation_seed + self.eval_groups),
            (self.critic_seed, self.critic_seed + self.critic_eval_hands),
        )
        if any(max(x0, y0) < min(x1, y1)
               for i, (x0, x1) in enumerate(ranges)
               for y0, y1 in ranges[i + 1:]):
            raise ValueError("training/evaluation seed ranges overlap")


def _settings(cfg: FiveHourConfig) -> dict:
    fields = asdict(cfg)
    fields.pop("resume")
    fields.pop("max_updates_this_call")
    return fields


def _stress_cfg(cfg: FiveHourConfig) -> StressConfig:
    """Reuse proven Phase16 evaluation and A0/A3 optimizer path unchanged."""
    return StressConfig(
        source_checkpoint=cfg.source_checkpoint,
        root=cfg.root, variant=cfg.variant, device=cfg.device,
        updates=cfg.max_updates, hands_per_update=cfg.hands_per_update,
        minibatch_size=cfg.minibatch_size, epochs=cfg.epochs,
        env_batch_size=cfg.env_batch_size,
        eval_every=cfg.eval_every, eval_groups=cfg.eval_groups,
        critic_eval_hands=cfg.critic_eval_hands, save_every=cfg.save_every,
        rollout_seed=cfg.rollout_seed,
        validation_seed=cfg.validation_seed,
        critic_seed=cfg.critic_seed,
        max_post_kl=cfg.max_post_kl, hard_stop_kl=cfg.hard_stop_kl,
    )


def _checkpoint(session, *, manifest_sha: str, start_update: int,
                elapsed_seconds: float, rows: list[dict],
                validations: list[dict], best_update: int,
                best_score: float) -> dict:
    """Single atomic full-state ledger; Phase9 load_checkpoint compatible."""
    return {
        "phase": 9, "algorithm": "ppo_mc_privileged_baseline",
        "completed_updates": session.completed_updates,
        "network_config": asdict(session.network_config),
        "ppo_config": asdict(session.ppo_config),
        "actor_state": session.actor.state_dict(),
        "critic_state": session.critic.state_dict(),
        "actor_optimizer": session.actor_optimizer.state_dict(),
        "critic_optimizer": session.critic_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else []),
        "retrain": {
            "protocol": PROTOCOL,
            "manifest_sha256": manifest_sha,
            "initial_global_update": start_update,
            "elapsed_active_seconds": float(elapsed_seconds),
            "attempted": len(rows),
            "accepted": sum(not r["rejected"] for r in rows),
            "rejected": sum(r["rejected"] for r in rows),
            "updates": rows,
            "validations": validations,
            "best_global_update": best_update,
            "best_validation_score": best_score,
        },
    }


def _commit(path: Path, session, *, manifest_sha: str, start_update: int,
            elapsed_seconds: float, rows: list[dict],
            validations: list[dict], best_update: int,
            best_score: float) -> None:
    payload = _checkpoint(
        session, manifest_sha=manifest_sha, start_update=start_update,
        elapsed_seconds=elapsed_seconds, rows=rows,
        validations=validations, best_update=best_update,
        best_score=best_score,
    )
    temp = path.with_suffix(".pt.tmp")
    try:
        with temp.open("xb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with tmp.open("xb") as f:
            with source.open("rb") as src:
                shutil.copyfileobj(src, f, length=1024 * 1024)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def _log_mirrors(root: Path, ledger: dict, *, cfg: FiveHourConfig,
                 start_update: int) -> None:
    rows, validations = ledger["updates"], ledger["validations"]
    path = root / "metrics.jsonl"
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"kind": "update", **row},
                               allow_nan=False, ensure_ascii=False) + "\n")
        for val in validations:
            f.write(json.dumps({"kind": "validation", **val},
                               allow_nan=False, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    _json_atomic(root / "progress.json", {
        "protocol": PROTOCOL,
        "variant": cfg.variant,
        "attempted_updates": len(rows),
        "accepted_updates": ledger["accepted"],
        "rejected_updates": ledger["rejected"],
        "global_update": start_update + len(rows),
        "duration_hours": cfg.duration_hours,
        "elapsed_active_hours": round(ledger["elapsed_active_seconds"] / 3600, 6),
        "remaining_active_hours": round(max(0., cfg.duration_hours
                                               - ledger["elapsed_active_seconds"] / 3600), 6),
        "latest_validation_attempt": (validations[-1]["attempted"] if validations else None),
        "best_global_update": ledger["best_global_update"],
        "best_validation_score": ledger["best_validation_score"],
        "last_post_update_kl": (rows[-1]["post_update_exact_kl"] if rows else None),
        "last_entropy": (rows[-1]["entropy"] if rows else None),
    })


def _read_latest(latest: Path, *, manifest_sha: str, start_update: int,
                 cfg: FiveHourConfig) -> tuple[dict, dict]:
    payload = torch.load(latest, map_location="cpu", weights_only=True)
    ledger = payload.get("retrain")
    if (not isinstance(ledger, dict)
            or ledger.get("protocol") != PROTOCOL
            or ledger.get("manifest_sha256") != manifest_sha
            or ledger.get("initial_global_update") != start_update):
        raise ValueError("not matching Phase17 PPO run checkpoint")
    rows, validations = ledger["updates"], ledger["validations"]
    if (not isinstance(rows, list) or not isinstance(validations, list)
            or payload.get("completed_updates") != start_update + len(rows)
            or len(rows) > cfg.max_updates
            or ledger["attempted"] != len(rows)
            or ledger["accepted"] != sum(not r["rejected"] for r in rows)
            or ledger["rejected"] != sum(r["rejected"] for r in rows)
            or not math.isfinite(ledger["elapsed_active_seconds"])
            or ledger["elapsed_active_seconds"] < 0):
        raise ValueError("inconsistent full-state training ledger")
    return payload, ledger


def _score(val: dict) -> float:
    # Exploratory selection ONLY: shared evaluation seeds bias best.pt.
    p = val["opponents"]
    return round((
        p["frozen_phase14"]["paired_vs_source"]["delta_mean_bb100"]
        + p["random_legal"]["paired_vs_source"]["delta_mean_bb100"]
    ) / 2., 4)


def _state_summary(root: Path, cfg: FiveHourConfig,
                   ledger: dict, start_update: int, *, reason: str) -> dict:
    n = ledger["attempted"]
    elapsed = ledger["elapsed_active_seconds"]
    result = {
        "protocol": PROTOCOL,
        "variant": cfg.variant, "stop_reason": reason,
        "attempted_updates": n,
        "accepted_updates": ledger["accepted"],
        "rejected_updates": ledger["rejected"],
        "global_update": start_update + n,
        "training_hands": n * cfg.hands_per_update,
        "elapsed_active_seconds": round(elapsed, 3),
        "duration_seconds": round(cfg.duration_hours * 3600, 3),
        "validation_count": len(ledger["validations"]),
        "best_global_update": ledger["best_global_update"],
        "best_validation_score_bb100": ledger["best_validation_score"],
        "latest_checkpoint": str(root / "latest.pt"),
        "best_checkpoint": str(root / "best.pt"),
        "complete": reason == "time_budget_reached",
        "note": "5h active wall time includes saving and periodic evaluations. "
                "A low KL and successful run do NOT prove stable poker strategy.",
    }
    _json_atomic(root / "summary.json", result)
    return result


def run_five_hour(cfg: FiveHourConfig) -> dict:
    """Checkpoint each attempt; Resume continues until cumulative active 5h.

    Time excludes human downtime between program invocations. No historical
    files or previous experiments are modified. With A0, a predeclared KL
    emergency stop (0.10) is analysis-only and blocks automatic resume.
    """
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source = Path(cfg.source_checkpoint).resolve(strict=True)
    root = Path(cfg.root).resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("run root must be disjoint from source checkpoint")
    if not cfg.resume and root.exists() and any(p.name != ".phase15.lock" for p in root.iterdir()):
        raise FileExistsError("fresh 5-hour run root exists: use --resume or another root")
    if cfg.resume and not (root / "run.json").is_file():
        raise FileNotFoundError("resume requires an existing run.json")
    with _exclusive_lock(root):
        startup_at = time.perf_counter()
        latest, manifest_path = root / "latest.pt", root / "run.json"
        if cfg.resume:
            if (root / "hard_stop.json").is_file():
                raise RuntimeError("KL hard-stop active; no automatic resume")
            # Interrupted atomic file publication may leave orphan temps.
            for orphan in (root / "latest.pt.tmp",
                           root / "best.pt.tmp",
                           root / "metrics.jsonl.tmp"):
                if orphan.exists():
                    orphan.unlink()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("settings") != _settings(cfg):
                raise ValueError("scientific settings differ from saved run")
            if manifest.get("source_sha256") != _sha(source):
                raise ValueError("original Phase14 source checkpoint changed")
            manifest_sha = _sha(manifest_path)
            payload, ledger = _read_latest(
                latest, manifest_sha=manifest_sha,
                start_update=manifest["initial_global_update"], cfg=cfg)
            session = load_checkpoint(latest, device=cfg.device)
            if cfg.device == "cuda" and payload.get("cuda_rng_states"):
                torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
            start_update = manifest["initial_global_update"]
            baseline = manifest["source_baseline"]
            if (ledger["updates"] and cfg.variant == "a0"
                    and ledger["updates"][-1]["post_update_exact_kl"] > cfg.hard_stop_kl):
                _json_atomic(root / "hard_stop.json", {
                    "reason": "a0_unsafe_post_update_kl",
                    "last_update": len(ledger["updates"]),
                })
                raise RuntimeError("A0 prior KL breach: refusing resume")
        else:
            session = load_checkpoint(source, device=cfg.device)
            # Preserve Phase16 A0/A3 comparison: fresh Critic Adam in either mode.
            session.critic_optimizer = torch.optim.Adam(
                session.critic.parameters(), lr=session.ppo_config.learning_rate)
            start_update = session.completed_updates
            session.ppo_config = replace(
                session.ppo_config,
                hands_per_update=cfg.hands_per_update,
                minibatch_size=cfg.minibatch_size,
                epochs=cfg.epochs,
                seed=cfg.rollout_seed - start_update * cfg.hands_per_update,
            )
            proxy = _stress_cfg(cfg)
            baseline = _baseline(
                load_actor(source, device=cfg.device),
                proxy, session.ppo_config.big_blind)
            manifest = {
                "protocol": PROTOCOL,
                "settings": _settings(cfg),
                "source_sha256": _sha(source),
                "initial_global_update": start_update,
                "source_baseline": baseline,
                "critic_adam_reset": True,
                "best_policy_selection": "exploratory fixed-seed validation only",
            }
            _json_atomic(manifest_path, manifest)
            manifest_sha = _sha(manifest_path)
            _commit(
                latest, session, manifest_sha=manifest_sha,
                start_update=start_update,
                elapsed_seconds=time.perf_counter() - startup_at,
                rows=[], validations=[], best_update=start_update, best_score=0.,
            )
            _copy_atomic(latest, root / "best.pt")
            _, ledger = _read_latest(
                latest, manifest_sha=manifest_sha,
                start_update=start_update, cfg=cfg)
        if _sha(source) != manifest["source_sha256"]:
            raise RuntimeError("original source changed during run")
        proxy = _stress_cfg(cfg)
        call_start = time.perf_counter()
        committed_elapsed = ledger["elapsed_active_seconds"]
        rows, validations = ledger["updates"], ledger["validations"]
        best_update = ledger["best_global_update"]
        best_score = ledger["best_validation_score"]

        def elapsed() -> float:
            return committed_elapsed + time.perf_counter() - call_start

        def commit_progress() -> dict:
            nonlocal ledger
            _commit(
                latest, session, manifest_sha=manifest_sha,
                start_update=start_update, elapsed_seconds=elapsed(),
                rows=rows, validations=validations,
                best_update=best_update, best_score=best_score,
            )
            _, ledger = _read_latest(
                latest, manifest_sha=manifest_sha,
                start_update=start_update, cfg=cfg)
            _log_mirrors(root, ledger, cfg=cfg, start_update=start_update)
            return ledger

        def save_snapshot(n: int) -> None:
            snapshot = root / "snapshots" / f"attempt_{n:05d}.pt"
            if snapshot.exists():
                return
            _copy_atomic(latest, snapshot)

        def evaluate_current(n: int) -> None:
            nonlocal best_update, best_score
            val = _validation(session, proxy, baseline, n)
            val["score_bb100_exploratory"] = _score(val)
            validations.append(val)
            if val["score_bb100_exploratory"] > best_score:
                best_score = val["score_bb100_exploratory"]
                best_update = session.completed_updates
            commit_progress()
            if best_update == session.completed_updates:
                _copy_atomic(latest, root / "best.pt")
            print(
                f"validation attempt={n} best_global={best_update} "
                f"score={val['score_bb100_exploratory']:.2f}bb/100 "
                f"criticEV={val['critic_holdout']['explained_variance']}",
                flush=True,
            )

        # If the process died after committing an initial/latest best but
        # before publishing best.pt, restore from the atomic latest.pt.
        if best_update == session.completed_updates:
            best_path = root / "best.pt"
            if (not best_path.is_file()
                    or int(torch.load(best_path, map_location="cpu",
                                      weights_only=True)["completed_updates"]) != best_update):
                _copy_atomic(latest, best_path)
        # Recover a missing scheduled OR final evaluation without drawing
        # another training rollout and without relying on a JSON mirror.
        at_end = bool(rows) and (
            len(rows) % cfg.eval_every == 0
            or len(rows) >= cfg.max_updates
            or (ledger["elapsed_active_seconds"] +
                rows[-1]["elapsed_seconds"] * 1.1 >= cfg.duration_hours * 3600.)
        )
        if at_end and not any(
            v["attempted"] == len(rows) for v in validations
        ):
            evaluate_current(len(rows))
        _log_mirrors(root, ledger, cfg=cfg, start_update=start_update)

        budget = cfg.duration_hours * 3600.
        previous = rows[-1]["elapsed_seconds"] if rows else 0.
        completed_this_call = 0
        reason = "time_budget_reached"
        try:
            while True:
                elapsed_now = elapsed()
                # Never launch an update expected to extend past the active
                # deadline; first update is allowed if any budget remains.
                if elapsed_now >= budget or (
                    previous > 0 and elapsed_now + previous * 1.1 >= budget
                ):
                    reason = "time_budget_reached"
                    break
                if len(rows) >= cfg.max_updates:
                    reason = "max_updates_ceiling"
                    break
                if (cfg.max_updates_this_call is not None
                        and completed_this_call >= cfg.max_updates_this_call):
                    reason = "invocation_update_limit"
                    break
                row = _one_update(session, proxy)
                row["attempted"] = len(rows) + 1
                rows.append(row)
                previous = row["elapsed_seconds"]
                completed_this_call += 1
                commit_progress()  # authoritative full Actor+Critic+Adam+RNG
                if len(rows) % cfg.save_every == 0:
                    save_snapshot(len(rows))
                print(
                    f"{cfg.variant} attempt={len(rows)} "
                    f"global={session.completed_updates} "
                    f"active_h={elapsed()/3600:.3f}/{cfg.duration_hours:.3f} "
                    f"KL={row['post_update_exact_kl']:.6f} "
                    f"entropy={row['entropy']:.4f} "
                    f"critic_loss={row['critic_loss']:.3f} "
                    f"allin={row['preflop_explicit_allin_fraction']:.3f} "
                    f"flop={row['flop_decision_fraction']:.3f}",
                    flush=True,
                )
                if cfg.variant == "a0" and row["post_update_exact_kl"] > cfg.hard_stop_kl:
                    _json_atomic(root / "hard_stop.json", {
                        "reason": "a0_unsafe_post_update_kl",
                        "attempted": len(rows),
                        "observed": row["post_update_exact_kl"],
                        "limit": cfg.hard_stop_kl,
                    })
                    reason = "kl_hard_stop"
                    break
                if len(rows) % cfg.eval_every == 0:
                    evaluate_current(len(rows))
        except KeyboardInterrupt:
            # No in-progress unsaved in-memory optimizer step is resumed.
            # The last atomic latest.pt remains authoritative.
            reason = "keyboard_interrupt"
            payload, ledger = _read_latest(
                latest, manifest_sha=manifest_sha,
                start_update=start_update, cfg=cfg)
            return _state_summary(root, cfg, ledger, start_update, reason=reason)

        # Last evaluation uses checkpointed model, with no optimizer updates.
        # This can continue after the 5-hour wall budget and is not training.
        if reason in ("time_budget_reached", "max_updates_ceiling"):
            n = len(rows)
            if n and not any(v["attempted"] == n for v in validations):
                evaluate_current(n)
        # Persist the final active wall time without changing the weights.
        commit_progress()
        if reason in ("time_budget_reached", "max_updates_ceiling") and rows:
            save_snapshot(len(rows))
        if _sha(source) != manifest["source_sha256"]:
            raise RuntimeError("source Phase14 checkpoint mutated during retrain")
        return _state_summary(root, cfg, ledger, start_update, reason=reason)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/experiments/ppo_five_hour.py" -->

## src/holdem_ai/experiments/ppo_safety.py

Original blob `56b2d6b81f74feef8967c51d5925ae6f7e88985e` (944 bytes).

<a name="py-171"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/experiments/ppo_safety.py" sha="56b2d6b81f74feef8967c51d5925ae6f7e88985e" -->

````````python
"""Phase 15 PPO stability safety policy.

Entropy below a fixed threshold is a warning, not proof of collapse.
KL breaches can still cause a hard stop after configured consecutive updates.
This function has no side effects and is testable independently.
"""
from __future__ import annotations


def safety_assessment(*, entropy: float, approx_kl: float,
                      entropy_threshold: float, max_approx_kl: float,
                      entropy_stop_enabled: bool, consecutive: int) -> dict:
    if type(entropy_stop_enabled) is not bool or consecutive < 0:
        raise ValueError("invalid safety settings")
    low_entropy = entropy < entropy_threshold
    high_kl = approx_kl > max_approx_kl
    safety_violation = high_kl or (low_entropy and entropy_stop_enabled)
    return {
        "low_entropy": low_entropy,
        "high_kl": high_kl,
        "consecutive_safety_warnings": consecutive + 1 if safety_violation else 0,
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/experiments/ppo_safety.py" -->

## src/holdem_ai/experiments/ppo_stress.py

Original blob `681320433c840a19bf87d0bb45fe61ab1c32d844` (30107 bytes).

<a name="py-172"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/experiments/ppo_stress.py" sha="681320433c840a19bf87d0bb45fe61ab1c32d844" -->

````````python
"""Phase16 long A0/A3 PPO stress test with transactional full-state resume.

No engine/GUI modifications. Never writes source checkpoint. The ORIGINAL
PPO optimize_ppo and Q-Critic are used for both arms; A3 adds an exact
post-update categorical KL gate with actor/critic/Adam state rollback.

This is an on-policy stability *stress experiment*, not a claim of stable
convergence or HUNL equilibrium. All evaluations are seed-blocked and
opponent-frozen. Every attempted update consumes its unique card-seed block,
including rejected A3 updates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.evaluation.initial_policy_comparison import _action_counts_on_rows
from holdem_ai.evaluation.seed_blocked import (
    EvaluationConfig, evaluate_seed_blocks, summarize_block_scores,
)
from holdem_ai.evaluation.terminal_reward_audit import summarize_terminal_distribution
from holdem_ai.experiments.longrun import _exclusive_lock
from holdem_ai.experiments.ppo_ablation import (
    critic_on_frozen_rows, exact_policy_kl, snapshot_probs,
)
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.training.ppo import PPOConfig, freeze_old_policy_data, optimize_ppo
from holdem_ai.training.ppo_session import load_checkpoint


@dataclass(frozen=True)
class StressConfig:
    source_checkpoint: str
    root: str
    variant: str
    device: str = "cpu"
    updates: int = 100                 # target additional attempts (NOT iterations since Phase14)
    hands_per_update: int = 128
    minibatch_size: int = 128
    epochs: int = 1
    env_batch_size: int = 32
    eval_every: int = 20
    eval_groups: int = 40              # 4 balanced games per seed group
    critic_eval_hands: int = 128
    rollout_seed: int = 680_000_000
    validation_seed: int = 78_000_000
    critic_seed: int = 690_000_000
    max_post_kl: float = 0.02
    hard_stop_kl: float = 0.10         # A0 diagnostic safety stop only
    save_every: int = 10
    resume: bool = False
    max_updates_this_call: int | None = None

    def __post_init__(self) -> None:
        if not self.source_checkpoint or not self.root:
            raise ValueError("source checkpoint and run root required")
        if self.variant not in ("a0", "a3"):
            raise ValueError("stress test supports only a0 and a3")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        for name in ("updates", "hands_per_update", "minibatch_size", "epochs",
                     "env_batch_size", "eval_every", "eval_groups",
                     "critic_eval_hands", "save_every"):
            value = getattr(self, name)
            if type(value) is not int or value < (2 if name == "eval_groups" else 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.env_batch_size > self.hands_per_update:
            raise ValueError("env_batch_size > hands_per_update")
        if self.max_updates_this_call is not None and (
            type(self.max_updates_this_call) is not int or self.max_updates_this_call < 1
        ):
            raise ValueError("max_updates_this_call must be positive or omitted")
        for name in ("rollout_seed", "validation_seed", "critic_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("max_post_kl", "hard_stop_kl"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or (
                not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be positive finite")
        if self.hard_stop_kl <= self.max_post_kl:
            raise ValueError("hard_stop_kl must exceed max_post_kl")
        ranges = (
            (self.rollout_seed, self.rollout_seed + self.updates * self.hands_per_update),
            (self.validation_seed, self.validation_seed + self.eval_groups),
            (self.critic_seed, self.critic_seed + self.critic_eval_hands),
        )
        if any(max(a0, b0) < min(a1, b1)
               for i, (a0, a1) in enumerate(ranges)
               for b0, b1 in ranges[i + 1:]):
            raise ValueError("rollout, evaluation and critic deck blocks overlap")


def _settings(cfg: StressConfig) -> dict:
    d = asdict(cfg)
    d.pop("resume")
    d.pop("max_updates_this_call")
    return d


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _checkpoint_payload(session, *, manifest_hash: str,
                        start_update: int, records: list[dict],
                        validations: list[dict]) -> dict:
    """Phase-9-compatible full PPO checkpoint with atomic stress metadata.

    load_checkpoint() accepts these additional keys using weights_only=True.
    Embedding records and evaluations makes the checkpoint the single source
    of truth when a process dies between saving it and writing JSON logs.
    """
    return {
        "phase": 9,
        "algorithm": "ppo_mc_privileged_baseline",
        "completed_updates": session.completed_updates,
        "network_config": asdict(session.network_config),
        "ppo_config": asdict(session.ppo_config),
        "actor_state": session.actor.state_dict(),
        "critic_state": session.critic.state_dict(),
        "actor_optimizer": session.actor_optimizer.state_dict(),
        "critic_optimizer": session.critic_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else []),
        "stress": {
            "protocol": "phase16_a0_a3_stress_v1",
            "manifest_sha256": manifest_hash,
            "start_update": start_update,
            "attempted_updates": len(records),
            "accepted_updates": sum(not r["rejected"] for r in records),
            "rejected_updates": sum(r["rejected"] for r in records),
            "records": records,
            "validations": validations,
        },
    }


def _save_full(path: Path, session, *, manifest_hash: str,
               start_update: int, records: list[dict],
               validations: list[dict]) -> None:
    data = _checkpoint_payload(
        session, manifest_hash=manifest_hash, start_update=start_update,
        records=records, validations=validations,
    )
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("xb") as f:
            torch.save(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _mirrors(root: Path, *, records: list[dict], validations: list[dict],
             start_update: int, target_updates: int) -> None:
    """Regenerate readable logs from the committed full-checkpoint ledger.

    JSON files are DERIVED mirrors. An interrupted write never changes the
    authoritative checkpoint; resuming re-materializes any missing rows.
    """
    path = root / "metrics.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps({"kind": "update", **record},
                               ensure_ascii=False, allow_nan=False) + "\n")
        for val in validations:
            f.write(json.dumps({"kind": "validation", **val},
                               ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _json_atomic(root / "progress.json", {
        "protocol": "phase16_a0_a3_stress_v1",
        "attempted": len(records),
        "accepted": sum(not r["rejected"] for r in records),
        "rejected": sum(r["rejected"] for r in records),
        "global_completed_updates": start_update + len(records),
        "target_attempted_updates": target_updates,
        "remaining_attempts": target_updates - len(records),
        "latest_validation_update": (
            validations[-1]["attempted"] if validations else None),
        "latest_post_update_kl": (records[-1]["post_update_exact_kl"]
                                  if records else None),
    })


def _baseline(source_actor, cfg: StressConfig, big_blind: int) -> dict:
    eval_cfg = EvaluationConfig(
        groups=cfg.eval_groups, seed=cfg.validation_seed,
        mode="sampled", device=cfg.device, big_blind=big_blind,
    )
    frozen = load_actor(cfg.source_checkpoint, device=cfg.device)
    baseline = {}
    for name, opponent in (("random_legal", None), ("frozen_phase14", frozen)):
        result = evaluate_seed_blocks(source_actor, opponent=opponent, config=eval_cfg)
        baseline[name] = {
            "summary": result.summary(),
            "block_mean_bb": list(result.means_bb),
            "opponent_id": result.opponent_id,
        }
    return baseline


def _validation(session, cfg: StressConfig, baseline: dict,
                attempted: int) -> dict:
    """Fixed held-out deck groups and unchanged opponents for every assessment."""
    eval_cfg = EvaluationConfig(
        groups=cfg.eval_groups, seed=cfg.validation_seed,
        mode="sampled", device=cfg.device,
        big_blind=session.ppo_config.big_blind,
    )
    frozen = load_actor(cfg.source_checkpoint, device=cfg.device)
    results = {}
    for name, opponent in (("random_legal", None), ("frozen_phase14", frozen)):
        benchmark = evaluate_seed_blocks(
            session.actor, opponent=opponent, config=eval_cfg)
        if benchmark.opponent_id != baseline[name]["opponent_id"]:
            raise ValueError("opponent fingerprint changed")
        changes = tuple(
            val - ref for val, ref in zip(
                benchmark.means_bb, baseline[name]["block_mean_bb"])
        )
        if len(changes) != cfg.eval_groups:
            raise AssertionError("mismatched evaluation blocks")
        diff = summarize_block_scores(changes)
        results[name] = {
            "summary": benchmark.summary(),
            "paired_vs_source": {
                "delta_mean_bb100": diff["mean_bb100"],
                "delta_ci95_lower_bb100": diff["ci95_lower_bb100"],
                "delta_ci95_upper_bb100": diff["ci95_upper_bb100"],
                "delta_se_bb100": diff["standard_error_bb100"],
            },
            "block_mean_bb": list(benchmark.means_bb),
            "opponent_id": benchmark.opponent_id,
        }
    # Critic/behavior check on NEW independently seeded cards, not training.
    critic_cfg = PPOConfig(
        seed=cfg.critic_seed,
        hands_per_update=cfg.critic_eval_hands,
        big_blind=session.ppo_config.big_blind,
    )
    holdout = collect_batched_self_play(
        session.actor, critic_cfg, iteration=0, device=cfg.device,
        env_batch_size=min(cfg.env_batch_size, cfg.critic_eval_hands),
    )
    calibration = critic_on_frozen_rows(
        session.actor, session.critic, holdout, variant="a0",
        scale=200., device=cfg.device,
    )
    terminal = summarize_terminal_distribution(holdout)
    counts = _action_counts_on_rows(holdout)
    behavior = {
        "hands": holdout.hands,
        "preflop_explicit_allin_fraction": round(
            terminal["preflop"]["explicit_all_in_hand_count"] / holdout.hands, 6),
        "preflop_allin_given_legal": counts["preflop"]["all_in_rate_given_legal"],
        "preflop_fold_end_fraction": round(
            terminal["preflop"]["fold_ended_count"] / holdout.hands, 6),
        "flop_decision_fraction": round(
            terminal["decision_coverage"]["flop_hands"] / holdout.hands, 6),
        "full_stack_fraction": terminal["seat0_hand_weighted_reward"]["near_full_stack_fraction"],
        "reward_std_bb": terminal["seat0_hand_weighted_reward"]["std_bb"],
    }
    return {
        "attempted": attempted,
        "global_update": session.completed_updates,
        "opponents": results,
        "critic_holdout": calibration,
        "behavior": behavior,
    }


def _one_update(session, cfg: StressConfig) -> dict:
    """Perform or reject one proposed update; consume its seeded rollout once."""
    step = session.completed_updates
    start = time.perf_counter()
    rows = collect_batched_self_play(
        session.actor, session.ppo_config,
        iteration=step, device=cfg.device, env_batch_size=cfg.env_batch_size,
    )
    frozen = freeze_old_policy_data(
        session.actor, session.critic, rows, device=cfg.device,
        chunk_size=session.ppo_config.inference_batch_size,
    )
    old = snapshot_probs(session.actor, rows, device=cfg.device)
    if cfg.variant == "a3":
        # Roll back optimizer moments as well as model parameters.
        before = (
            copy.deepcopy(session.actor.state_dict()),
            copy.deepcopy(session.critic.state_dict()),
            copy.deepcopy(session.actor_optimizer.state_dict()),
            copy.deepcopy(session.critic_optimizer.state_dict()),
        )
    stats = optimize_ppo(
        session.actor, session.critic, session.actor_optimizer,
        session.critic_optimizer, frozen, session.ppo_config,
        device=cfg.device,
        shuffle_seed=session.ppo_config.seed + 100003 * step,
    )
    kl = exact_policy_kl(
        session.actor, rows, old_probs=old, device=cfg.device)
    if not all(math.isfinite(float(x)) for x in stats.values()
               if isinstance(x, (float, int))) :
        raise FloatingPointError("nonfinite PPO diagnostic metric")
    if not math.isfinite(kl):
        raise FloatingPointError("nonfinite post-update KL")
    rejected = cfg.variant == "a3" and kl > cfg.max_post_kl
    if rejected:
        (actor_state, critic_state, actor_opt_state, critic_opt_state) = before
        session.actor.load_state_dict(actor_state)
        session.critic.load_state_dict(critic_state)
        session.actor_optimizer.load_state_dict(actor_opt_state)
        session.critic_optimizer.load_state_dict(critic_opt_state)
    terminal = summarize_terminal_distribution(rows)
    selected = _action_counts_on_rows(rows)
    session.completed_updates += 1    # ATTEMPTED, even when A3 rejected
    if cfg.device == "cuda":
        torch.cuda.synchronize()
    return {
        "attempted": None,            # completed index set by outer driver
        "global_update": session.completed_updates,
        "training_hands": rows.hands,
        "decisions": len(rows.actions),
        "post_update_exact_kl": round(kl, 8),
        "above_gate": bool(kl > cfg.max_post_kl),
        "rejected": bool(rejected),
        "preflop_explicit_allin_fraction": round(
            terminal["preflop"]["explicit_all_in_hand_count"] / rows.hands, 6),
        "flop_decision_fraction": round(
            terminal["decision_coverage"]["flop_hands"] / rows.hands, 6),
        "allin_given_legal": selected["preflop"]["all_in_rate_given_legal"],
        "full_stack_fraction": terminal["seat0_hand_weighted_reward"]["near_full_stack_fraction"],
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        **stats,
    }


def _check_state(payload: dict, manifest_hash: str, start_update: int,
                 cfg: StressConfig) -> tuple[list[dict], list[dict]]:
    extra = payload.get("stress")
    if not isinstance(extra, dict) or extra.get("protocol") != "phase16_a0_a3_stress_v1":
        raise ValueError("not a Phase16 stress checkpoint")
    if extra.get("manifest_sha256") != manifest_hash:
        raise ValueError("checkpoint manifest fingerprint mismatch")
    if extra.get("start_update") != start_update:
        raise ValueError("checkpoint start-update mismatch")
    records = extra["records"]
    validations = extra["validations"]
    if not isinstance(records, list) or not isinstance(validations, list):
        raise ValueError("corrupted stress records/validations")
    if len(records) > cfg.updates or payload.get("completed_updates") != start_update + len(records):
        raise ValueError("checkpoint update count inconsistent with records")
    if (extra.get("attempted_updates") != len(records)
            or extra.get("accepted_updates") != sum(not r["rejected"] for r in records)
            or extra.get("rejected_updates") != sum(r["rejected"] for r in records)):
        raise ValueError("checkpoint rejection accounting invalid")
    return records, validations


def run_stress(cfg: StressConfig) -> dict:
    """Run until target or per-call cap. Resume never repeats a committed update.

    All progress is stored atomically with full Torch PPO weights AND Adam
    states in latest.pt. JSON logs are merely rebuildable mirrors. A new
    snapshot/NNNN.pt is retained every save_every attempted updates.
    """
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source = Path(cfg.source_checkpoint).resolve(strict=True)
    root = Path(cfg.root).resolve()
    if source == root or root in source.parents or source.is_relative_to(root):
        raise ValueError("root and source checkpoint must be disjoint")
    if not cfg.resume and root.exists() and any(root.iterdir()):
        raise FileExistsError("fresh stress run requires an unused root")
    if cfg.resume and not (root / "run.json").is_file():
        raise FileNotFoundError("resume requires existing run.json")
    with _exclusive_lock(root):
        manifest_path = root / "run.json"
        latest = root / "latest.pt"
        spec = _settings(cfg)
        if cfg.resume:
            if (root / "hard_stop.json").is_file():
                raise RuntimeError("A0 hard-stop is active; do not resume before review")
            # A previous process may have died while writing an uncommitted temporary file.
            stale_tmp = latest.with_suffix(latest.suffix + ".tmp")
            if stale_tmp.exists():
                stale_tmp.unlink()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("settings") != spec:
                raise ValueError("resume configuration differs from run manifest")
            if manifest.get("source_sha256") != _sha(source):
                raise ValueError("source checkpoint changed")
            manifest_hash = _sha(manifest_path)
            if not latest.is_file():
                raise FileNotFoundError("missing full-state latest.pt")
            raw = torch.load(latest, map_location="cpu", weights_only=True)
            records, validations = _check_state(
                raw, manifest_hash, manifest["start_update"], cfg,
            )
            session = load_checkpoint(latest, device=cfg.device)
            if cfg.device == "cuda" and raw.get("cuda_rng_states"):
                torch.cuda.set_rng_state_all(raw["cuda_rng_states"])
            start_update = manifest["start_update"]
            baseline = manifest["source_baseline"]
        else:
            session = load_checkpoint(source, device=cfg.device)
            # Reset Critic Adam in both arms, matching the A0/A3 Phase16 ablation.
            session.critic_optimizer = torch.optim.Adam(
                session.critic.parameters(), lr=session.ppo_config.learning_rate)
            start_update = session.completed_updates
            session.ppo_config = replace(
                session.ppo_config,
                hands_per_update=cfg.hands_per_update,
                minibatch_size=cfg.minibatch_size,
                epochs=cfg.epochs,
                seed=cfg.rollout_seed - start_update * cfg.hands_per_update,
            )
            baseline = _baseline(
                load_actor(source, device=cfg.device),
                cfg, session.ppo_config.big_blind,
            )
            records, validations = [], []
            manifest = {
                "protocol": "phase16_a0_a3_stress_v1",
                "settings": spec,
                "source_sha256": _sha(source),
                "start_update": start_update,
                "source_baseline": baseline,
                "source_actor_frozen": True,
                "critic_adam_reset_both_arms": True,
            }
            _json_atomic(manifest_path, manifest)
            manifest_hash = _sha(manifest_path)
            _save_full(latest, session, manifest_hash=manifest_hash,
                       start_update=start_update, records=records, validations=validations)
        if _sha(source) != manifest["source_sha256"]:
            raise ValueError("source checkpoint changed")
        _mirrors(root, records=records, validations=validations,
                 start_update=start_update, target_updates=cfg.updates)
        # If power was lost after committing the update checkpoint but
        # before the scheduled evaluation, complete THAT evaluation first.
        # No rollout iteration is repeated.
        if records:
            n = len(records)
            eval_due = n % cfg.eval_every == 0 or n == cfg.updates
            if eval_due and not any(v["attempted"] == n for v in validations):
                val = _validation(session, cfg, baseline, n)
                validations.append(val)
                _save_full(latest, session, manifest_hash=manifest_hash,
                           start_update=start_update, records=records,
                           validations=validations)
                _mirrors(root, records=records, validations=validations,
                         start_update=start_update, target_updates=cfg.updates)
        if len(records) >= cfg.updates:
            return _finish(root, records, validations, cfg)
        started = time.perf_counter()
        limit = min(cfg.updates - len(records),
                    cfg.max_updates_this_call or cfg.updates)
        for _ in range(limit):
            row = _one_update(session, cfg)
            row["attempted"] = len(records) + 1
            records.append(row)
            _save_full(latest, session, manifest_hash=manifest_hash,
                       start_update=start_update, records=records, validations=validations)
            if len(records) % cfg.save_every == 0:
                snapshot = root / "snapshots" / f"attempt_{len(records):05d}.pt"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                if snapshot.exists():
                    raise FileExistsError(f"snapshot already exists: {snapshot}")
                # Snapshot is immutable and atomically published. A process
                # crash can leave only a disposable .tmp, never a partial .pt.
                tmp_snapshot = snapshot.with_suffix(".pt.tmp")
                try:
                    with tmp_snapshot.open("wb") as f:
                        with latest.open("rb") as src:
                            while part := src.read(1024 * 1024):
                                f.write(part)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_snapshot, snapshot)
                finally:
                    if tmp_snapshot.exists():
                        tmp_snapshot.unlink()
            _mirrors(root, records=records, validations=validations,
                     start_update=start_update, target_updates=cfg.updates)
            print(
                f"{cfg.variant} {len(records)}/{cfg.updates} "
                f"KL={row['post_update_exact_kl']:.6f} "
                f"gate={row['above_gate']} rejected={row['rejected']} "
                f"entropy={row['entropy']:.4f} "
                f"actor_grad={row['grad_norm_actor']:.4f}",
                flush=True,
            )
            if cfg.variant == "a0" and row["post_update_exact_kl"] > cfg.hard_stop_kl:
                _json_atomic(root / "hard_stop.json", {
                    "reason": "a0_hard_stop_post_update_kl",
                    "attempted": len(records),
                    "observed": row["post_update_exact_kl"],
                    "limit": cfg.hard_stop_kl,
                })
                raise RuntimeError(
                    "A0 emergency KL limit exceeded: progress saved; "
                    "do not resume automatically without reviewing the event")
            if len(records) % cfg.eval_every == 0 or len(records) == cfg.updates:
                val = _validation(session, cfg, baseline, len(records))
                validations.append(val)
                _save_full(latest, session, manifest_hash=manifest_hash,
                           start_update=start_update, records=records, validations=validations)
                _mirrors(root, records=records, validations=validations,
                         start_update=start_update, target_updates=cfg.updates)
                print(f"{cfg.variant} validation at {len(records)} "
                      f"delta_vs_source_frozen="
                      f"{val['opponents']['frozen_phase14']['paired_vs_source']['delta_mean_bb100']} "
                      f"criticEV={val['critic_holdout']['explained_variance']}",
                      flush=True)
        result = _finish(root, records, validations, cfg)
        result["elapsed_seconds_this_call"] = round(time.perf_counter() - started, 2)
        return result


def _finish(root: Path, records: list[dict], validations: list[dict],
            cfg: StressConfig) -> dict:
    result = {
        "protocol": "phase16_a0_a3_stress_v1",
        "variant": cfg.variant,
        "root": str(root),
        "attempted": len(records),
        "target_attempts": cfg.updates,
        "accepted": sum(not r["rejected"] for r in records),
        "rejected": sum(r["rejected"] for r in records),
        "max_post_update_kl": max((r["post_update_exact_kl"] for r in records),
                                  default=0.),
        "max_entropy_drop_warning_only": None,
        "training_hands": len(records) * cfg.hands_per_update,
        "validations": len(validations),
        "latest_checkpoint": str(root / "latest.pt"),
        "completed": len(records) == cfg.updates,
        "note": "Short or long on-policy PPO stress observations do NOT prove GTO "
                "or stable convergence. Rejected A3 attempts consume seed blocks.",
    }
    _json_atomic(root / "summary.json", result)
    return result


def compare_stress_runs(root_a0: str | Path, root_a3: str | Path) -> dict:
    """Paired comparisons using the *same* held-out deck/opponent fingerprints."""
    roots = {"a0": Path(root_a0), "a3": Path(root_a3)}
    manifests = {k: json.loads((v / "run.json").read_text(encoding="utf-8"))
                 for k, v in roots.items()}
    records = {}
    for arm, root in roots.items():
        latest = torch.load(root / "latest.pt", map_location="cpu", weights_only=True)
        ledger = latest["stress"]
        if ledger["manifest_sha256"] != _sha(root / "run.json"):
            raise ValueError("stress manifest does not match checkpoint")
        records[arm] = ledger
    s0 = manifests["a0"]["settings"]
    s3 = manifests["a3"]["settings"]
    for key in ("source_checkpoint", "updates", "hands_per_update", "epochs",
                "minibatch_size", "rollout_seed", "validation_seed",
                "critic_seed", "eval_groups", "critic_eval_hands"):
        if s0[key] != s3[key]:
            raise ValueError(f"cannot pair runs: {key} differs")
    if manifests["a0"]["source_sha256"] != manifests["a3"]["source_sha256"]:
        raise ValueError("different source checkpoint SHA")
    vals0 = {r["attempted"]: r for r in records["a0"]["validations"]}
    vals3 = {r["attempted"]: r for r in records["a3"]["validations"]}
    checkpoints = {}
    for n in sorted(set(vals0) & set(vals3)):
        each = {}
        for opponent in ("random_legal", "frozen_phase14"):
            a = vals0[n]["opponents"][opponent]
            b = vals3[n]["opponents"][opponent]
            if a["opponent_id"] != b["opponent_id"]:
                raise ValueError("different opponent fingerprint")
            p = a["block_mean_bb"]
            q = b["block_mean_bb"]
            if len(p) != s0["eval_groups"] or len(q) != len(p):
                raise ValueError("evaluation group mismatch")
            paired = summarize_block_scores(tuple(y - x for x, y in zip(p, q)))
            each[opponent] = {
                "a3_minus_a0_delta_bb100": paired["mean_bb100"],
                "ci95_lower_bb100": paired["ci95_lower_bb100"],
                "ci95_upper_bb100": paired["ci95_upper_bb100"],
                "se_bb100": paired["standard_error_bb100"],
                "a0_mean_bb100": a["summary"]["mean_bb100"],
                "a3_mean_bb100": b["summary"]["mean_bb100"],
            }
        checkpoints[str(n)] = each
    return {
        "protocol": "phase16_paired_a0_a3_validation",
        "a0_attempted": records["a0"]["attempted_updates"],
        "a3_attempted": records["a3"]["attempted_updates"],
        "a0_rejected": records["a0"]["rejected_updates"],
        "a3_rejected": records["a3"]["rejected_updates"],
        "eval_seed_start": s0["validation_seed"],
        "groups": s0["eval_groups"],
        "paired_validations": checkpoints,
        "limitations": "CI is per independent deck seed block; repeated "
                       "validation across updates is NOT independent, and "
                       "watching/choosing best checkpoint induces selection bias.",
    }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/experiments/ppo_stress.py" -->

## src/holdem_ai/experiments/time_budget.py

Original blob `3d70921f43f9e02cc578f1f1c5987b40d6986bbb` (15737 bytes).

<a name="py-173"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/experiments/time_budget.py" sha="3d70921f43f9e02cc578f1f1c5987b40d6986bbb" -->

````````python
"""Safe wall-clock bounded Phase11 training with crash-recoverable observability.

All optimizer updates are on-policy; rollout is batched *only across current
observations*, never reused once the actor changes. Single-process GPU batching
is not multiprocessing or a distributed replay buffer.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.training.ppo import PPOConfig, freeze_old_policy_data, optimize_ppo
from holdem_ai.training.ppo_session import create_session, load_checkpoint, save_checkpoint
from holdem_ai.training.qboost import QBoostConfig, freeze_qboost_data, optimize_qboost
from holdem_ai.training.qboost_session import create_qboost_session, from_phase9_checkpoint, load_qboost_checkpoint, save_qboost_checkpoint
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.evaluation.balanced import evaluate_balanced


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: str = 'qboost'
    device: str = 'cpu'
    duration_seconds: float = 5 * 3600
    max_updates: int | None = None
    env_batch_size: int = 8
    eval_every: int = 25
    eval_hands: int = 80
    hands_per_update: int = 32
    epochs: int = 2
    minibatch_size: int = 32
    seed: int = 42
    beta: float = 1.0
    checkpoint: str = 'checkpoints/phase11_qboost.pt'
    log_dir: str = 'runs/phase11_qboost'
    resume: str | None = None
    from_phase9: str | None = None
    from_phase10: str | None = None
    small_network: bool = False
    max_decisions: int = 200

    def __post_init__(self) -> None:
        if self.algorithm not in ('ppo', 'qboost'):
            raise ValueError('algorithm must be ppo or qboost')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        if isinstance(self.duration_seconds, bool) or not isinstance(self.duration_seconds, (float, int)) or not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError('duration_seconds must be finite and positive')
        for field in ('env_batch_size', 'eval_every', 'hands_per_update', 'epochs',
                      'minibatch_size', 'max_decisions', 'eval_hands'):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f'{field} must be a positive integer')
        if self.eval_hands < 4 or self.eval_hands % 4:
            raise ValueError('eval_hands must be a multiple of four and >= 4')
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates < 1):
            raise ValueError('max_updates must be None or positive integer')
        if type(self.seed) is not int:
            raise ValueError('seed must be int')
        if not isinstance(self.beta, (float, int)) or isinstance(self.beta, bool) or not math.isfinite(self.beta) or not 0 <= self.beta <= 1:
            raise ValueError('beta must be in [0,1]')
        if sum(bool(x) for x in (self.resume, self.from_phase9, self.from_phase10)) > 1:
            raise ValueError('--resume, --from-phase9, --from-phase10 are mutually exclusive')
        if self.from_phase10 and self.algorithm != 'qboost':
            raise ValueError('--from-phase10 works only for qboost')
        if not self.checkpoint or not self.log_dir:
            raise ValueError('checkpoint and log directory are required')


def _write_atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _log(log_path: Path, record: dict, emit: Callable[[dict], None] | None = None) -> None:
    with log_path.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    if emit is not None:
        emit(record)


def _session(config: ExperimentConfig):
    if config.resume:
        return (load_checkpoint(config.resume, device=config.device) if config.algorithm == 'ppo'
                else load_qboost_checkpoint(config.resume, device=config.device))
    if config.from_phase10:
        loaded = load_qboost_checkpoint(config.from_phase10, device=config.device)
        if not math.isclose(loaded.qconfig.beta, config.beta):
            raise ValueError('from-phase10 beta must match the checkpoint beta')
        return loaded
    if config.from_phase9:
        if config.algorithm == 'ppo':
            return load_checkpoint(config.from_phase9, device=config.device)
        return from_phase9_checkpoint(config.from_phase9, device=config.device,
                                      qconfig=QBoostConfig(beta=config.beta))
    network = (NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                             ff_dim=64, hidden_dim=48, max_tokens=256)
               if config.small_network else NetworkConfig())
    ppo = PPOConfig(hands_per_update=config.hands_per_update,
                    epochs=config.epochs, minibatch_size=config.minibatch_size,
                    max_decisions=config.max_decisions, seed=config.seed)
    if config.algorithm == 'ppo':
        return create_session(device=config.device, seed=config.seed,
                              network_config=network, ppo_config=ppo)
    return create_qboost_session(device=config.device, seed=config.seed,
                                 network_config=network, ppo_config=ppo,
                                 qconfig=QBoostConfig(beta=config.beta))


def _base(session, algorithm: str):
    return session if algorithm == 'ppo' else session.base


def _checkpoint_save(session, algorithm: str, path: Path) -> None:
    if algorithm == 'ppo':
        save_checkpoint(session, path)
    else:
        save_qboost_checkpoint(session, path)


def _step(session, config: ExperimentConfig) -> dict:
    base = _base(session, config.algorithm)
    start = time.perf_counter()
    iteration = base.completed_updates
    rows = collect_batched_self_play(
        base.actor, base.ppo_config, iteration=iteration, device=base.device,
        env_batch_size=config.env_batch_size,
    )
    if config.algorithm == 'ppo':
        frozen = freeze_old_policy_data(base.actor, base.critic, rows,
                                        device=base.device,
                                        chunk_size=base.ppo_config.inference_batch_size)
        metrics = optimize_ppo(
            base.actor, base.critic, base.actor_optimizer,
            base.critic_optimizer, frozen, base.ppo_config,
            device=base.device,
            shuffle_seed=base.ppo_config.seed + 100003 * iteration,
        )
    else:
        frozen = freeze_qboost_data(base.actor, base.critic, rows,
                                    device=base.device,
                                    chunk_size=base.ppo_config.inference_batch_size)
        metrics = optimize_qboost(
            base.actor, base.critic, base.actor_optimizer,
            base.critic_optimizer, frozen, base.ppo_config, session.qconfig,
            device=base.device,
            shuffle_seed=base.ppo_config.seed + 100003 * iteration,
        )
        metrics['beta'] = session.qconfig.beta
    if base.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    base.completed_updates += 1
    return {
        'kind': 'update', 'algorithm': config.algorithm,
        'update': base.completed_updates, 'hands': rows.hands,
        'elapsed_seconds': round(elapsed, 4),
        'hands_per_second_training': round(rows.hands / max(elapsed, 1e-9), 3),
        **metrics,
    }


def _load_or_create_anchor(base, *, path: Path, resume: bool, device: str):
    anchor = TransformerActor(base.network_config).to(device)
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f'frozen anchor missing: {path}')
        payload = torch.load(path, weights_only=True, map_location='cpu')
        if payload.get('network_config') != asdict(base.network_config):
            raise ValueError('anchor architecture mismatch')
        anchor.load_state_dict(payload['actor_state'])
    else:
        anchor.load_state_dict(base.actor.state_dict())
        tmp = path.with_name(path.name + '.tmp')
        torch.save({'network_config': asdict(base.network_config),
                    'actor_state': {k: v.cpu().clone() for k, v in base.actor.state_dict().items()}}, tmp)
        os.replace(tmp, path)
    anchor.eval().requires_grad_(False)
    return anchor


def _budget_remaining(deadline: float, last_update_seconds: float | None = None) -> bool:
    margin = 0.0 if last_update_seconds is None else max(0.02, 1.1 * last_update_seconds)
    return time.perf_counter() + margin < deadline


def run_experiment(config: ExperimentConfig, *,
                   emit: Callable[[dict], None] | None = None) -> dict:
    """Run until time budget or maximum extra updates; checkpoint after every update.

    Uses an *immutable frozen anchor* for all evaluations across resume. A run
    directory is exclusive: fresh runs never overwrite existing run metadata.
    Time limits apply at update/evaluation boundaries, not inside a poker hand.
    """
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable on this machine')
    checkpoint = Path(config.checkpoint)
    root = Path(config.log_dir)
    meta_path, anchor_path = root / 'run.json', root / 'anchor.pt'
    events_path, summary_path = root / 'metrics.jsonl', root / 'summary.json'
    resuming = config.resume is not None
    if resuming:
        if not meta_path.is_file() or not events_path.is_file():
            raise FileNotFoundError('resume requires existing --log-dir/run.json and metrics.jsonl')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('algorithm') != config.algorithm:
            raise ValueError('resume algorithm does not match log directory')
        if Path(meta['checkpoint']).resolve() != Path(config.resume).resolve() or checkpoint.resolve() != Path(config.resume).resolve():
            raise ValueError('resume must use the SAME checkpoint path in --checkpoint and --resume')
    else:
        if (root.exists() and any(root.iterdir())) or checkpoint.exists():
            raise FileExistsError('fresh run would overwrite log/checkpoint; choose unique paths or --resume')
        root.mkdir(parents=True, exist_ok=True)
    session = _session(config)
    base = _base(session, config.algorithm)
    if config.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if resuming:
        if (asdict(base.network_config) != meta['network_config']
                or asdict(base.ppo_config) != meta['ppo_config']):
            raise ValueError('resume checkpoint architecture/config mismatch')
        if config.algorithm == 'qboost' and not math.isclose(session.qconfig.beta, meta['beta']):
            raise ValueError('resume beta differs from original run')
        if base.completed_updates < meta['initial_updates']:
            raise ValueError('checkpoint update count older than run start')
    else:
        meta = {
            'phase': 11, 'algorithm': config.algorithm,
            'checkpoint': str(checkpoint.resolve()),
            'initial_updates': base.completed_updates,
            'network_config': asdict(base.network_config),
            'ppo_config': asdict(base.ppo_config),
            'beta': session.qconfig.beta if config.algorithm == 'qboost' else 0.0,
            'config': asdict(config),
        }
        _write_atomic_json(meta_path, meta)
    anchor = _load_or_create_anchor(base, path=anchor_path,
                                    resume=resuming, device=config.device)
    if not resuming:
        _checkpoint_save(session, config.algorithm, checkpoint)
    started = time.perf_counter()
    deadline = started + config.duration_seconds
    updates_this_run = 0
    last_seconds = None
    evaluations = 0
    reason = 'duration_limit'
    _log(events_path, {'kind': 'resume' if resuming else 'start',
                       'completed_updates': base.completed_updates,
                       'duration_seconds': config.duration_seconds,
                       'env_batch_size': config.env_batch_size}, emit)

    def evaluate_now():
        nonlocal evaluations
        for opp in (None, anchor):
            # Same seeds and four layout combinations at every evaluation.
            report = evaluate_balanced(
                base.actor, opponent=opp, hands=config.eval_hands,
                seed=1_200_000 + base.ppo_config.seed * 10,
                device=config.device, big_blind=base.ppo_config.big_blind,
            )
            _log(events_path, {'kind': 'evaluation', 'update': base.completed_updates,
                               'algorithm': config.algorithm, **report}, emit)
            evaluations += 1

    interrupted = False
    try:
        if not resuming and _budget_remaining(deadline):
            evaluate_now()
        while True:
            if config.max_updates is not None and updates_this_run >= config.max_updates:
                reason = 'update_limit'
                break
            if not _budget_remaining(deadline, last_seconds):
                break
            report = _step(session, config)
            # Atomic checkpoint FIRST; a missed log can be recovered using update counter.
            _checkpoint_save(session, config.algorithm, checkpoint)
            _log(events_path, report, emit)
            updates_this_run += 1
            last_seconds = float(report['elapsed_seconds'])
            if (base.completed_updates % config.eval_every == 0
                    and _budget_remaining(deadline)):
                evaluate_now()
        if (updates_this_run > 0 and base.completed_updates % config.eval_every != 0
                and _budget_remaining(deadline)):
            evaluate_now()
    except KeyboardInterrupt:
        interrupted = True
        reason = 'keyboard_interrupt'
        # Optimizer checkpoint is last completed update only. A mid-update SIGINT
        # must NOT overwrite it with partially updated, inconsistent parameters.
        _log(events_path, {'kind': 'interrupt', 'completed_updates_on_disk':
                           int(torch.load(checkpoint, map_location='cpu', weights_only=True)['completed_updates'])}, emit)
    elapsed = time.perf_counter() - started
    summary = {
        'phase': 11, 'algorithm': config.algorithm, 'device': config.device,
        'stop_reason': reason, 'interrupted': interrupted,
        'updates_this_run': updates_this_run,
        'completed_updates': int(torch.load(checkpoint, map_location='cpu',
                                            weights_only=True)['completed_updates']),
        'elapsed_seconds': round(elapsed, 4),
        'evaluations': evaluations, 'checkpoint': str(checkpoint),
        'log_dir': str(root), 'env_batch_size': config.env_batch_size,
        'cuda_peak_allocated_mb': (round(torch.cuda.max_memory_allocated()/2**20, 3)
                                   if config.device == 'cuda' else None),
    }
    _write_atomic_json(summary_path, summary)
    _log(events_path, {'kind': 'end', **summary}, emit)
    return summary

````````

<!-- END_PY_SOURCE path="src/holdem_ai/experiments/time_budget.py" -->

## src/holdem_ai/game_service/__init__.py

Original blob `dc375ea04dd77f2a577c11893d7edcf20339a732` (182 bytes).

<a name="py-174"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/game_service/__init__.py" sha="dc375ea04dd77f2a577c11893d7edcf20339a732" -->

````````python
"""Server-side HUNL human-vs-agent integration (GUI-1.0)."""

from .session_manager import HUNLGameSession, HUNLSessionRegistry

__all__ = ["HUNLGameSession", "HUNLSessionRegistry"]

````````

<!-- END_PY_SOURCE path="src/holdem_ai/game_service/__init__.py" -->

## src/holdem_ai/game_service/action_adapter.py

Original blob `366dc8f00cc55b8f2d3054b8e98df019e96cfc6c` (2708 bytes).

<a name="py-175"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/game_service/action_adapter.py" sha="366dc8f00cc55b8f2d3054b8e98df019e96cfc6c" -->

````````python
"""Adapt the existing seven-action HUNL200 policy head to GUI action offers.

IMPORTANT: policy indices are NOT the Phase-3 engine action IDs. A browser
sends a seven-head index; only HUNL200Adapter is allowed to execute it.
Arbitrary human raise-to amounts are deferred to GUI-1.3, where the adapter
will need an explicit public method rather than mutating its private fields.
"""
from __future__ import annotations

import json
from typing import Any

from holdem_ai.core.schema import InformationState
from holdem_ai.environment.bet_abstraction import (
    BET_ACTION_LABELS,
    NoLimitBetContext,
    abstract_bet_actions,
)


def legal_gui_actions(observation: InformationState, engine: Any) -> list[dict[str, Any]]:
    """Provide legal labels and exact engine-validated raise-to targets.

    Only invoked for the acting HUMAN, using the same observable information
    available to the actor; never consumes an opponent's hidden hole cards.
    A mask mismatch is an integration error, not a reason to guess a bet size.
    """
    if engine.is_over() or engine.current_player != observation.player_id:
        raise ValueError("GUI action choices require the acting player's turn")
    raw = engine.observe(observation.player_id)["raw_obs"]
    i = observation.player_id
    engine_actions = set(engine.legal_actions)
    can_raise = 4 in engine_actions
    committed = int(raw["street_bets"][i])
    stack = int(raw["stacks"][i])
    to_call = int(raw["to_call"])
    ctx = NoLimitBetContext(
        pot=int(raw["pot"]),
        committed=committed,
        to_call=to_call,
        stack=stack,
        min_raise_to=int(raw["min_raise_to"]) if can_raise else None,
        max_raise_to=committed + stack if can_raise else None,
        can_fold=0 in engine_actions,
        can_check=1 in engine_actions and to_call == 0,
        can_call=1 in engine_actions and to_call > 0,
        can_raise=can_raise,
    )
    choices = abstract_bet_actions(ctx)
    if choices.mask != observation.legal_action_mask:
        raise AssertionError("GUI and agent legal-action masks disagree")
    return [
        {
            "index": index,
            "label": BET_ACTION_LABELS[index],
            "kind": action.kind,
            "raise_to": action.raise_to if action.kind == "raise_to" else (
                committed + stack if action.kind == "all_in" else None
            ),
        }
        for index, action in enumerate(choices.actions)
        if action is not None
    ]


def validate_human_action(observation: InformationState, action_index: int) -> int:
    """Validate without trusting a browser-supplied legal mask."""
    observation.assert_legal(action_index)
    return action_index

````````

<!-- END_PY_SOURCE path="src/holdem_ai/game_service/action_adapter.py" -->

## src/holdem_ai/game_service/bot_adapter.py

Original blob `85a324cd1d84cd5dbb4d1402095dffeccd72b0da` (787 bytes).

<a name="py-176"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/game_service/bot_adapter.py" sha="85a324cd1d84cd5dbb4d1402095dffeccd72b0da" -->

````````python
"""Apply any Phase-5 AgentPolicy to a bot turn, including NeuralActorPolicy.

The AgentPolicy receives only HUNL200Adapter's acting-player InformationState,
never an engine reference, privileged CriticState, or human hole cards.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from holdem_ai.core.schema import InformationState


class GUIBotPolicy(Protocol):
    def act(self, observation: InformationState, rng: random.Random) -> int: ...


@dataclass(slots=True)
class BotActionAdapter:
    policy: GUIBotPolicy
    rng: random.Random

    def choose(self, observation: InformationState) -> int:
        action = self.policy.act(observation, self.rng)
        observation.assert_legal(action)
        return action

````````

<!-- END_PY_SOURCE path="src/holdem_ai/game_service/bot_adapter.py" -->

## src/holdem_ai/game_service/checkpoint_catalog.py

Original blob `73de908a356ba415106f72b1ecd722b9749dafe6` (4986 bytes).

<a name="py-177"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/game_service/checkpoint_catalog.py" sha="73de908a356ba415106f72b1ecd722b9749dafe6" -->

````````python
"""Trusted local checkpoint selection for GUI-1.4 HUNL evaluation.

Clients choose an ID, never a checkpoint path. The GUI neither accepts uploaded
weights nor downloads weights, and never exposes local filesystem paths.
Torch / the Phase-16 loader are imported only when a neural session is made.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Mapping

from holdem_ai.agents.diagnostic_opponents import build_builtin

BUILTIN_OPPONENTS = ("random", "calling_station", "tight", "pot_pressure")

MODEL_PATH_VARIABLES = {
    "ppo": "HOLDEM_AI_PPO_CHECKPOINT",
    "qboost": "HOLDEM_AI_QBOOST_CHECKPOINT",
}
LABELS = {
    "random": "Random policy",
    "calling_station": "Calling station",
    "tight": "Tight policy",
    "pot_pressure": "Pot pressure",
    "ppo": "Trained PPO",
    "qboost": "Trained Q-Boost PPO",
}


@dataclass(frozen=True, slots=True)
class BotInfo:
    id: str
    label: str
    trained: bool
    available: bool
    message: str

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "trained": self.trained,
                "available": self.available, "message": self.message}


class CheckpointCatalog:
    """Two server-configured, trusted model slots plus four baselines.

    One policy object is shared across game sessions; each session owns an
    independent RNG. NeuralActorPolicy holds a frozen eval-only Actor, so
    no training, optimizer update, or policy mutation occurs in this service.
    """

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        env = dict(os.environ if environment is None else environment)
        self.paths = {key: env.get(var, "").strip() for key, var in MODEL_PATH_VARIABLES.items()}
        self.device = env.get("HOLDEM_AI_GUI_DEVICE", "cpu").strip().lower()
        self.mode = env.get("HOLDEM_AI_GUI_POLICY_MODE", "sampled").strip().lower()
        if self.device not in ("cpu", "cuda"):
            raise ValueError("HOLDEM_AI_GUI_DEVICE must be cpu or cuda")
        if self.mode not in ("sampled", "greedy"):
            raise ValueError("HOLDEM_AI_GUI_POLICY_MODE must be sampled or greedy")
        self._lock = threading.RLock()
        self._cached: dict[str, tuple[tuple[str, int, int], object]] = {}

    def _model_file(self, name: str) -> Path | None:
        raw = self.paths[name]
        return Path(raw).expanduser().resolve() if raw else None

    def options(self) -> list[dict]:
        results = [BotInfo(name, LABELS[name], False, True, "Ready to play").as_dict()
                   for name in BUILTIN_OPPONENTS]
        for name in MODEL_PATH_VARIABLES:
            path = self._model_file(name)
            exists = bool(path and path.is_file())
            message = ("Checkpoint found; compatibility checked when the game starts"
                       if exists else "Checkpoint file not configured or missing")
            results.append(BotInfo(name, LABELS[name], True, exists, message).as_dict())
        return results

    def build_policy(self, name: str):
        if name in BUILTIN_OPPONENTS:
            return build_builtin(name)
        if name not in MODEL_PATH_VARIABLES:
            raise ValueError("Unknown bot selection")
        path = self._model_file(name)
        if path is None or not path.is_file():
            raise ValueError(f"{LABELS[name]} checkpoint is not configured or missing. Configure {MODEL_PATH_VARIABLES[name]}.")
        try:
            metadata = path.stat()
            key = (str(path), metadata.st_mtime_ns, metadata.st_size)
        except OSError as exc:
            raise ValueError("Unable to inspect configured checkpoint") from exc
        with self._lock:
            if name in self._cached and self._cached[name][0] == key:
                return self._cached[name][1]
            try:
                # Phase 16's shared loader uses torch.load(weights_only=True),
                # NetworkConfig, and strict Actor weights. No Python pickle load.
                from holdem_ai.evaluation.checkpoint_loader import load_actor
                from holdem_ai.agents.neural_policy import NeuralActorPolicy

                actor = load_actor(path, device=self.device)
                if actor.config.dropout != 0:
                    raise ValueError("evaluation model must have dropout=0")
                policy = NeuralActorPolicy(actor, device=self.device,
                                           big_blind=100, greedy=self.mode == "greedy")
            except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Could not load {LABELS[name]} checkpoint. Confirm it is a trusted, "
                    "compatible Phase 9-15 Actor file and the required PyTorch device is available."
                ) from exc
            self._cached[name] = (key, policy)
            return policy

````````

<!-- END_PY_SOURCE path="src/holdem_ai/game_service/checkpoint_catalog.py" -->

## src/holdem_ai/game_service/interactive_adapter.py

Original blob `00e4ad6ba74d2f2eedd1031bf8b22078a5337281` (2755 bytes).

<a name="py-178"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/game_service/interactive_adapter.py" sha="00e4ad6ba74d2f2eedd1031bf8b22078a5337281" -->

````````python
"""Human-only exact bet sizing over the trained HUNL200 seven-action adapter.

The neural actor still receives the same 7-action InformationState.  A human
raise is recorded using the existing observable raise_to history grammar so
future bot observations contain the public betting action and exact amount.
This does not mutate or replace the Phase-6 HUNL200Adapter implementation.
"""
from __future__ import annotations

from holdem_ai.core.schema import Terminal, Turn
from holdem_ai.environment.bet_abstraction import EngineBetAction
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter


class InteractiveHUNLAdapter(HUNL200Adapter):
    def step_human_raise_to(self, amount: int) -> Turn | Terminal:
        engine = self.engine
        if type(amount) is not int:
            raise ValueError("raise-to amount must be an integer")
        if engine.is_over() or not hasattr(self, "_observation") or self._observation is None:
            raise RuntimeError("no active decision")
        actor, street = engine.current_player, engine.street
        if self._observation.player_id != actor:
            raise RuntimeError("observation must match acting player")
        # The Phase-3 engine validates the full raise, a permitted short all-in,
        # betting-reopen rules and stack maximum BEFORE any state is committed.
        engine.step_raise_to(amount)
        self.last_engine_action = EngineBetAction("raise_to", amount)
        self._history.append(f"{street}|seat={actor}|raise_to|raise_to={amount}")
        return self._receive()

    def reset_with_stacks(self, stacks: tuple[int, int], *, seed: int | None = None) -> Turn | Terminal:
        """GUI-only hand initialization with chips carried from previous hand.

        The Phase-6 HUNL200Adapter.reset() is deliberately NOT modified, so
        training and Phase-16 evaluations still start with two 200BB stacks.
        """
        from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine

        if (not isinstance(stacks, tuple) or len(stacks) != 2
                or any(type(value) is not int or value <= 0 for value in stacks)):
            raise ValueError("continuous hands require two strictly positive integer stacks")
        if sum(stacks) != 2 * self.stack_bb * self.big_blind:
            raise ValueError("continuous match chip conservation violated")
        self._engine = MultiPlayerHandEngine(
            stacks=stacks,
            small_blind=self.small_blind, big_blind=self.big_blind,
            button=self.button, seed=seed,
        )
        self._history = []
        self.last_engine_action = None
        self._choices = None
        self._observation = None
        self._engine.reset()
        return self._receive()

````````

<!-- END_PY_SOURCE path="src/holdem_ai/game_service/interactive_adapter.py" -->

## src/holdem_ai/game_service/session_analytics.py

Original blob `fdfe71bc594e8d344af4f1731cbfb7ddfd76ed8e` (5496 bytes).

<a name="py-179"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/game_service/session_analytics.py" sha="fdfe71bc594e8d344af4f1731cbfb7ddfd76ed8e" -->

````````python
"""GUI-1.5: bounded, hero-visible completed-hand ledger and online metrics.

Only receive the already-sanitized project_human_view() result. The analytics
layer never accesses deck, critic state or the opponent's unrevealed cards.
Statistics cover ALL completed hands; detailed records are capped in memory.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
from typing import Any


class SessionAnalytics:
    def __init__(self, *, history_limit: int = 500) -> None:
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError('history_limit must be a positive integer')
        self._records: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self.count = 0
        self.wins = 0
        self.losses = 0
        self.break_even = 0
        self.showdowns = 0
        self.total_chips = 0
        self.total_bb = 0.0
        self.mean_bb = 0.0
        self.m2_bb = 0.0
        self.peak_bb = 0.0
        self.max_drawdown_bb = 0.0

    def append_completed(self, human_view: dict[str, Any]) -> None:
        """Accept exactly one terminal, sanitized view per hand."""
        result = human_view.get('hand_result')
        if human_view.get('phase') != 'complete' or result is None:
            raise ValueError('analytics only accepts completed hands')
        hero = human_view['hero_seat']
        chips = int(result['payoffs_chips'][hero])
        bb = float(result['payoffs_bb'][hero])
        if not math.isfinite(bb) or not math.isclose(bb * human_view['big_blind'], chips, abs_tol=1e-5):
            raise ValueError('inconsistent payoff units')
        if result['opponent_cards'] is not None and not result['showdown']:
            raise ValueError('folded opponent cards must remain hidden')
        expected_hand = self.count + 1
        if human_view['hand_number'] != expected_hand:
            raise ValueError('completed hand numbers must be contiguous')
        outcome = 'win' if chips > 0 else 'loss' if chips < 0 else 'even'
        self.count += 1
        self.wins += outcome == 'win'
        self.losses += outcome == 'loss'
        self.break_even += outcome == 'even'
        self.showdowns += bool(result['showdown'])
        self.total_chips += chips
        self.total_bb += bb
        difference = bb - self.mean_bb
        self.mean_bb += difference / self.count
        self.m2_bb += difference * (bb - self.mean_bb)
        self.peak_bb = max(self.peak_bb, self.total_bb)
        self.max_drawdown_bb = max(self.max_drawdown_bb, self.peak_bb - self.total_bb)
        record = {
            'hand_number': self.count,
            'starting_stacks': list(human_view.get('starting_stacks', [200 * human_view['big_blind']] * 2)),
            'button': human_view['button'],
            'street': human_view['street'],
            'big_blind': human_view['big_blind'],
            'small_blind': human_view['small_blind'],
            'hero_cards': list(human_view['hero_cards']),
            'board': list(human_view['board']),
            'action_log': deepcopy(human_view['action_log']),
            'hero_net_chips': chips,
            'hero_net_bb': bb,
            'cumulative_bb': self.total_bb,
            'outcome': outcome,
            'result': deepcopy(result),
        }
        self._records.append(record)

    def summary(self) -> dict[str, Any]:
        n = self.count
        # A simple normal-approximation interval, not a guarantee of independent
        # samples or statistical significance against a general opponent pool.
        deviation = math.sqrt(max(0.0, self.m2_bb) / (n - 1)) if n > 1 else None
        radius = 1.96 * deviation / math.sqrt(n) * 100 if deviation is not None else None
        return {
            'completed_hands': n,
            'wins': self.wins,
            'losses': self.losses,
            'break_even': self.break_even,
            'showdowns': self.showdowns,
            'net_chips': self.total_chips,
            'net_bb': self.total_bb,
            'bb_per_100': 100 * self.total_bb / n if n else 0.0,
            'mean_bb_per_hand': self.mean_bb if n else 0.0,
            'std_bb_per_hand': deviation,
            'ci95_lower_bb100': 100 * self.mean_bb - radius if radius is not None else None,
            'ci95_upper_bb100': 100 * self.mean_bb + radius if radius is not None else None,
            'max_drawdown_bb': self.max_drawdown_bb,
            'retained_hand_records': len(self._records),
            'records_truncated': n > len(self._records),
            'note': ('Descriptive local-session results. Approximate hand-level 95% CI assumes '
                     'independent hands; not a GTO, exploitability or population-strength estimate.'),
        }

    def history(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        if type(offset) is not int or offset < 0:
            raise ValueError('offset must be a nonnegative integer')
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('limit must be between 1 and 100')
        records = list(reversed(self._records))
        items = deepcopy(records[offset:offset + limit])
        return {
            'items': items,
            'offset': offset,
            'limit': limit,
            'total': len(records),
            'completed_hands': self.count,
            'has_more': offset + len(items) < len(records),
            'records_truncated': self.count > len(records),
        }

````````

<!-- END_PY_SOURCE path="src/holdem_ai/game_service/session_analytics.py" -->

## src/holdem_ai/game_service/session_manager.py

Original blob `4df2f8bc73feeed3e8c6922e5bff5caaa8278512` (11214 bytes).

<a name="py-180"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/game_service/session_manager.py" sha="4df2f8bc73feeed3e8c6922e5bff5caaa8278512" -->

````````python
"""Local HUNL200 human-vs-agent game sessions; transport agnostic.

No server or React dependency. Training remains independent 200BB; local
GUI sessions may carry stacks between hands. Synchronous revision-checked play.
"""
from __future__ import annotations

import random
import secrets
import threading
from typing import Any

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.core.schema import Terminal, Turn
from .interactive_adapter import InteractiveHUNLAdapter

from .action_adapter import validate_human_action
from .bot_adapter import BotActionAdapter, GUIBotPolicy
from .state_projection import project_human_view
from .session_analytics import SessionAnalytics


class HUNLGameSession:
    """An in-memory multi-hand heads-up session: human=seat 0, bot=seat 1.

    A bot policy can be RandomLegalPolicy, a diagnostic agent or a compatible
    NeuralActorPolicy. A trusted server must load checkpoints separately.
    """

    def __init__(
        self, *, bot_policy: GUIBotPolicy | None = None,
        big_blind: int = 100, small_blind: int = 50,
        stack_bb: int = 200, hero_seat: int = 0,
        seed: int | None = None, max_bot_actions: int = 1000,
        stack_mode: str = "independent",
    ) -> None:
        if hero_seat != 0:
            raise ValueError("GUI-1.0 fixes human seat to 0")
        if type(max_bot_actions) is not int or max_bot_actions < 1:
            raise ValueError("max_bot_actions must be positive integer")
        if stack_mode not in ("independent", "continuous"):
            raise ValueError("stack_mode must be independent or continuous")
        self.stack_mode = stack_mode
        # Validate configuration BEFORE accepting a game session.
        InteractiveHUNLAdapter(big_blind=big_blind, small_blind=small_blind, stack_bb=stack_bb)
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.stack_bb = stack_bb
        self.hero_seat = hero_seat
        self.max_bot_actions = max_bot_actions
        self._seed_rng = random.Random(seed)
        self.bot = BotActionAdapter(bot_policy or RandomLegalPolicy(), random.Random(self._seed_rng.getrandbits(64)))
        self._lock = threading.RLock()
        self._adapter: InteractiveHUNLAdapter | None = None
        self._transition: Turn | Terminal | None = None
        self._action_log: list[dict[str, Any]] = []
        self.hand_number = 0
        self.revision = 0
        self.completed_hands = 0
        self.cumulative_bb = 0.0
        self.analytics = SessionAnalytics()

    def start_hand(self) -> dict[str, Any]:
        with self._lock:
            if self._transition is not None and not isinstance(self._transition, Terminal):
                raise RuntimeError("finish the current hand before starting another")
            initial_stacks: tuple[int, int] | None = None
            if self.stack_mode == "continuous" and self._adapter is not None:
                final = self._adapter.engine.get_result().final_stacks
                initial_stacks = (int(final[0]), int(final[1]))
                if min(initial_stacks) == 0:
                    raise RuntimeError("match over: a player is bankrupt; start a new session")
                if sum(initial_stacks) != 2 * self.stack_bb * self.big_blind:
                    raise AssertionError("chips changed across consecutive hands")
            next_hand_number = self.hand_number + 1
            adapter = InteractiveHUNLAdapter(
                big_blind=self.big_blind, small_blind=self.small_blind,
                stack_bb=self.stack_bb, button=(next_hand_number - 1) % 2,
                expose_training_state=False,
            )
            deck_seed = self._seed_rng.getrandbits(64)
            next_transition = (adapter.reset_with_stacks(initial_stacks, seed=deck_seed)
                               if initial_stacks is not None else adapter.reset(seed=deck_seed))
            self.hand_number = next_hand_number
            self._adapter = adapter
            self._action_log = []
            self._transition = next_transition
            self.revision += 1
            if isinstance(next_transition, Terminal):
                # A short blind may force a zero-decision, immediate showdown.
                # It is still a completed hand with a real payoff.
                self._finish_terminal(next_transition)
            else:
                self._run_bots()
            return self._view_unlocked()

    def start_next_hand(self, *, expected_revision: int) -> dict[str, Any]:
        """Start the next hand atomically, rejecting stale/double clicks.

        GUI-1.1 extension. Unlike the legacy start_hand() entry point, this
        is safe to call from an HTTP handler with optimistic concurrency.
        """
        with self._lock:
            if type(expected_revision) is not int or expected_revision != self.revision:
                raise ValueError("stale revision: refresh game state before acting")
            if not isinstance(self._transition, Terminal):
                raise RuntimeError("finish the current hand before starting another")
            return self.start_hand()

    def play_human_action(self, action_index: int, *, expected_revision: int) -> dict[str, Any]:
        with self._lock:
            if type(expected_revision) is not int or expected_revision != self.revision:
                raise ValueError("stale revision: refresh game state before acting")
            if self._adapter is None or not isinstance(self._transition, Turn):
                raise RuntimeError("start a live hand before submitting an action")
            if self._transition.player_id != self.hero_seat:
                raise RuntimeError("it is not the human's turn")
            action_index = validate_human_action(self._transition.observation, action_index)
            self._take_action(action_index)
            self._run_bots()
            return self._view_unlocked()

    def play_human_raise_to(self, amount: int, *, expected_revision: int) -> dict[str, Any]:
        """Human-only arbitrary legal raise-to; the bot remains 7-action.

        An invalid amount or stale revision leaves the hand and revision intact.
        """
        with self._lock:
            if type(expected_revision) is not int or expected_revision != self.revision:
                raise ValueError("stale revision: refresh game state before acting")
            if self._adapter is None or not isinstance(self._transition, Turn):
                raise RuntimeError("start a live hand before submitting an action")
            if self._transition.player_id != self.hero_seat:
                raise RuntimeError("it is not the human's turn")
            actor = self._transition.player_id
            street = self._transition.observation.stage
            # Engine validates BEFORE changing state; no revision or event on error.
            next_state = self._adapter.step_human_raise_to(amount)
            self._record_transition(next_state, actor, street, -1,
                                    kind="raise_to", raise_to=amount)
            self._run_bots()
            return self._view_unlocked()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return self.analytics.summary()

    def history(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        with self._lock:
            return self.analytics.history(offset=offset, limit=limit)

    def view(self) -> dict[str, Any]:
        with self._lock:
            return self._view_unlocked()

    def _take_action(self, action_index: int) -> None:
        assert self._adapter is not None and isinstance(self._transition, Turn)
        actor = self._transition.player_id
        street = self._transition.observation.stage
        # If illegal, step raises and no action/revision/event is committed.
        next_state = self._adapter.step(action_index)
        resolved = self._adapter.last_engine_action
        assert resolved is not None
        self._record_transition(next_state, actor, street, action_index,
                                kind=resolved.kind, raise_to=resolved.raise_to)

    def _record_transition(self, next_state: Turn | Terminal, actor: int,
                           street: str, action_index: int, *, kind: str,
                           raise_to: int | None) -> None:
        self._transition = next_state
        self._action_log.append({
            "seat": actor, "street": street, "index": action_index,
            "kind": kind, "raise_to": raise_to,
        })
        self.revision += 1
        if isinstance(next_state, Terminal):
            self._finish_terminal(next_state)

    def _finish_terminal(self, transition: Terminal) -> None:
        self.completed_hands += 1
        self.cumulative_bb += transition.rewards[self.hero_seat]
        self.analytics.append_completed(self._view_unlocked())

    def _run_bots(self) -> None:
        count = 0
        while isinstance(self._transition, Turn) and self._transition.player_id != self.hero_seat:
            count += 1
            if count > self.max_bot_actions:
                raise RuntimeError("bot action limit reached")
            action_index = self.bot.choose(self._transition.observation)
            self._take_action(action_index)

    def _view_unlocked(self) -> dict[str, Any]:
        if self._adapter is None or self._transition is None:
            raise RuntimeError("start_hand must be called before view")
        return project_human_view(
            self._adapter, self._transition,
            hero_seat=self.hero_seat, hand_number=self.hand_number,
            revision=self.revision, action_log=self._action_log,
            stack_mode=self.stack_mode,
            completed_hands=self.completed_hands, cumulative_bb=self.cumulative_bb,
        )


class HUNLSessionRegistry:
    """Opaque session ID lookup for a future local-only FastAPI router.

    Store in memory only. GUI-1.1 must add API-level request validation and
    restrict CORS and binding to localhost; no auth or remote hosting yet.
    """

    def __init__(self, *, max_sessions: int = 128) -> None:
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self._sessions: dict[str, HUNLGameSession] = {}
        self._lock = threading.RLock()

    def create(self, **session_options: Any) -> tuple[str, dict[str, Any]]:
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("session capacity reached")
            session = HUNLGameSession(**session_options)
            first_view = session.start_hand()
            session_id = secrets.token_urlsafe(24)
            self._sessions[session_id] = session
            return session_id, first_view

    def get(self, session_id: str) -> HUNLGameSession:
        with self._lock:
            if not isinstance(session_id, str):
                raise KeyError("unknown session")
            return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        with self._lock:
            del self._sessions[session_id]

````````

<!-- END_PY_SOURCE path="src/holdem_ai/game_service/session_manager.py" -->
