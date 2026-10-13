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
        if (len({self.rollout_seed, self.validation_seed, self.critic_seed}) != 3
                or not self.validation_seed + self.eval_groups < self.critic_seed
                or not self.rollout_seed + self.hands_per_update * self.updates < self.critic_seed):
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
        evaluation = _eval(session.actor, source_actor, config=eval_cfg, control=control)
        _actor_save(session.actor, arm_root / "final_actor.pt")
        data = {
            "variant": variant, "label": LABELS[variant],
            "updates": updates, "critic_holdout": critic_stats,
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
            "evaluation": data["evaluation"],
            "rejected_updates": data["rejected_updates"],
            "post_kl_max": data["post_kl_max"], "checkpoint": data["checkpoint"],
        }
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("source checkpoint changed during ablation")
    _put_json(root / "summary.json", report)
    return report
