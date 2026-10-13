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
