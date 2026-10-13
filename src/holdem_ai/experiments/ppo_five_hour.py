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

        # If the process died after committing a validation and before
        # publishing best.pt, restore it from the authoritative latest.pt.
        if (best_update == session.completed_updates and validations
                and validations[-1]["attempted"] == len(rows)):
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
