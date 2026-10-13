"""Phase16 long PPO A0/A3 stress, full-state resume, and paired comparison.

Train one arm or campaign both arms sequentially on the local user GPU.
All outputs go to NEW run directories. Source PPO checkpoint is read-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from holdem_ai.experiments.ppo_stress import (
    StressConfig, compare_stress_runs, run_stress,
)


def _arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", required=True,
                   help="trusted ORIGINAL full PPO source checkpoint")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--updates", type=int, default=100,
                   help="TOTAL attempted updates PER arm, not per invocation")
    p.add_argument("--hands-per-update", type=int, default=128)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--env-batch-size", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-groups", type=int, default=40)
    p.add_argument("--critic-eval-hands", type=int, default=128)
    p.add_argument("--rollout-seed", type=int, default=680000000)
    p.add_argument("--validation-seed", type=int, default=78000000)
    p.add_argument("--critic-seed", type=int, default=690000000)
    p.add_argument("--max-post-kl", type=float, default=0.02)
    p.add_argument("--hard-stop-kl", type=float, default=0.10)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--max-updates-this-call", type=int, default=None,
                   help="optional per-process update budget per arm")
    p.add_argument("--resume", action="store_true",
                   help="resume existing arm from atomic latest.pt")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase16 long A0/A3 stress and comparison")
    subs = p.add_subparsers(dest="command", required=True)
    arm = subs.add_parser("train", help="train one specific arm with full-state resume")
    _arguments(arm)
    arm.add_argument("--variant", choices=("a0", "a3"), required=True)
    arm.add_argument("--root", required=True, help="new per-arm run root")
    all_arms = subs.add_parser("campaign", help="run A0 then A3 and auto-compare")
    _arguments(all_arms)
    all_arms.add_argument("--root", required=True,
                          help="campaign root containing a0/ and a3/")
    cmp_parser = subs.add_parser("compare", help="paired read-only comparison")
    cmp_parser.add_argument("--a0", required=True, help="A0 run root")
    cmp_parser.add_argument("--a3", required=True, help="A3 run root")
    cmp_parser.add_argument("--output", required=True, help="new comparison JSON")
    return p


def _config(args, *, variant: str, root: str, resume: bool) -> StressConfig:
    return StressConfig(
        source_checkpoint=args.source, root=root, variant=variant,
        device=args.device, updates=args.updates,
        hands_per_update=args.hands_per_update,
        minibatch_size=args.minibatch_size, epochs=args.epochs,
        env_batch_size=args.env_batch_size,
        eval_every=args.eval_every, eval_groups=args.eval_groups,
        critic_eval_hands=args.critic_eval_hands,
        rollout_seed=args.rollout_seed,
        validation_seed=args.validation_seed,
        critic_seed=args.critic_seed,
        max_post_kl=args.max_post_kl,
        hard_stop_kl=args.hard_stop_kl,
        save_every=args.save_every,
        max_updates_this_call=args.max_updates_this_call,
        resume=resume,
    )


def _save_comparison(output: Path, value: dict):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main():
    args = parser().parse_args()
    if args.command == "compare":
        result = compare_stress_runs(args.a0, args.a3)
        _save_comparison(Path(args.output), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.command == "train":
        config = _config(
            args, variant=args.variant, root=args.root, resume=args.resume)
        report = run_stress(config)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return

    campaign = Path(args.root)
    if not args.resume and campaign.exists() and any(campaign.iterdir()):
        raise FileExistsError("new campaign root must be unused")
    summaries = {}
    for arm in ("a0", "a3"):
        path = campaign / arm
        existing = (path / "run.json").is_file()
        if existing and not args.resume:
            raise FileExistsError(f"arm exists; use --resume: {path}")
        if args.resume and path.exists() and any(path.iterdir()) and not existing:
            raise ValueError(f"incomplete arm directory: {path}")
        config = _config(
            args, variant=arm, root=str(path), resume=existing and args.resume)
        print(f"START {arm}: {path} resume={config.resume}", flush=True)
        report = run_stress(config)
        summaries[arm] = report
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
              flush=True)
    if all(summaries[arm]["completed"] for arm in ("a0", "a3")):
        compared = compare_stress_runs(campaign / "a0", campaign / "a3")
        result_path = campaign / "paired_comparison.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing != compared:
                raise ValueError("existing paired comparison differs; refuse overwrite")
        else:
            _save_comparison(result_path, compared)
        print(json.dumps({"campaign": str(campaign),
                          "paired_report": str(result_path),
                          "paired_validation": compared["paired_validations"]},
                         ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(json.dumps({
            "campaign": str(campaign), "status": "partial",
            "note": "Run the SAME campaign command with --resume to continue "
                    "from the saved full-state checkpoint.",
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
