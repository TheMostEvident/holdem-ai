"""CLI: fixed-state policy audit and existing PPO metrics review.

Example:
  uv run --extra gpu python -m scripts.phase15_policy_audit \
    --reference runs/phase14_main/seed_52/ppo/best.pt \
    --before runs/phase15_ppo_stability_pilot/best.pt \
    --after runs/phase15_ppo_stability_pilot/latest.pt \
    --metrics runs/phase15_ppo_stability_pilot/metrics.jsonl \
    --output runs/phase15_audit/policy_audit.json --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from holdem_ai.evaluation.fixed_state_policy_audit import compare_checkpoints
from holdem_ai.evaluation.ppo_metric_audit import audit_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit HUNL PPO collapse without training")
    parser.add_argument("--reference", required=True, help="trusted rollout source checkpoint")
    parser.add_argument("--before", required=True, help="earlier/best trusted checkpoint")
    parser.add_argument("--after", required=True, help="later/latest trusted checkpoint")
    parser.add_argument("--metrics", default=None, help="optional Phase15 metrics.jsonl")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hands", type=int, default=128)
    parser.add_argument("--seed", type=int, default=550000000)
    parser.add_argument("--env-batch-size", type=int, default=32)
    parser.add_argument("--output", required=True, help="new JSON report; never overwrite")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = compare_checkpoints(
        {"before": args.before, "after": args.after},
        reference_checkpoint=args.reference, device=args.device,
        hands=args.hands, seed=args.seed, env_batch_size=args.env_batch_size,
    )
    if args.metrics:
        report["ppo_update_metrics"] = audit_metrics(args.metrics)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")
    print(json.dumps({
        "output": str(target),
        "states": report["states"],
        "delta_conditional_fold_probability": report["comparison"]["delta_conditional_fold_probability"],
        "warning": report["warning"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
