"""Read-only HUNL PPO per-hand reward tails and preflop exit audit.

Run on two existing trusted checkpoints; saves a NEW JSON report.
No optimizer calls, training checkpoint updates or final-test sealing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from holdem_ai.evaluation.terminal_reward_audit import run_terminal_reward_audit


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HUNL PPO terminal rewards + preflop exits")
    p.add_argument("--before", required=True, help="trusted PPO best checkpoint")
    p.add_argument("--after", required=True, help="trusted PPO latest checkpoint")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--hands", type=int, default=512, help="hands per model")
    p.add_argument("--seed", type=int, default=580000000)
    p.add_argument("--env-batch-size", type=int, default=32)
    p.add_argument("--stack-bb", type=float, default=200.0)
    p.add_argument("--output", required=True, help="new JSON report; refuses overwrite")
    return p


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"report exists: {output}")
    report = run_terminal_reward_audit(
        checkpoints={"before": args.before, "after": args.after},
        device=args.device, hands=args.hands,
        seed=args.seed, env_batch_size=args.env_batch_size,
        stack_bb=args.stack_bb)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    selected = {}
    for label, result in report["models"].items():
        dist = result["seat0_hand_weighted_reward"]
        selected[label] = {
            "hands": result["hands"],
            "mean_abs_reward_bb": dist["mean_absolute_bb"],
            "reward_std_bb": dist["std_bb"],
            "abs_ge_190bb_fraction": dist["abs_ge_190bb_fraction"],
            "near_full_stack_fraction": dist["near_full_stack_fraction"],
            "preflop_last_decision_fraction": result["preflop"]["last_decision_hand_fraction"],
            "preflop_fold_ended": result["preflop"]["fold_ended_count"],
            "preflop_nonfold_showdown": result["preflop"]["nonfold_showdown_count"],
            "preflop_explicit_all_in": result["preflop"]["explicit_all_in_hand_count"],
        }
    print(json.dumps({"report": str(output), "summary": selected},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
