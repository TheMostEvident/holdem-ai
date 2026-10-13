"""Run a read-only diagnostic for two trusted Phase-15 HUNL PPO checkpoints.

Example (PowerShell):
  uv run --extra gpu python -m scripts.phase15_critic_audit \
    --reference runs/phase14_main/seed_52/ppo/best.pt \
    --before runs/phase15_ppo_safety_v2_pilot/best.pt \
    --after runs/phase15_ppo_safety_v2_pilot/latest.pt \
    --device cuda --hands 256 --on-policy-hands 128 \
    --output runs/phase15_audit/safety_v2_critic_audit.json

No training, gradient steps, checkpoint writes, or final evaluation sealing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from holdem_ai.evaluation.critic_diagnostics import run_critic_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Phase 15 PPO critic diagnosis")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hands", type=int, default=256,
                        help="hands on shared reference policy")
    parser.add_argument("--on-policy-hands", type=int, default=128,
                        help="hands per checkpoint, using its own policy")
    parser.add_argument("--seed", type=int, default=570000000)
    parser.add_argument("--env-batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output", required=True, help="new report filename; refuses overwrite")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"report exists: {target}")
    report = run_critic_audit(
        reference_checkpoint=args.reference,
        before_checkpoint=args.before,
        after_checkpoint=args.after,
        device=args.device, hands=args.hands,
        on_policy_hands=args.on_policy_hands,
        seed=args.seed, env_batch_size=args.env_batch_size,
        chunk_size=args.chunk_size,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    summary = {
        "report": str(target),
        "updates": report["completed_updates"],
        "shared_states": report["shared_reference"]["decisions"],
        "on_policy": {
            label: {
                "states": data["decisions"],
                "q_rmse_bb": data["q_taken"]["overall"]["rmse_bb"],
                "q_ev": data["q_taken"]["overall"]["explained_variance"],
                "v_rmse_bb": data["v_reference_policy"]["overall"]["rmse_bb"],
                "v_ev": data["v_reference_policy"]["overall"]["explained_variance"],
            }
            for label, data in report["on_policy"]["results"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
