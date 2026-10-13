"""Read-only baseline action comparison for HUNL self-play, Phase 15.

Three sources: uniform legal action sampler, fresh seed-controlled random-init
TransformerActor (or explicitly supplied original checkpoint), Phase14 actor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from holdem_ai.evaluation.initial_policy_comparison import compare_three_agent_policies


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Uniform vs initial vs trained Phase14")
    p.add_argument("--trained", required=True, help="trusted Phase14 PPO checkpoint")
    p.add_argument("--initial-checkpoint", default=None,
                   help="optional exact saved pretraining checkpoint")
    p.add_argument("--init-seed", type=int, default=52,
                   help="random fresh init seed, used only if no initial checkpoint")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--hands", type=int, default=512, help="hands PER policy")
    p.add_argument("--seed", type=int, default=590000000)
    p.add_argument("--env-batch-size", type=int, default=32)
    p.add_argument("--output", required=True, help="new report filename, refuse overwrite")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"report exists: {out}")
    report = compare_three_agent_policies(
        trained_checkpoint=args.trained,
        initial_checkpoint=args.initial_checkpoint,
        init_seed=args.init_seed, device=args.device,
        hands=args.hands, seed=args.seed, env_batch_size=args.env_batch_size,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    summary = {}
    for name, result in report["on_policy"].items():
        hand = result["terminal"]
        action = result["actions"]
        summary[name] = {
            "hands": hand["hands"],
            "preflop_explicit_allin_hand_fraction": round(
                hand["preflop"]["explicit_all_in_hand_count"] / hand["hands"], 6),
            "preflop_allin_given_legal": action["preflop"]["all_in_rate_given_legal"],
            "first_decision_allin_given_legal": action["per_hand"]["first_decision_all_in_given_legal"],
            "preflop_last_decision_fraction": hand["preflop"]["last_decision_hand_fraction"],
            "near_full_stack_fraction": hand["seat0_hand_weighted_reward"]["near_full_stack_fraction"],
            "postflop_decision_hands": hand["decision_coverage"]["flop_hands"],
            "fixed_phase14_states_allin_probability": report["fixed_phase14_states"]["results"][name]["overall"]["mean_action_probability_given_legal"]["all_in"],
        }
    print(json.dumps({
        "report": str(out),
        "initial_policy_provenance": report["initial_policy_provenance"],
        "fixed_state_count": report["fixed_phase14_states"]["states"],
        "summary": summary,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
