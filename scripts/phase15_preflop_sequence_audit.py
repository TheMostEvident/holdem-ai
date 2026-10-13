"""Read-only HUNL Phase15 preflop action-sequence audit.

Compares three policies on identical card seeds, distinct sampled trajectories.
Writes a new JSON report only, never modifies checkpoints or train runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from holdem_ai.evaluation.preflop_sequence_audit import compare_preflop_sequences


def main() -> None:
    p = argparse.ArgumentParser(description="Preflop action paths: uniform, random init, Phase14")
    p.add_argument("--trained", required=True)
    p.add_argument("--initial-checkpoint", default=None)
    p.add_argument("--init-seed", type=int, default=52)
    p.add_argument("--hands", type=int, default=512)
    p.add_argument("--seed", type=int, default=600000000)
    p.add_argument("--env-batch-size", type=int, default=32)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"report exists: {target}")
    result = compare_preflop_sequences(
        trained_checkpoint=args.trained,
        initial_checkpoint=args.initial_checkpoint,
        init_seed=args.init_seed, hands=args.hands, seed=args.seed,
        env_batch_size=args.env_batch_size, device=args.device,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    summary = {}
    for label, data in result["models"].items():
        depth = data["action_choice_by_preflop_decision_number"]
        key = "2"
        summary[label] = {
            "hands": data["hands"],
            "pf_allin_hands": data["preflop_explicit_allin_hands"],
            "pf_allin_fraction": data["preflop_explicit_allin_fraction"],
            "pf_fold_terminated": data["preflop_terminal_fold_hands"],
            "pf_end_call": data["preflop_terminal_call_hands"],
            "pf_actions_per_hand": data["preflop_decisions_per_hand"]["mean"],
            "first_allin_decision_number_counts": data["first_allin_decision_number_counts"],
            "immediately_before_first_allin": data["immediately_before_first_allin"],
            "second_decision": depth.get(key),
            "top_allin_prefixes": data["most_common_prefixes_through_first_allin"][:8],
        }
    print(json.dumps({"report": str(target), "summary": summary},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
