"""Phase16 isolated A0/A1/A2/A3 PPO ablation (NEW RUN ONLY)."""
from __future__ import annotations

import argparse
import json

from holdem_ai.experiments.ppo_ablation import AblationConfig, VARIANTS, run_ablation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Short HUNL PPO optimizer/critic ablation")
    p.add_argument("--source", required=True, help="trusted Phase14 PPO full checkpoint")
    p.add_argument("--root", required=True, help="NEW empty root; refuses overwrite")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--variants", default=",".join(VARIANTS),
                   help="comma-separated a0,a1,a2,a3; all by default")
    p.add_argument("--hands-per-update", type=int, default=64)
    p.add_argument("--updates", type=int, default=8)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--env-batch-size", type=int, default=32)
    p.add_argument("--eval-groups", type=int, default=12)
    p.add_argument("--rollout-seed", type=int, default=620000000)
    p.add_argument("--validation-seed", type=int, default=75000000)
    p.add_argument("--critic-seed", type=int, default=660000000)
    p.add_argument("--critic-eval-hands", type=int, default=64)
    p.add_argument("--target-scale", type=float, default=200.0)
    p.add_argument("--max-post-kl", type=float, default=0.02)
    return p


def main():
    args = build_parser().parse_args()
    variants = tuple(x.strip() for x in args.variants.split(","))
    cfg = AblationConfig(
        source_checkpoint=args.source, root=args.root, device=args.device,
        variants=variants, hands_per_update=args.hands_per_update,
        updates=args.updates, minibatch_size=args.minibatch_size,
        epochs=args.epochs, env_batch_size=args.env_batch_size,
        eval_groups=args.eval_groups, rollout_seed=args.rollout_seed,
        validation_seed=args.validation_seed, critic_seed=args.critic_seed,
        critic_eval_hands=args.critic_eval_hands,
        target_scale=args.target_scale, max_post_kl=args.max_post_kl,
    )
    print(json.dumps({"protocol": "phase16_ablation", "root": args.root,
                      "variants": variants, "hands_per_update": cfg.hands_per_update,
                      "updates": cfg.updates,
                      "total_training_hands": len(variants) * cfg.updates * cfg.hands_per_update},
                     ensure_ascii=False, indent=2), flush=True)
    summary = run_ablation(cfg)
    compact = {
        variant: {
            "critic_ev": results["critic_holdout"]["explained_variance"],
            "critic_rmse_bb": results["critic_holdout"]["rmse_bb"],
            "max_exact_post_kl": results["post_kl_max"],
            "rejected_updates": results["rejected_updates"],
            "vs_phase14_paired_delta_bb100": results["evaluation"]["frozen_phase14"]["paired_change_from_source"]["delta_mean_bb100"],
            "vs_phase14_delta_ci95_bb100": [
                results["evaluation"]["frozen_phase14"]["paired_change_from_source"]["delta_ci95_lower_bb100"],
                results["evaluation"]["frozen_phase14"]["paired_change_from_source"]["delta_ci95_upper_bb100"],
            ],
            "vs_random_paired_delta_bb100": results["evaluation"]["uniform_random_legal"]["paired_change_from_source"]["delta_mean_bb100"],
        }
        for variant, results in summary["variants"].items()
    }
    print(json.dumps({"root": args.root, "results": compact},
                     ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
