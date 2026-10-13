"""Five-hour Phase17 GPU retraining: new run by default; resume explicitly."""
from __future__ import annotations

import argparse
import json

from holdem_ai.experiments.ppo_five_hour import FiveHourConfig, run_five_hour


def build_parser():
    p = argparse.ArgumentParser(
        description="Phase17 5-hour HUNL PPO training with full-state restart")
    p.add_argument("--source", required=True,
                   help="trusted Phase14 best.pt, full PPO checkpoint")
    p.add_argument("--root", required=True,
                   help="NEW output directory; refuses nonempty existing runs")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--duration-hours", type=float, default=5.0,
                   help="TOTAL cumulative active hours, excluding downtime")
    p.add_argument("--variant", choices=("a0", "a3"), default="a0")
    p.add_argument("--hands-per-update", type=int, default=128)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--env-batch-size", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-groups", type=int, default=80)
    p.add_argument("--critic-eval-hands", type=int, default=128)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--max-updates", type=int, default=5000,
                   help="high safety ceiling; does not determine 5h duration")
    p.add_argument("--rollout-seed", type=int, default=700000000)
    p.add_argument("--validation-seed", type=int, default=80000000)
    p.add_argument("--critic-seed", type=int, default=710000000)
    p.add_argument("--max-post-kl", type=float, default=0.02,
                   help="used only by opt-in A3 rollback")
    p.add_argument("--hard-stop-kl", type=float, default=0.10,
                   help="A0 emergency hard-stop for extreme policy drift")
    p.add_argument("--resume", action="store_true",
                   help="reuse saved full-state checkpoint and scientific settings")
    p.add_argument("--max-updates-this-call", type=int, default=None,
                   help="optional per-process update cap for smoke/segmentation")
    return p


def main():
    a = build_parser().parse_args()
    cfg = FiveHourConfig(
        source_checkpoint=a.source, root=a.root, device=a.device,
        duration_hours=a.duration_hours, variant=a.variant,
        hands_per_update=a.hands_per_update,
        minibatch_size=a.minibatch_size, epochs=a.epochs,
        env_batch_size=a.env_batch_size,
        eval_every=a.eval_every, eval_groups=a.eval_groups,
        critic_eval_hands=a.critic_eval_hands, save_every=a.save_every,
        max_updates=a.max_updates, rollout_seed=a.rollout_seed,
        validation_seed=a.validation_seed, critic_seed=a.critic_seed,
        max_post_kl=a.max_post_kl, hard_stop_kl=a.hard_stop_kl,
        resume=a.resume, max_updates_this_call=a.max_updates_this_call,
    )
    print(json.dumps({
        "protocol": "phase17_five_hour_ppo_v1",
        "root": cfg.root, "variant": cfg.variant,
        "total_active_hours": cfg.duration_hours,
        "resume": cfg.resume,
        "note": "on-policy learner starts on the trusted Phase14 source, "
                "NOT from the unvalidated Phase16 Update204 actor",
    }, ensure_ascii=False, indent=2), flush=True)
    report = run_five_hour(cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
          flush=True)


if __name__ == "__main__":
    main()
