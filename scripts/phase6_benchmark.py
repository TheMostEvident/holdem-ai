"""CPU-only Phase 6 HUNL200 benchmark; no PyTorch/CUDA required.

Run at repository root:
    uv run python scripts/phase6_benchmark.py --hands 200 --workers 1,2,4
The default 100 warmup/test hands per worker mode is not a five-hour train.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time

from holdem_agent_phase5.policy import RandomLegalPolicy
from holdem_agent_phase5.runner import run_self_play
from holdem_agent_phase6 import HUNL200Adapter


def _run_one(hand_index: int, seed: int, max_decisions: int) -> tuple[float, int]:
    env = HUNL200Adapter(button=hand_index % 2)
    ep = run_self_play(
        env, {0: RandomLegalPolicy(), 1: RandomLegalPolicy()},
        seed=seed + hand_index, max_decisions=max_decisions,
    )
    if abs(sum(ep.rewards)) > 1e-9 or not ep.transitions:
        raise AssertionError("bad zero-sum rewards or missing decisions")
    if any(t.action_index not in t.observation.legal_actions for t in ep.transitions):
        raise AssertionError("illegal action in trajectory")
    return ep.rewards[0], ep.num_decisions


def run_benchmark(n_hands: int, n_workers: int, *, seed: int = 42,
                  max_decisions: int = 200) -> dict[str, float | int]:
    if n_hands < 1 or n_workers < 1 or max_decisions < 1:
        raise ValueError("hands, workers and max_decisions must be positive")
    begin = time.perf_counter()
    args = ((i, seed, max_decisions) for i in range(n_hands))
    if n_workers == 1:
        samples = [_run_one(*arg) for arg in args]
    else:
        # spawn-compatible Windows worker function; no GPU context created.
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_run_one, *arg) for arg in args]
            samples = [f.result() for f in futures]
    elapsed = time.perf_counter() - begin
    return {
        "hands": n_hands, "workers": n_workers, "elapsed_seconds": round(elapsed, 4),
        "hands_per_second": round(n_hands / elapsed, 3),
        "decisions": sum(x[1] for x in samples),
        "decisions_per_second": round(sum(x[1] for x in samples) / elapsed, 3),
        "seat0_mean_bb100": round(100.0 * sum(x[0] for x in samples) / n_hands, 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="HUNL200 Phase 6 CPU rollout speed test")
    p.add_argument("--hands", type=int, default=100)
    p.add_argument("--workers", default="1,2,4", help="comma separated, e.g. 1,2,4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-decisions", type=int, default=200)
    p.add_argument("--json-out", default=None, help="optional JSON report filepath")
    args = p.parse_args()
    modes = [int(x) for x in args.workers.split(",")]
    # Warm up imports + one hand; excluded from timed benchmark.
    _run_one(-2, args.seed, args.max_decisions)
    reports = [run_benchmark(args.hands, w, seed=args.seed, max_decisions=args.max_decisions) for w in modes]
    for report in reports:
        print(json.dumps(report, ensure_ascii=False))
    if args.json_out:
        from pathlib import Path
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {output}")


if __name__ == "__main__":
    main()
