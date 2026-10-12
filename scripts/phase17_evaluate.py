"""Phase 17 variable-stack continuous-match evaluation (no training).

Run with --candidate PATH for a trusted Phase15 best.pt checkpoint or use
--candidate-builtin random for a quick environment sanity test.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

from holdem_ai.agents.diagnostic_opponents import BUILTIN_OPPONENTS, build_builtin
from holdem_ai.evaluation.continuous_stack import (
    ContinuousEvaluationConfig, evaluate_continuous_stacks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase17 continuous-stack HUNL evaluation")
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate", help="trusted trained checkpoint .pt")
    candidate.add_argument("--candidate-builtin", choices=BUILTIN_OPPONENTS,
                           help="baseline policy sanity run without checkpoint")
    opponent = parser.add_mutually_exclusive_group()
    opponent.add_argument("--opponent", choices=BUILTIN_OPPONENTS, default="calling_station")
    opponent.add_argument("--opponent-checkpoint", help="trained opponent actor .pt")
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--control", help="trained control actor .pt")
    control.add_argument("--control-builtin", choices=BUILTIN_OPPONENTS)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--hands-per-match", type=int, default=30)
    parser.add_argument("--candidate-stacks-bb", default="50,100,200,300,350")
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17_000_000)
    parser.add_argument("--max-decisions-per-hand", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("sampled", "greedy"), default="sampled")
    parser.add_argument("--output-dir", required=True,
                        help="new / empty directory, e.g. runs/phase17/ppo-vs-calling-v1")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _parse_stacks(spec: str) -> tuple[int, ...]:
    try:
        if not spec or any(not token.strip() for token in spec.split(",")):
            raise ValueError("empty token")
        return tuple(int(token.strip()) for token in spec.split(","))
    except ValueError as exc:
        raise ValueError("--candidate-stacks-bb must contain comma-separated integer BB values") from exc


def _load_checkpoint(path: str, *, device: str, mode: str, big_blind: int):
    # Reuse the existing Phase16 loader, with torch weights_only=True,
    # exact actor-state validation and NO critic on the inference path.
    from holdem_ai.evaluation.checkpoint_loader import load_actor
    from holdem_ai.evaluation.seed_blocked import model_fingerprint
    from holdem_ai.agents.neural_policy import NeuralActorPolicy

    actor = load_actor(path, device=device)
    if actor.config.dropout != 0:
        raise ValueError("Phase17 inference requires checkpoint actor dropout=0")
    return (NeuralActorPolicy(actor, device=device, big_blind=big_blind,
                              greedy=mode == "greedy"),
            "actor:" + model_fingerprint(actor), actor)


def _policy(builtin: str | None, path: str | None, *, args):
    if path:
        return _load_checkpoint(path, device=args.device, mode=args.mode,
                                big_blind=args.big_blind)
    assert builtin is not None
    return build_builtin(builtin), "builtin:" + builtin, None


def _atomic(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _csv_content(result: dict) -> str:
    columns = ("candidate_start_bb", "opponent_start_bb", "policy", "groups", "matches",
               "hands", "mean_bb100", "ci95_lower_bb100", "ci95_upper_bb100",
               "mean_net_bb_per_match", "mean_hands_per_match", "match_profit_fraction",
               "hand_win_fraction", "hand_loss_fraction", "hand_tie_fraction",
               "candidate_bust_fraction", "opponent_bust_fraction")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for scenario in result["scenarios"]:
        for name in ("candidate", "control"):
            if name not in scenario:
                continue
            data = scenario[name]
            writer.writerow({
                "candidate_start_bb": scenario["candidate_start_bb"],
                "opponent_start_bb": scenario["opponent_start_bb"],
                "policy": name,
                **{key: data[key] for key in columns if key in data},
            })
    return output.getvalue()


def _markdown(result: dict) -> str:
    lines = [
        "# Phase 17 — Continuous Stack HUNL Evaluation", "",
        f"- Candidate: `{result['candidate_id']}`",
        f"- Opponent: `{result['opponent_id']}`",
        f"- Control: `{result.get('control_id', 'none')}`",
        f"- Protocol: `{result['protocol']}`", "",
        "| Candidate / Opponent starting BB | Matches | Hands | Mean bb/100 | 95% approx CI | Candidate bust |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for scenario in result["scenarios"]:
        stats = scenario["candidate"]
        lines.append(
            f"| {scenario['candidate_start_bb']} / {scenario['opponent_start_bb']} "
            f"| {stats['matches']} | {stats['hands']} | {stats['mean_bb100']:+.3f} "
            f"| [{stats['ci95_lower_bb100']:+.3f}, {stats['ci95_upper_bb100']:+.3f}] "
            f"| {100 * stats['candidate_bust_fraction']:.1f}% |"
        )
        if "paired_delta" in scenario:
            delta = scenario["paired_delta"]
            lines.append(
                f"| **Candidate − control** | | | {delta['mean_bb100']:+.3f} "
                f"| [{delta['ci95_lower_bb100']:+.3f}, {delta['ci95_upper_bb100']:+.3f}] | |"
            )
    lines.extend([
        "", "## Statistics", "",
        "For each independent deck-seed group $g$, pool the four physical-layout matches:",
        "",
        r"\[b_g=100\frac{\sum_{\ell=1}^4 \Delta S_{g,\ell}^{BB}}{\sum_{\ell=1}^4 H_{g,\ell}}\]",
        "",
        r"\[\overline b=\frac1G\sum_g b_g,\qquad CI_{95\%}\approx\overline b\pm1.96\frac{s_b}{\sqrt G}\]",
        "",
        "Each seed group is the replicate; hands within one continuous match are correlated.",
        "Match lengths vary after bankruptcy. The ratio estimator is calculated within each",
        "group and then averaged equally across groups, not pooled globally.",
        "",
        "## Interpretation and limitations", "",
        result["interpretation"], "",
        "The actor uses only the active player's InformationState. Neither critic nor",
        "opponent hidden cards reach the acting policy. No GTO/exploitability claim.",
        "", "The 200/200 BB row is STILL a continuous match, not Phase16 resets.", "",
    ])
    return "\n".join(lines)


def run(args) -> dict:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("output directory is not empty; use a fresh directory")
    cfg = ContinuousEvaluationConfig(
        groups=args.groups, hands_per_match=args.hands_per_match,
        candidate_stacks_bb=_parse_stacks(args.candidate_stacks_bb),
        big_blind=args.big_blind, small_blind=args.small_blind,
        seed=args.seed, max_decisions_per_hand=args.max_decisions_per_hand,
    )
    # Fail early if requested CUDA is unavailable.
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
    candidate, candidate_id, candidate_actor = _policy(args.candidate_builtin, args.candidate, args=args)
    opponent, opponent_id, opponent_actor = _policy(
        None if args.opponent_checkpoint else args.opponent,
        args.opponent_checkpoint, args=args)
    control = None
    control_id = None
    control_actor = None
    if args.control or args.control_builtin:
        control, control_id, control_actor = _policy(args.control_builtin, args.control, args=args)
    fingerprint = None
    if any(actor is not None for actor in (candidate_actor, opponent_actor, control_actor)):
        from holdem_ai.evaluation.seed_blocked import model_fingerprint
        fingerprint = model_fingerprint
    before = [fingerprint(model) if model is not None else None
              for model in (candidate_actor, opponent_actor, control_actor)] if fingerprint else []

    def progress(start_bb, done, total):
        if not args.no_progress:
            print(f"Phase17 candidate {start_bb}BB: group {done}/{total}", flush=True)

    result = evaluate_continuous_stacks(candidate, opponent, config=cfg,
                                        control=control, on_group=progress)
    if fingerprint:
        after = [fingerprint(model) if model is not None else None
                 for model in (candidate_actor, opponent_actor, control_actor)]
        if before != after:
            raise AssertionError("Phase17 evaluation mutated model weights")
    result.update(candidate_id=candidate_id, opponent_id=opponent_id,
                  control_id=control_id, mode=args.mode, device=args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic(output_dir / "results.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _atomic(output_dir / "summary.csv", _csv_content(result))
    _atomic(output_dir / "report.md", _markdown(result))
    print(json.dumps({
        "phase": 17, "output_dir": str(output_dir.resolve()),
        "candidate_id": candidate_id, "scenarios": [
            {"candidate_start_bb": s["candidate_start_bb"],
             "mean_bb100": s["candidate"]["mean_bb100"],
             "hands": s["candidate"]["hands"]} for s in result["scenarios"]
        ],
    }, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
