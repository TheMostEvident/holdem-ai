# Python source code: volume 4 of 28

Exact full .py sources from baseline commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. Integral appendix of [final technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md).

## scripts/phase15_finalize.py

Original Git blob SHA: `dedc8c74a631e6a541379ff17a507f03a44f137d` (639 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase15_finalize.py" sha="dedc8c74a631e6a541379ff17a507f03a44f137d" -->

````````python
"""Phase 15 independent held-out evaluation, run ONCE after training is finished."""
from __future__ import annotations

import argparse
import json

from holdem_ai.experiments.longrun import finalize_longrun


def build_parser():
    p=argparse.ArgumentParser(description='Seal Phase15 best checkpoint and run held-out audit')
    p.add_argument('--root',required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    return p


def main():
    args=build_parser().parse_args()
    print(json.dumps(finalize_longrun(args.root,device=args.device),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase15_finalize.py" -->

## scripts/phase15_initial_policy_comparison.py

Original Git blob SHA: `7b87e08b4b9c76c054e9de78b125b8e580999019` (3227 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase15_initial_policy_comparison.py" sha="7b87e08b4b9c76c054e9de78b125b8e580999019" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase15_initial_policy_comparison.py" -->

## scripts/phase15_policy_audit.py

Original Git blob SHA: `9785d075192f770364bc8f1c62b8c151fcda3b17` (2491 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase15_policy_audit.py" sha="9785d075192f770364bc8f1c62b8c151fcda3b17" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase15_policy_audit.py" -->

## scripts/phase15_preflop_sequence_audit.py

Original Git blob SHA: `d54717d50e012b8261aaf720859c5245ee701bc1` (2636 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase15_preflop_sequence_audit.py" sha="d54717d50e012b8261aaf720859c5245ee701bc1" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase15_preflop_sequence_audit.py" -->

## scripts/phase15_terminal_reward_audit.py

Original Git blob SHA: `e3214f569214051bd30b8bb1d4308814ea30dd82` (2694 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase15_terminal_reward_audit.py" sha="e3214f569214051bd30b8bb1d4308814ea30dd82" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase15_terminal_reward_audit.py" -->

## scripts/phase15_train.py

Original Git blob SHA: `e2e58d6600bb85d48c113bd212eb5dbe6ad1e6f9` (2089 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase15_train.py" sha="e2e58d6600bb85d48c113bd212eb5dbe6ad1e6f9" -->

````````python
"""Phase 15 HUNL long-run training, from a Phase14 best.pt (or trusted PPO/QBoost checkpoint)."""
from __future__ import annotations

import argparse
import json

from holdem_ai.experiments.longrun import LongRunConfig, run_longrun


def build_parser():
    p=argparse.ArgumentParser(description='Phase15 safe long-run PPO/QBoost self-play training')
    p.add_argument('--source-checkpoint',default=None)
    p.add_argument('--algorithm',choices=('ppo','qboost'),default=None)
    p.add_argument('--root',default='runs/phase15_ppo')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--duration-minutes',type=float,default=300.0)
    p.add_argument('--max-updates',type=int,default=None,help='additional updates this invocation')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-progress',action='store_true')
    for name,kwargs in (
        ('hands-per-update',dict(type=int)), ('env-batch-size',dict(type=int)),
        ('minibatch-size',dict(type=int)), ('epochs',dict(type=int)),
        ('rollout-seed',dict(type=int)),('eval-every',dict(type=int)),
        ('eval-groups',dict(type=int)),('validation-seed',dict(type=int)),
        ('final-groups',dict(type=int)),('final-seed',dict(type=int)),
        ('entropy-alert-threshold',dict(type=float)),
        ('entropy-stop-patience',dict(type=int)),('max-approx-kl',dict(type=float)),
        ('beta',dict(type=float)),
    ):
        p.add_argument('--'+name,default=None,**kwargs)
    p.add_argument('--eval-mode',choices=('greedy','sampled'),default=None)
    return p


def run(args):
    options=vars(args).copy()
    options['progress']=not options.pop('no_progress')
    cfg=LongRunConfig(**options)
    def emit(e):
        if e.get('kind') in ('start','resume','validation','end'):
            print(json.dumps(e,ensure_ascii=False),flush=True)
    result=run_longrun(cfg,emit=emit)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    run(build_parser().parse_args())


if __name__=='__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase15_train.py" -->

## scripts/phase16_evaluate.py

Original Git blob SHA: `168ba0b04bb392390c8bf1534730a5783b6d4d5b` (6132 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase16_evaluate.py" sha="168ba0b04bb392390c8bf1534730a5783b6d4d5b" -->

````````python
"""No-training Phase16 benchmark for Phase14 and future Phase15 best.pt actors."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.agents.diagnostic_opponents import BUILTIN_OPPONENTS, OpponentSpec
from holdem_ai.evaluation.opponent_pool import PoolEvaluationConfig, evaluate_opponent_pool
from holdem_ai.evaluation.strategy_report import markdown_report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase16 opponent pool and strategy diagnostics (evaluation only)')
    p.add_argument('--candidate', required=True, help='trusted Phase9/10/14/15 full checkpoint or actor snapshot')
    p.add_argument('--opponents', default=','.join(BUILTIN_OPPONENTS),
                   help='comma-separated built-in names; empty disables builtin pool')
    p.add_argument('--opponent-checkpoint', action='append', default=[], metavar='NAME=PATH',
                   help='repeat to add one frozen neural opponent per checkpoint')
    p.add_argument('--groups', type=int, default=40)
    p.add_argument('--seed', type=int, default=16_000_000)
    p.add_argument('--mode', choices=('sampled', 'greedy'), default='sampled')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--big-blind', type=int, default=100)
    p.add_argument('--max-decisions', type=int, default=200)
    p.add_argument('--entropy-batch-size', type=int, default=64)
    p.add_argument('--output-json', default='runs/phase16/evaluation.json')
    p.add_argument('--output-csv', default='runs/phase16/summary.csv')
    p.add_argument('--output-md', default='runs/phase16/report.md')
    p.add_argument('--no-progress', action='store_true')
    return p


def _specs(args) -> tuple[OpponentSpec, ...]:
    names = [x.strip() for x in args.opponents.split(',') if x.strip()]
    specs = [OpponentSpec(name) for name in names]
    for entry in args.opponent_checkpoint:
        if '=' not in entry:
            raise ValueError('--opponent-checkpoint must have NAME=PATH format')
        name, path = entry.split('=', 1)
        if not name or not path:
            raise ValueError('opponent checkpoint name and path must be provided')
        specs.append(OpponentSpec(name=name, checkpoint=path))
    if not specs or len({x.name for x in specs}) != len(specs):
        raise ValueError('opponent pool must be nonempty with unique names')
    return tuple(specs)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run(args) -> dict:
    specs = _specs(args)
    cfg = PoolEvaluationConfig(groups=args.groups, seed=args.seed,
                               mode=args.mode, device=args.device, big_blind=args.big_blind,
                               max_decisions=args.max_decisions,
                               entropy_batch_size=args.entropy_batch_size)
    json_path, csv_path, md_path = (Path(args.output_json), Path(args.output_csv), Path(args.output_md))
    paths = (json_path, csv_path, md_path)
    if len({p.resolve() for p in paths}) != 3:
        raise ValueError('JSON, CSV and Markdown outputs require distinct paths')
    if any(p.exists() for p in paths):
        raise FileExistsError('Phase16 output exists; use new paths to preserve comparisons')
    candidate = load_actor(args.candidate, device=cfg.device)
    bar = None
    if not args.no_progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=len(specs) * cfg.groups, desc='Phase16', unit='group')
        except ImportError:
            pass
    try:
        outcome = evaluate_opponent_pool(candidate, specs, config=cfg,
                                         on_group=(lambda name: bar.update() if bar is not None else None))
    finally:
        if bar is not None:
            bar.close()
    outcome['candidate_checkpoint'] = str(Path(args.candidate).resolve())
    outcome['opponent_specs'] = [dict(name=s.name, checkpoint=s.checkpoint) for s in specs]
    _atomic_write(json_path, json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    # CSV is human-readable summary. Detailed action diagnostics are in JSON.
    import io
    buf = io.StringIO(newline='')
    fields = ['opponent', 'opponent_id', 'hands', 'mean_bb100', 'ci95_lower_bb100',
              'ci95_upper_bb100', 'positive_group_fraction',
              'mean_policy_entropy', 'mean_normalized_entropy',
              'fold_when_legal', 'check_when_legal', 'call_when_legal',
              'raise_double_pot_when_legal', 'all_in_when_legal']
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for result in outcome['opponents']:
        action = result['candidate_action_profile']
        row = {k: result.get(k) for k in fields}
        for key in ('mean_policy_entropy', 'mean_normalized_entropy'):
            row[key] = action[key]
        for name in ('fold', 'check', 'call', 'raise_double_pot', 'all_in'):
            row[name + '_when_legal'] = action['conditional_action_rate'][name]
        writer.writerow(row)
    _atomic_write(csv_path, buf.getvalue())
    _atomic_write(md_path, markdown_report(outcome))
    print(json.dumps({
        'phase': 16, 'candidate_id': outcome['candidate_id'],
        'pool_summary': outcome['pool_summary'],
        'results': [{k: v for k, v in result.items()
                     if k in ('opponent', 'mean_bb100', 'ci95_lower_bb100',
                              'ci95_upper_bb100', 'hands')}
                    for result in outcome['opponents']],
        'output_json': str(json_path), 'output_csv': str(csv_path),
        'output_md': str(md_path),
    }, ensure_ascii=False, indent=2))
    return outcome


def main() -> None:
    run(parser().parse_args())


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase16_evaluate.py" -->

## scripts/phase16_ppo_ablation.py

Original Git blob SHA: `e8974af0585e765651af579628d92748db6648b1` (3897 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase16_ppo_ablation.py" sha="e8974af0585e765651af579628d92748db6648b1" -->

````````python
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
            "preflop_all_in_fraction": results["on_policy_behavior"]["preflop_explicit_allin_fraction"],
            "flop_decision_hand_fraction": results["on_policy_behavior"]["postflop_decision_hand_fraction"],
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

````````

<!-- END_PY_SOURCE path="scripts/phase16_ppo_ablation.py" -->

## scripts/phase16_ppo_stress.py

Original Git blob SHA: `e7deff678fe99096022703357df2e1abbe66ba27` (6453 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase16_ppo_stress.py" sha="e7deff678fe99096022703357df2e1abbe66ba27" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase16_ppo_stress.py" -->

## scripts/phase17_evaluate.py

Original Git blob SHA: `ddc4788e1d16225c0a14dc9b7166186ffb8d0115` (10446 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase17_evaluate.py" sha="ddc4788e1d16225c0a14dc9b7166186ffb8d0115" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase17_evaluate.py" -->

## scripts/phase17_gpu_retrain.py

Original Git blob SHA: `5dbfef8eaea66e0d9fa44dc592cb6378f41ef868` (3573 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase17_gpu_retrain.py" sha="5dbfef8eaea66e0d9fa44dc592cb6378f41ef868" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase17_gpu_retrain.py" -->

## scripts/phase5_rlcard_smoke.py

Original Git blob SHA: `dbcd804c51636f80d9836e1407f7bd654216a619` (880 bytes).

<!-- BEGIN_PY_SOURCE path="scripts/phase5_rlcard_smoke.py" sha="dbcd804c51636f80d9836e1407f7bd654216a619" -->

````````python
"""Run from repository root: uv run python scripts/phase5_rlcard_smoke.py"""
from __future__ import annotations

import rlcard
from holdem_ai.environment.rlcard_adapter import RLCardTurnDriver
from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.training.episode_runner import run_self_play


def main() -> None:
    env = rlcard.make("no-limit-holdem", config={"seed": 19})
    driver = RLCardTurnDriver(env, tuple(f"action_{i}" for i in range(env.num_actions)))
    baseline = {i: RandomLegalPolicy() for i in range(env.num_players)}
    result = run_self_play(driver, baseline, seed=19, max_decisions=500)
    print(f"players={len(result.rewards)} decisions={result.num_decisions} rewards={result.rewards}")
    for seat in range(len(result.rewards)):
        print(f"seat={seat} actions={len(result.for_seat(seat))}")


if __name__ == "__main__":
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase5_rlcard_smoke.py" -->
