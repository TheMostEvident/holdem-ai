"""Phase12 independent paired-deck evaluator for Phase9/10/11 local checkpoints.

uv run --extra gpu python -m scripts.phase12_evaluate \
  --candidate checkpoints/phase11_pilot.pt \
  --opponent-checkpoint runs/phase11_pilot/anchor.pt \
  --groups 50 --partition validation --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from contextlib import nullcontext

from holdem_ai.evaluation.seed_blocked import EvaluationConfig, compare_results, evaluate_seed_blocks
from holdem_ai.evaluation.checkpoint_loader import load_actor

SEED_PARTITIONS = {'validation': 12_000_000, 'test': 15_000_000}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase12 paired deck-seed HUNL200 evaluation (no training)')
    p.add_argument('--candidate', required=True, help='Phase9/10/11 local checkpoint')
    p.add_argument('--control', default=None, help='optional PPO/control actor; evaluates paired delta')
    p.add_argument('--opponent-checkpoint', default=None,
                   help='frozen actor checkpoint; default opponent is RandomLegalPolicy')
    p.add_argument('--groups', type=int, default=50, help='independent decks; 4 hands each; >=2')
    p.add_argument('--partition', choices=('validation', 'test'), default='validation')
    p.add_argument('--seed', type=int, default=None, help='override partition base seed')
    p.add_argument('--mode', choices=('greedy', 'sampled'), default='greedy')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--big-blind', type=int, default=100)
    p.add_argument('--max-decisions', type=int, default=200)
    p.add_argument('--no-progress', action='store_true', help='hide tqdm ETA progress')
    p.add_argument('--output-json', default=None)
    p.add_argument('--output-csv', default=None, help='write 4 paired returns per deck seed')
    return p


def run(args: argparse.Namespace) -> dict:
    paths = [Path(x).resolve() for x in (args.candidate, args.control, args.opponent_checkpoint) if x]
    if len(paths) != len(set(paths)):
        raise ValueError('candidate, control and opponent checkpoint paths must be distinct')
    seed = SEED_PARTITIONS[args.partition] if args.seed is None else args.seed
    config = EvaluationConfig(groups=args.groups, seed=seed, mode=args.mode,
                              device=args.device, big_blind=args.big_blind,
                              max_decisions=args.max_decisions)
    candidate = load_actor(args.candidate, device=args.device)
    control = load_actor(args.control, device=args.device) if args.control else None
    opponent = load_actor(args.opponent_checkpoint, device=args.device) if args.opponent_checkpoint else None
    # tqdm always writes to stderr; stdout stays a machine-readable JSON object.
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    bar = (tqdm(total=config.groups * (2 if control is not None else 1),
                desc='Phase 12 evaluation', unit='group', disable=args.no_progress)
           if tqdm is not None and not args.no_progress else nullcontext())
    with bar as progress:
        callback = progress.update if tqdm is not None and not args.no_progress else None
        result = evaluate_seed_blocks(candidate, opponent=opponent, config=config,
                                      on_group=(lambda: callback(1)) if callback is not None else None)
        output = {'phase': 12, 'partition': args.partition, 'candidate': result.summary()}
        if control is not None:
            baseline = evaluate_seed_blocks(control, opponent=opponent, config=config,
                                            on_group=(lambda: callback(1)) if callback is not None else None)
            output['control'] = baseline.summary()
            output['paired_comparison'] = compare_results(result, baseline)
    if args.output_csv:
        file = Path(args.output_csv)
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('seed', 'button0_actor0', 'button0_actor1',
                             'button1_actor0', 'button1_actor1', 'candidate_group_mean_bb',
                             'control_group_mean_bb', 'delta_group_mean_bb'))
            for i, group in enumerate(result.blocks):
                other = baseline.blocks[i].average_bb if control is not None else None
                writer.writerow((group.seed, *group.payoffs_bb, group.average_bb,
                                 other, group.average_bb - other if other is not None else None))
    if args.output_json:
        file = Path(args.output_json)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return output


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
