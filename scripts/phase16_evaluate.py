"""No-training Phase16 benchmark for Phase14 and future Phase15 best.pt actors."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from holdem_agent_phase12.checkpoint import load_actor
from holdem_agent_phase16 import (BUILTIN_OPPONENTS, OpponentSpec,
                                   PoolEvaluationConfig, evaluate_opponent_pool)
from holdem_agent_phase16.report import markdown_report


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
