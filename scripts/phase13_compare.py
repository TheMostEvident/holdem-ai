"""Compare Phase13 PPO and QBoost training diagnostics without claiming GTO.

Reads completed metrics.jsonl streams. Only comparison runs sharing identical
source checkpoint and rollout schedule are labeled matched-start.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_run(root: str | Path) -> dict:
    root = Path(root)
    meta = json.loads((root / 'run.json').read_text(encoding='utf-8'))
    if meta.get('phase') != 13:
        raise ValueError('expected Phase13 run metadata')
    rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    updates = [r for r in rows if r.get('kind') == 'update']
    if not updates:
        raise ValueError('no updates in run')
    total_hands = sum(int(r['hands']) for r in updates)
    train_seconds = sum(float(r['elapsed_seconds']) for r in updates)
    def weighted_mean(field):
        values = [(float(row[field]), int(row['hands'])) for row in updates if field in row]
        return (round(sum(x * n for x, n in values) / sum(n for _, n in values), 6)
                if values else None)
    statistics = {
        'algorithm': meta['algorithm'],
        'completed_logged_updates': len(updates),
        'first_update': updates[0]['update'], 'last_update': updates[-1]['update'],
        'training_hands': total_hands,
        'aggregate_training_hands_per_second': round(total_hands / train_seconds, 3),
        'mean_actor_entropy': weighted_mean('entropy'),
        'mean_approx_kl': weighted_mean('approx_kl'),
        'mean_critic_loss': weighted_mean('critic_loss'),
        'mean_logit_variance_ratio': weighted_mean('logit_variance_ratio'),
        'total_fold_actions': sum(row['action_diagnostics']['actions']['fold'] for row in updates),
        'total_check_actions': sum(row['action_diagnostics']['actions']['check'] for row in updates),
        'total_fold_opportunities': sum(row['action_diagnostics']['legal_opportunities']['fold'] for row in updates),
        'total_check_opportunities': sum(row['action_diagnostics']['legal_opportunities']['check'] for row in updates),
    }
    for label in ('fold', 'check'):
        opportunities = statistics[f'total_{label}_opportunities']
        statistics[f'{label}_rate_when_legal'] = (
            round(statistics[f'total_{label}_actions'] / opportunities, 6)
            if opportunities else None)
    return {'metadata': meta, 'summary': statistics}


def compare(ppo_log_dir, qboost_log_dir):
    ppo, qb = load_run(ppo_log_dir), load_run(qboost_log_dir)
    if (ppo['metadata']['algorithm'], qb['metadata']['algorithm']) != ('ppo', 'qboost'):
        raise ValueError('first run must be PPO, second QBoost')
    pc, qc = ppo['metadata']['ppo_config'], qb['metadata']['ppo_config']
    equal_training = pc == qc
    same_initial_ckpt = (ppo['metadata']['source_checkpoint'] is not None and
                         ppo['metadata']['source_checkpoint'] == qb['metadata']['source_checkpoint'])
    p, q = ppo['summary'], qb['summary']
    return {'phase': 13, 'protocol': 'train_log_descriptive_comparison',
            'matched_start_and_hyperparameters': same_initial_ckpt and equal_training,
            'matched_start_only': same_initial_ckpt,
            'ppo': p, 'qboost': q,
            'qboost_vs_ppo_training_speed_ratio': round(
                q['aggregate_training_hands_per_second'] /
                max(p['aggregate_training_hands_per_second'], 1e-12), 5),
            'note': 'These are training diagnostics, not paired gameplay payoffs or GTO exploitability.'}


def main():
    parser = argparse.ArgumentParser(description='Compare Phase13 PPO/QBoost log metrics')
    parser.add_argument('--ppo-log-dir', required=True)
    parser.add_argument('--qboost-log-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.ppo_log_dir, args.qboost_log_dir),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
