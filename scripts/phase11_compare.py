"""Compare separate PPO and QBoost JSONL logs at matched update counts.

This is descriptive only; single-run scores do not establish superiority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_log(path: str):
    p = Path(path)
    events = []
    for line_num, raw in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
        if raw.strip():
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError as err:
                # A torn FINAL line can result from sudden power loss.
                if line_num != len(p.read_text(encoding='utf-8').splitlines()):
                    raise ValueError(f'corrupt JSONL line {line_num}') from err
    return events


def compare_logs(ppo_log: str, qboost_log: str) -> dict:
    def collect(path):
        e = _read_log(path)
        by_update = {r['update']: r for r in e if r.get('kind') == 'update'}
        random_eval = {r['update']: r for r in e if r.get('kind') == 'evaluation'
                       and r.get('opponent') == 'random'}
        anchor_eval = {r['update']: r for r in e if r.get('kind') == 'evaluation'
                       and r.get('opponent') == 'frozen_anchor'}
        return by_update, random_eval, anchor_eval
    pu, pr, pa = collect(ppo_log)
    qu, qr, qa = collect(qboost_log)
    p_meta_path = Path(ppo_log).with_name('run.json')
    q_meta_path = Path(qboost_log).with_name('run.json')
    p_meta = json.loads(p_meta_path.read_text(encoding='utf-8')) if p_meta_path.exists() else None
    q_meta = json.loads(q_meta_path.read_text(encoding='utf-8')) if q_meta_path.exists() else None
    compatible_configs = bool(p_meta and q_meta and
        p_meta.get('algorithm') == 'ppo' and q_meta.get('algorithm') == 'qboost' and
        p_meta.get('network_config') == q_meta.get('network_config') and
        p_meta.get('ppo_config') == q_meta.get('ppo_config') and
        p_meta.get('initial_updates') == q_meta.get('initial_updates'))
    fresh_matched_design = bool(compatible_configs and
        p_meta['initial_updates'] == 0 and
        not any(p_meta.get('config', {}).get(k) for k in ('from_phase9', 'from_phase10', 'resume')) and
        not any(q_meta.get('config', {}).get(k) for k in ('from_phase9', 'from_phase10', 'resume')))
    shared = sorted(set(pu) & set(qu))
    if not shared:
        raise ValueError('no shared update index between experiments')
    per_update = []
    for update in shared:
        p, q = pu[update], qu[update]
        report = {'update': update,
                  'ppo_hands_per_second': p.get('hands_per_second_training'),
                  'qboost_hands_per_second': q.get('hands_per_second_training'),
                  'ppo_critic_loss': p.get('critic_loss'),
                  'qboost_critic_loss': q.get('critic_loss'),
                  'qboost_logit_variance_ratio': q.get('logit_variance_ratio')}
        if update in pr and update in qr:
            report['ppo_random_bb100'] = pr[update]['mean_bb100']
            report['qboost_random_bb100'] = qr[update]['mean_bb100']
            report['qboost_minus_ppo_random_bb100'] = round(
                qr[update]['mean_bb100'] - pr[update]['mean_bb100'], 4)
        if update in pa and update in qa:
            report['ppo_frozen_anchor_bb100'] = pa[update]['mean_bb100']
            report['qboost_frozen_anchor_bb100'] = qa[update]['mean_bb100']
        per_update.append(report)
    return {'phase': 11, 'shared_updates': len(shared), 'last_shared_update': shared[-1],
            'matching_configs': compatible_configs,
            'fresh_matched_initialization_design': fresh_matched_design,
            'caution': 'Different policies generate different on-policy hands; tiny-sample bb/100 deltas are not significance tests.',
            'records': per_update}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ppo-log', required=True)
    p.add_argument('--qboost-log', required=True)
    args = p.parse_args()
    print(json.dumps(compare_logs(args.ppo_log, args.qboost_log), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
