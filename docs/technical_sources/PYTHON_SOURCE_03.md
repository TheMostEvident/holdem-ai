# Python source code: volume 3 of 28

Exact full .py sources from baseline commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. Integral appendix of [final technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md).

## scripts/gui15_smoke.py

Original Git blob SHA: `bf28285a1a270aaec53825368f0fa3c9a67cf08b` (3671 bytes).

<a id="py-025"></a>

<!-- BEGIN_PY_SOURCE path="scripts/gui15_smoke.py" sha="bf28285a1a270aaec53825368f0fa3c9a67cf08b" -->

````````python
"""Exercise GUI-1.5 HTTP end-to-end without launching React/browser.

Run: uv run --extra dev --extra gui python -m scripts.gui15_smoke --hands 20
"""
from __future__ import annotations

import argparse
import json
import math

from fastapi.testclient import TestClient

from backend.main import create_app

BASE = '/api/v1/hunl'


def run(*, hands: int = 20, seed: int = 42) -> dict:
    if type(hands) is not int or hands < 1:
        raise ValueError('hands must be a positive integer')
    with TestClient(create_app()) as client:
        r = client.post(f'{BASE}/sessions', json={'bot': 'calling_station', 'seed': seed})
        r.raise_for_status()
        data = r.json()
        session_id = data['session_id']
        state = data['state']
        for h in range(hands):
            for decision in range(200):
                if state['phase'] == 'complete':
                    break
                choices = state['actions']
                action = next((a for a in choices if a['kind'] == 'check'), None)
                if action is None:
                    action = next((a for a in choices if a['kind'] == 'call'), None)
                if action is None:
                    action = choices[0]
                r = client.post(f'{BASE}/sessions/{session_id}/actions', json={
                    'action_index': action['index'], 'expected_revision': state['revision']})
                r.raise_for_status()
                state = r.json()['state']
            else:
                raise AssertionError('game did not finish within 200 human actions')
            if h + 1 < hands:
                r = client.post(f'{BASE}/sessions/{session_id}/next-hand',
                                json={'expected_revision': state['revision']})
                r.raise_for_status()
                state = r.json()['state']
        r = client.get(f'{BASE}/sessions/{session_id}/stats')
        r.raise_for_status()
        stats = r.json()
        cursor, numbers = 0, []
        while True:
            r = client.get(f'{BASE}/sessions/{session_id}/history',
                           params={'offset': cursor, 'limit': 20})
            r.raise_for_status()
            page = r.json()
            for row in page['items']:
                numbers.append(row['hand_number'])
                if row['result']['showdown'] is False:
                    assert row['result']['opponent_cards'] is None
            cursor += len(page['items'])
            if not page['has_more']:
                break
        assert stats['completed_hands'] == hands
        assert stats['wins'] + stats['losses'] + stats['break_even'] == hands
        assert math.isclose(stats['net_bb'], state['cumulative_bb'], abs_tol=1e-8)
        assert numbers == list(range(hands, max(0, hands-500), -1))
        assert sum(state['hand_result']['payoffs_chips']) == 0
        summary = {'status': 'PASS', 'completed_hands': hands,
                   'retained_records': len(numbers),
                   'wins': stats['wins'], 'losses': stats['losses'],
                   'even': stats['break_even'],
                   'net_bb': round(stats['net_bb'], 4),
                   'bb_per_100': round(stats['bb_per_100'], 4)}
        client.delete(f'{BASE}/sessions/{session_id}').raise_for_status()
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='GUI-1.5 HTTP history & metrics smoke test')
    parser.add_argument('--hands', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(hands=args.hands, seed=args.seed), indent=2))


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/gui15_smoke.py" -->

## scripts/gui16_smoke.py

Original Git blob SHA: `698b824bc59b0c9ca9eeca2f3dfbe1d249204b25` (2543 bytes).

<a id="py-026"></a>

<!-- BEGIN_PY_SOURCE path="scripts/gui16_smoke.py" sha="698b824bc59b0c9ca9eeca2f3dfbe1d249204b25" -->

````````python
"""Phase GUI-1.6 local HTTP multi-hand carryover smoke.

Run: uv run --extra dev --extra gui --extra gpu python -m scripts.gui16_smoke
"""
from __future__ import annotations

import argparse

from fastapi.testclient import TestClient
from backend.main import create_app


def run(hands: int = 20, seed: int = 42) -> None:
    if hands < 2:
        raise ValueError('hands must be >=2')
    with TestClient(create_app()) as client:
        start = client.post('/api/v1/hunl/sessions', json={
            'bot': 'calling_station', 'seed': seed, 'stack_mode': 'continuous'})
        start.raise_for_status()
        session_id, view = start.json()['session_id'], start.json()['state']
        last_final = None
        for h in range(1, hands + 1):
            if last_final is not None:
                assert view['starting_stacks'] == last_final
            initial = view['starting_stacks']
            assert sum(initial) == 40000
            for decision in range(100):
                if view['phase'] == 'complete':
                    break
                selected = next(x for x in view['actions'] if x['kind'] == 'fold')
                response = client.post(f'/api/v1/hunl/sessions/{session_id}/actions', json={
                    'action_index': selected['index'], 'expected_revision': view['revision']})
                response.raise_for_status()
                view = response.json()['state']
            else:
                raise AssertionError('hand did not finish')
            last_final = view['hand_result']['final_stacks']
            assert sum(last_final) == sum(initial)
            assert view['completed_hands'] == h
            print(f'hand={h:02d} start={initial} final={last_final} button={view["button"]}')
            if view['match_over'] or h == hands:
                break
            response = client.post(f'/api/v1/hunl/sessions/{session_id}/next-hand', json={
                'expected_revision': view['revision']})
            response.raise_for_status()
            view = response.json()['state']
        stats = client.get(f'/api/v1/hunl/sessions/{session_id}/stats').json()
        assert stats['net_chips'] == last_final[0] - 20000
        print(f'PASS: {stats["completed_hands"]} completed hands; net {stats["net_bb"]} BB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.hands, args.seed)


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/gui16_smoke.py" -->

## scripts/phase10_train.py

Original Git blob SHA: `1bff9b10282a0dc301cb73aff8f953e111344b45` (8433 bytes).

<a id="py-027"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase10_train.py" sha="1bff9b10282a0dc301cb73aff8f953e111344b45" -->

````````python
"""Phase 10 experimental Q-Boost PPO; optional matched-start PPO comparator.

Execute at repository root: uv run --extra gpu python -m scripts.phase10_train ...
No optimizer checkpoints are committed to Git. All training uses on-policy data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig, collect_self_play, evaluate_vs_random, freeze_old_policy_data, optimize_ppo
from holdem_ai.training.ppo_session import create_session, save_checkpoint
from holdem_ai.training.qboost import QBoostConfig
from holdem_ai.training.qboost_session import create_qboost_session, from_phase9_checkpoint, load_qboost_checkpoint, train_qboost_updates


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase10 doubly-corrected Q-Boosting PPO experiment')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--hands-per-update', type=int, default=16)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--minibatch-size', type=int, default=32)
    p.add_argument('--eval-hands', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--checkpoint', default='checkpoints/phase10_qboost.pt')
    p.add_argument('--resume', default=None, help='Phase10 QBoost checkpoint, continues saved hyperparameters')
    p.add_argument('--from-phase9', default=None, help='warm-start from Phase9 PPO checkpoint')
    p.add_argument('--compare', action='store_true', help='train separate matched-start on-policy PPO control')
    p.add_argument('--ppo-checkpoint', default='checkpoints/phase10_ppo_control.pt')
    p.add_argument('--skip-eval', action='store_true')
    p.add_argument('--small-network', action='store_true')
    return p


def run(args: argparse.Namespace) -> dict:
    if args.resume and args.from_phase9:
        raise ValueError('--resume and --from-phase9 cannot both be set')
    if args.compare and (args.resume or args.from_phase9):
        raise ValueError('--compare requires a fresh matched-start run')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    if args.compare and Path(args.checkpoint).resolve() == Path(args.ppo_checkpoint).resolve():
        raise ValueError('PPO and QBoost checkpoints must have different paths')
    begin = time.perf_counter()
    if args.resume:
        session = load_qboost_checkpoint(args.resume, device=args.device)
        checkpoint_path = (args.resume if args.checkpoint == 'checkpoints/phase10_qboost.pt'
                           else args.checkpoint)
    elif args.from_phase9:
        session = from_phase9_checkpoint(args.from_phase9, device=args.device,
                                          qconfig=QBoostConfig(beta=args.beta))
        checkpoint_path = args.checkpoint
    else:
        net = (NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                             ff_dim=64, hidden_dim=48)
               if args.small_network else NetworkConfig())
        cfg = PPOConfig(hands_per_update=args.hands_per_update,
                        epochs=args.epochs, minibatch_size=args.minibatch_size,
                        seed=args.seed)
        session = create_qboost_session(device=args.device, seed=args.seed,
                                       network_config=net, ppo_config=cfg,
                                       qconfig=QBoostConfig(beta=args.beta))
        checkpoint_path = args.checkpoint
    base = session.base
    eval_seed = 900000 + base.ppo_config.seed * 10
    def evaluate(actor):
        return evaluate_vs_random(actor, hands=args.eval_hands, device=base.device,
                                  seed=eval_seed, big_blind=base.ppo_config.big_blind)
    first = None if args.skip_eval else evaluate(base.actor)
    if first is not None:
        print(json.dumps({'kind': 'evaluation', 'agent': 'qboost', 'point': 'before', **first}), flush=True)
    # Compare: distinct on-policy actors, common initial weights and seed schedule.
    # After update 1 trajectories can diverge; do NOT train both on same stale data.
    control = None
    control_first = None
    control_reports = []
    if args.compare:
        control = create_session(device=base.device, seed=base.ppo_config.seed,
                                 network_config=base.network_config,
                                 ppo_config=base.ppo_config)
        # create_session reseeds torch but self-play samples use local Python RNG.
        control.actor.load_state_dict(base.actor.state_dict())
        control.critic.load_state_dict(base.critic.state_dict())
        control_first = None if args.skip_eval else evaluate(control.actor)
        if first is not None and control_first != first:
            raise AssertionError('comparison must start with identical evaluation')
    reports = []
    for _ in range(args.updates):
        # The comparator update uses its OWN rollout and frozen old policy.
        # Both use common seed schedule and stacks, not identical realized paths.
        reports.extend(train_qboost_updates(session, updates=1, checkpoint=checkpoint_path))
        print(json.dumps({'kind': 'update', 'agent': 'qboost', **reports[-1]}), flush=True)
        if control is not None:
            tic = time.perf_counter()
            it = control.completed_updates
            rows = collect_self_play(control.actor, control.ppo_config, iteration=it, device=control.device)
            frozen = freeze_old_policy_data(control.actor, control.critic, rows,
                                            device=control.device,
                                            chunk_size=control.ppo_config.inference_batch_size)
            stats = optimize_ppo(control.actor, control.critic, control.actor_optimizer,
                                 control.critic_optimizer, frozen, control.ppo_config,
                                 device=control.device,
                                 shuffle_seed=control.ppo_config.seed + 100003 * it)
            control.completed_updates += 1
            elapsed = time.perf_counter() - tic
            stats.update({'update': control.completed_updates, 'hands': rows.hands,
                          'elapsed_seconds': round(elapsed, 3),
                          'hands_per_second_training': round(rows.hands/max(elapsed, 1e-9), 3)})
            control_reports.append(stats)
            save_checkpoint(control, args.ppo_checkpoint)
            print(json.dumps({'kind': 'update', 'agent': 'ppo_control', **stats}), flush=True)
    last = None if args.skip_eval else evaluate(base.actor)
    control_last = None if control is None or args.skip_eval else evaluate(control.actor)
    for agent, report in (('qboost', last), ('ppo_control', control_last)):
        if report is not None:
            print(json.dumps({'kind': 'evaluation', 'agent': agent, 'point': 'after', **report}), flush=True)
    if base.device == 'cuda':
        torch.cuda.synchronize()
    return {
        'phase': 10, 'algorithm': 'qboost_ppo_doubly_corrected_lite',
        'paper_reproduction': False, 'trained_policy': True,
        'device': base.device, 'torch': torch.__version__,
        'completed_updates': base.completed_updates, 'beta': session.qconfig.beta,
        'checkpoint': str(checkpoint_path),
        'actor_parameters': sum(p.numel() for p in base.actor.parameters()),
        'critic_parameters': sum(p.numel() for p in base.critic.parameters()),
        'last_update': reports[-1] if reports else None,
        'evaluation_before': first, 'evaluation_after': last,
        'ppo_control_last_update': control_reports[-1] if control_reports else None,
        'ppo_control_evaluation_before': control_first,
        'ppo_control_evaluation_after': control_last,
        'elapsed_seconds': round(time.perf_counter() - begin, 3),
        'cuda_peak_allocated_mb': round(torch.cuda.max_memory_allocated()/2**20, 2)
        if base.device == 'cuda' else None,
    }


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase10_train.py" -->

## scripts/phase11_benchmark.py

Original Git blob SHA: `89766f73f3df1043889b728cec6994418c975ce0` (2925 bytes).

<a id="py-028"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase11_benchmark.py" sha="89766f73f3df1043889b728cec6994418c975ce0" -->

````````python
"""Measure inference batching gains without PPO optimizer steps."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.batched_collector import collect_batched_self_play


def run(*, device='cpu', hands=100, batch_sizes=(1, 4, 8, 16), seed=42,
        small_network=False) -> list[dict]:
    if type(hands) is not int or hands < 1:
        raise ValueError('hands must be a positive integer')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.manual_seed(seed)
    net = (NetworkConfig(embed_dim=32, transformer_layers=1, attention_heads=2,
                         ff_dim=64, hidden_dim=48)
           if small_network else NetworkConfig())
    actor = TransformerActor(net).to(device).eval()
    cfg = PPOConfig(hands_per_update=hands, seed=seed)
    warm = replace(cfg, hands_per_update=min(hands, 4))
    collect_batched_self_play(actor, warm, device=device, env_batch_size=1)
    reports = []
    for n in batch_sizes:
        if type(n) is not int or n < 1:
            raise ValueError('batch size must be positive')
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rows = collect_batched_self_play(actor, cfg, iteration=0,
                                         device=device, env_batch_size=n)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        reports.append({
            'phase': 11, 'device': device, 'hands': rows.hands,
            'decisions': len(rows.actions), 'env_batch_size': n,
            'elapsed_seconds': round(elapsed, 4),
            'hands_per_second': round(rows.hands / elapsed, 3),
            'decisions_per_second': round(len(rows.actions) / elapsed, 3),
            'peak_cuda_allocated_mb': (round(torch.cuda.max_memory_allocated()/2**20, 3)
                                       if device == 'cuda' else None),
        })
    return reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--hands', type=int, default=100)
    p.add_argument('--batch-sizes', default='1,4,8,16')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--small-network', action='store_true')
    args = p.parse_args()
    for item in run(device=args.device, hands=args.hands,
                    batch_sizes=tuple(int(x) for x in args.batch_sizes.split(',')),
                    seed=args.seed, small_network=args.small_network):
        print(json.dumps(item, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase11_benchmark.py" -->

## scripts/phase11_compare.py

Original Git blob SHA: `bc400682ee0a9f99af2b185fbeee9034646a0942` (4163 bytes).

<a id="py-029"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase11_compare.py" sha="bc400682ee0a9f99af2b185fbeee9034646a0942" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase11_compare.py" -->

## scripts/phase11_train.py

Original Git blob SHA: `296b330148c5765b110980cb3a471d7b003e25ee` (4449 bytes).

<a id="py-030"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase11_train.py" sha="296b330148c5765b110980cb3a471d7b003e25ee" -->

````````python
"""Phase 11 training CLI with real-time progress, time-left and ETA.

The JSONL log and per-update atomic checkpoints remain owned by the original
Phase-11 runner. The progress display only observes existing runner events.

Example (from project root):
  uv run --extra gpu python -m scripts.phase11_train --device cuda --duration-minutes 15 --max-updates 200

Optional tqdm visuals:
  uv add tqdm
Without tqdm, a dependency-free terminal bar is used automatically.
"""
from __future__ import annotations

import argparse
import json

from holdem_ai.experiments.time_budget import ExperimentConfig, run_experiment
from holdem_ai.utils.progress import TrainingProgress


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase11 batched HUNL200 training with live ETA and checkpoints')
    p.add_argument('--algorithm', choices=('ppo', 'qboost'), default='qboost')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--duration-minutes', type=float, default=300.0,
                   help='wall-clock budget for THIS invocation (default five hours)')
    p.add_argument('--max-updates', type=int, default=None,
                   help='limit ADDITIONAL updates in this invocation')
    p.add_argument('--env-batch-size', type=int, default=8,
                   help='simultaneously active hands for one actor forward pass')
    p.add_argument('--hands-per-update', type=int, default=32)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--minibatch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--eval-hands', type=int, default=80)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--log-dir', default=None)
    p.add_argument('--resume', default=None)
    p.add_argument('--from-phase9', default=None)
    p.add_argument('--from-phase10', default=None)
    p.add_argument('--small-network', action='store_true')
    p.add_argument('--no-progress', action='store_true',
                   help='disable live display and print all legacy JSON events')
    p.add_argument('--json-events', action='store_true',
                   help='print every update JSON in addition to progress (normally only saved in metrics.jsonl)')
    p.add_argument('--progress-interval', type=float, default=1.0,
                   help='live display refresh interval in seconds (>=0.1)')
    return p


def run(args: argparse.Namespace) -> dict:
    checkpoint = args.checkpoint or (args.resume if args.resume is not None
                                     else f'checkpoints/phase11_{args.algorithm}.pt')
    logs = args.log_dir or f'runs/phase11_{args.algorithm}'
    cfg = ExperimentConfig(
        algorithm=args.algorithm, device=args.device,
        duration_seconds=args.duration_minutes * 60,
        max_updates=args.max_updates, env_batch_size=args.env_batch_size,
        eval_every=args.eval_every, eval_hands=args.eval_hands,
        hands_per_update=args.hands_per_update, epochs=args.epochs,
        minibatch_size=args.minibatch_size, seed=args.seed,
        beta=args.beta, checkpoint=checkpoint, log_dir=logs,
        resume=args.resume, from_phase9=args.from_phase9,
        from_phase10=args.from_phase10, small_network=args.small_network,
    )
    if args.no_progress:
        return run_experiment(cfg, emit=lambda d: print(json.dumps(d, ensure_ascii=False), flush=True))

    with TrainingProgress(duration_seconds=cfg.duration_seconds,
                          max_updates=cfg.max_updates,
                          interval=args.progress_interval) as progress:
        def emit(event: dict) -> None:
            progress.on_event(event)
            # Full per-step data continues to be written in metrics.jsonl.
            # Print evaluations and final information, rather than a flood of
            # JSON lines that would obscure the live progress meter.
            if args.json_events or event.get('kind') in ('start', 'resume', 'evaluation', 'interrupt', 'end'):
                progress.message(json.dumps(event, ensure_ascii=False))

        return run_experiment(cfg, emit=emit)


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase11_train.py" -->

## scripts/phase12_evaluate.py

Original Git blob SHA: `451bc3f8abed3c239464251601ae1f2b8cc0eed2` (5139 bytes).

<a id="py-031"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase12_evaluate.py" sha="451bc3f8abed3c239464251601ae1f2b8cc0eed2" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase12_evaluate.py" -->

## scripts/phase13_benchmark.py

Original Git blob SHA: `67fcd569e1e15cce30b345139a5bce6f6accbac8` (4229 bytes).

<a id="py-032"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase13_benchmark.py" sha="67fcd569e1e15cce30b345139a5bce6f6accbac8" -->

````````python
"""Measure ACTUAL training update throughput for each hands/update + env batch pair.

These are full PPO/QBoost gradient-update measurements, unlike pure forward
benchmark from Phase11. Every case uses an independent run and a fresh or
identically warm-started checkpoint. The first update includes warmup.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from holdem_ai.experiments.batch_scaling import BatchExperimentConfig, run_batch_experiment


def parse_sizes(value: str) -> list[int]:
    try:
        numbers = [int(s.strip()) for s in value.split(',')]
    except ValueError as e:
        raise ValueError('comma-separated positive integers required') from e
    if not numbers or any(n < 1 for n in numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('batch sizes must be unique positive integers')
    return numbers


def build_parser():
    parser = argparse.ArgumentParser(description='Phase13 real training throughput benchmark')
    parser.add_argument('--algorithm', choices=('ppo', 'qboost'), default='qboost')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--hands-per-update', default='16,64,128')
    parser.add_argument('--env-batch-sizes', default='16,32,64')
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--duration-minutes', type=float, default=5.0)
    parser.add_argument('--from-checkpoint', default=None)
    parser.add_argument('--root', default='runs/phase13_bench')
    parser.add_argument('--small-network', action='store_true')
    return parser


def run(args):
    if args.updates < 1:
        raise ValueError('updates must be >=1')
    h_sizes = parse_sizes(args.hands_per_update)
    env_sizes = parse_sizes(args.env_batch_sizes)
    root = Path(args.root)
    table = []
    for h in h_sizes:
        for env in env_sizes:
            if env > h:
                continue   # tests an effective, not merely nominal, env batch
            case = root / f'h{h}_b{env}'
            cfg = BatchExperimentConfig(
                algorithm=args.algorithm, device=args.device,
                duration_seconds=args.duration_minutes * 60,
                max_updates=args.updates, hands_per_update=h,
                minibatch_size=args.minibatch_size, epochs=args.epochs,
                env_batch_size=env, skip_eval=True,
                small_network=args.small_network,
                from_checkpoint=args.from_checkpoint,
                checkpoint=str(case / 'model.pt'), log_dir=str(case / 'logs'))
            print(json.dumps({'kind': 'case_start', 'hands_per_update': h,
                              'env_batch_size': env}), flush=True)
            result = run_batch_experiment(cfg)
            update_rows = [json.loads(line) for line in
                           (case / 'logs' / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            records = [x for x in update_rows if x['kind'] == 'update']
            # Drop update 1 as warmup if available.
            measured = records[1:] if len(records) > 1 else records
            total_hands = sum(x['hands'] for x in measured)
            elapsed = sum(x['elapsed_seconds'] for x in measured)
            rate = total_hands / elapsed if elapsed > 0 else 0.0
            item = {'hands_per_update': h, 'env_batch_size': env,
                    'completed_updates': len(records), 'measured_updates': len(measured),
                    'full_training_hands_per_second': round(rate, 3),
                    'cuda_peak_allocated_mb': result['cuda_peak_allocated_mb'],
                    'checkpoint': result['checkpoint']}
            table.append(item)
            print(json.dumps({'kind': 'case_end', **item}), flush=True)
    return {'phase': 13, 'algorithm': args.algorithm,
            'benchmark': 'real_training_including_optimizer',
            'results': table}


def main():
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase13_benchmark.py" -->

## scripts/phase13_compare.py

Original Git blob SHA: `0efa2bb403ecc22f627c1f946e323c2b5f719758` (4027 bytes).

<a id="py-033"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase13_compare.py" sha="0efa2bb403ecc22f627c1f946e323c2b5f719758" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase13_compare.py" -->

## scripts/phase13_train.py

Original Git blob SHA: `fde50a27a8585828ec663cefa82080e24a4b6468` (4257 bytes).

<a id="py-034"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase13_train.py" sha="fde50a27a8585828ec663cefa82080e24a4b6468" -->

````````python
"""Phase 13: PPO/QBoost batch scaling, checkpoint lineage, live tqdm ETA.

From repo root:
  uv run --extra gpu python -m scripts.phase13_train --device cuda --algorithm qboost --from-checkpoint checkpoints/phase11_pilot.pt --hands-per-update 64 --env-batch-size 32 --max-updates 2
"""
from __future__ import annotations

import argparse
import json

from holdem_ai.utils.progress import TrainingProgress
from holdem_ai.experiments.batch_scaling import BatchExperimentConfig, run_batch_experiment


class Phase13Progress(TrainingProgress):
    def _open_display(self) -> None:
        super()._open_display()
        if self._bar is not None:
            self._bar.set_description('Phase 13')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase13 HUNL PPO/QBoost training batch scaling')
    p.add_argument('--algorithm', choices=('ppo', 'qboost'), default='qboost')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--duration-minutes', type=float, default=15.0)
    p.add_argument('--max-updates', type=int, default=200, help='ADDITIONAL updates')
    p.add_argument('--hands-per-update', type=int, default=None, help='default 64 for a NEW phase13 stage')
    p.add_argument('--minibatch-size', type=int, default=None, help='default 64 for a NEW phase13 stage')
    p.add_argument('--epochs', type=int, default=None, help='default 2 for a NEW phase13 stage')
    p.add_argument('--rollout-seed', type=int, default=None, help='optional new self-play seed namespace')
    p.add_argument('--env-batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--eval-groups', type=int, default=40, help='Phase12 four-layout seed groups')
    p.add_argument('--eval-mode', choices=('sampled', 'greedy'), default='sampled')
    p.add_argument('--skip-eval', action='store_true')
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--from-checkpoint', default=None, help='warm start a NEW experiment; source never overwritten')
    p.add_argument('--resume', default=None, help='continue Phase13 run in SAME checkpoint/log-dir')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--log-dir', default=None)
    p.add_argument('--small-network', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.add_argument('--json-events', action='store_true')
    p.add_argument('--progress-interval', type=float, default=1.0)
    return p


def run(args: argparse.Namespace) -> dict:
    checkpoint = args.checkpoint or args.resume or f'checkpoints/phase13_{args.algorithm}.pt'
    logs = args.log_dir or f'runs/phase13_{args.algorithm}'
    cfg = BatchExperimentConfig(
        algorithm=args.algorithm, device=args.device,
        duration_seconds=args.duration_minutes * 60,
        max_updates=args.max_updates,
        hands_per_update=args.hands_per_update,
        minibatch_size=args.minibatch_size, epochs=args.epochs,
        rollout_seed=args.rollout_seed,
        env_batch_size=args.env_batch_size, eval_every=args.eval_every,
        eval_groups=args.eval_groups, eval_mode=args.eval_mode,
        skip_eval=args.skip_eval, beta=args.beta,
        checkpoint=checkpoint, log_dir=logs,
        from_checkpoint=args.from_checkpoint, resume=args.resume,
        small_network=args.small_network)
    if args.no_progress:
        return run_batch_experiment(cfg,
            emit=lambda evt: print(json.dumps(evt, ensure_ascii=False), flush=True))
    with Phase13Progress(duration_seconds=cfg.duration_seconds,
                         max_updates=cfg.max_updates,
                         interval=args.progress_interval) as progress:
        def on_event(evt: dict):
            progress.on_event(evt)
            if args.json_events or evt.get('kind') in ('start', 'resume', 'evaluation', 'interrupt', 'end'):
                # Keep detailed update/action histograms in metrics.jsonl.
                progress.message(json.dumps(evt, ensure_ascii=False))
        return run_batch_experiment(cfg, emit=on_event)


def main():
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase13_train.py" -->

## scripts/phase14_study.py

Original Git blob SHA: `174f15d82750584d8dd65e1a6cb1a92f8346c364` (3502 bytes).

<a id="py-035"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase14_study.py" sha="174f15d82750584d8dd65e1a6cb1a92f8346c364" -->

````````python
"""Multi-seed controlled PPO/QBoost HUNL experiment.

Example from repo root:
  uv run --extra gpu python -m scripts.phase14_study --device cuda --source-checkpoint checkpoints/phase9_ppo.pt --seeds 41,42,43 --updates 100
"""
from __future__ import annotations

import argparse
import json

from holdem_ai.experiments.controlled_study import ComparisonConfig, run_study


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(x.strip()) for x in value.split(','))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError('seeds must be comma-separated integers') from exc
    if not result or len(set(result)) != len(result) or any(not 0 <= s < 100_000 for s in result):
        raise argparse.ArgumentTypeError('seeds must be unique integers in [0,100000)')
    return result


def build_parser():
    p = argparse.ArgumentParser(description='Phase14 preregistered PPO vs QBoost controlled multi-seed study')
    p.add_argument('--source-checkpoint', required=True, help='trusted FULL Phase9 checkpoint, common start')
    p.add_argument('--root', default='runs/phase14_study')
    p.add_argument('--seeds', type=parse_seeds, default=(41, 42, 43))
    p.add_argument('--updates', type=int, default=100, help='equal number of updates per arm')
    p.add_argument('--hands-per-update', type=int, default=64)
    p.add_argument('--env-batch-size', type=int, default=32)
    p.add_argument('--minibatch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--validation-groups', type=int, default=40)
    p.add_argument('--test-groups', type=int, default=200)
    p.add_argument('--mode', choices=('sampled','greedy'), default='sampled')
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--arm-duration-minutes', type=float, default=30.0)
    p.add_argument('--validation-seed', type=int, default=12_000_000)
    p.add_argument('--test-seed', type=int, default=22_000_000)
    p.add_argument('--entropy-warning-threshold', type=float, default=0.6)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    return p


def run(args):
    cfg = ComparisonConfig(
        source_checkpoint=args.source_checkpoint, root=args.root,
        seeds=args.seeds, updates=args.updates,
        hands_per_update=args.hands_per_update, env_batch_size=args.env_batch_size,
        minibatch_size=args.minibatch_size, epochs=args.epochs,
        eval_every=args.eval_every, validation_groups=args.validation_groups,
        test_groups=args.test_groups, eval_mode=args.mode, device=args.device,
        beta=args.beta, arm_duration_minutes=args.arm_duration_minutes,
        validation_seed=args.validation_seed, test_seed=args.test_seed,
        entropy_warning_threshold=args.entropy_warning_threshold,
        resume=args.resume, progress=not args.no_progress,
    )
    def emit(item):
        if item.get('kind') in ('validation', 'arm_end'):
            print(json.dumps(item, ensure_ascii=False), flush=True)
    result = run_study(cfg, emit=emit)
    short = {k: v for k, v in result.items() if k not in ('seed_results',)}
    print(json.dumps(short, indent=2, ensure_ascii=False), flush=True)
    return result


def main():
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase14_study.py" -->

## scripts/phase15_critic_audit.py

Original Git blob SHA: `58c19a5675658d879e3c0adae0b1b543befaa334` (3142 bytes).

<a id="py-036"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase15_critic_audit.py" sha="58c19a5675658d879e3c0adae0b1b543befaa334" -->

````````python
"""Run a read-only diagnostic for two trusted Phase-15 HUNL PPO checkpoints.

Example (PowerShell):
  uv run --extra gpu python -m scripts.phase15_critic_audit \
    --reference runs/phase14_main/seed_52/ppo/best.pt \
    --before runs/phase15_ppo_safety_v2_pilot/best.pt \
    --after runs/phase15_ppo_safety_v2_pilot/latest.pt \
    --device cuda --hands 256 --on-policy-hands 128 \
    --output runs/phase15_audit/safety_v2_critic_audit.json

No training, gradient steps, checkpoint writes, or final evaluation sealing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from holdem_ai.evaluation.critic_diagnostics import run_critic_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Phase 15 PPO critic diagnosis")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hands", type=int, default=256,
                        help="hands on shared reference policy")
    parser.add_argument("--on-policy-hands", type=int, default=128,
                        help="hands per checkpoint, using its own policy")
    parser.add_argument("--seed", type=int, default=570000000)
    parser.add_argument("--env-batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output", required=True, help="new report filename; refuses overwrite")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"report exists: {target}")
    report = run_critic_audit(
        reference_checkpoint=args.reference,
        before_checkpoint=args.before,
        after_checkpoint=args.after,
        device=args.device, hands=args.hands,
        on_policy_hands=args.on_policy_hands,
        seed=args.seed, env_batch_size=args.env_batch_size,
        chunk_size=args.chunk_size,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    summary = {
        "report": str(target),
        "updates": report["completed_updates"],
        "shared_states": report["shared_reference"]["decisions"],
        "on_policy": {
            label: {
                "states": data["decisions"],
                "q_rmse_bb": data["q_taken"]["overall"]["rmse_bb"],
                "q_ev": data["q_taken"]["overall"]["explained_variance"],
                "v_rmse_bb": data["v_reference_policy"]["overall"]["rmse_bb"],
                "v_ev": data["v_reference_policy"]["overall"]["explained_variance"],
            }
            for label, data in report["on_policy"]["results"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase15_critic_audit.py" -->
