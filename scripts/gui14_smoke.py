"""Smoke a trusted local Phase 15/16 neural checkpoint through GUI FastAPI.

Usage (from repo root):
  uv run --extra gui --extra gpu python -m scripts.gui14_smoke --bot ppo --hands 3

Set HOLDEM_AI_PPO_CHECKPOINT / HOLDEM_AI_QBOOST_CHECKPOINT before starting.
This utility never downloads or auto-discovers a checkpoint.
"""
from __future__ import annotations

import argparse
from fastapi.testclient import TestClient
from backend.main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description='GUI-1.4 trained policy API smoke')
    parser.add_argument('--bot', choices=('ppo','qboost'), required=True)
    parser.add_argument('--hands', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.hands < 1:
        parser.error('--hands must be positive')
    with TestClient(create_app()) as client:
        catalog = client.get('/api/v1/hunl/bots').json()
        selected = next(b for b in catalog['bots'] if b['id'] == args.bot)
        if not selected['available']:
            parser.error(f'{args.bot}: {selected["message"]}')
        response = client.post('/api/v1/hunl/sessions', json={
            'bot': args.bot, 'seed': args.seed, 'big_blind': 100,
            'small_blind': 50, 'stack_bb': 200})
        response.raise_for_status()
        payload = response.json()
        sid, game = payload['session_id'], payload['state']
        for index in range(args.hands):
            for _ in range(200):
                if game['phase'] == 'complete':
                    break
                assert game['your_turn']
                offer = next((o for o in game['actions'] if o['kind'] in ('call','check')), game['actions'][0])
                answer = client.post(f'/api/v1/hunl/sessions/{sid}/actions', json={
                    'action_index': offer['index'], 'expected_revision': game['revision']})
                answer.raise_for_status()
                game = answer.json()['state']
            else:
                raise RuntimeError('max human decisions exceeded')
            result = game['hand_result']
            assert sum(result['payoffs_chips']) == 0
            assert sum(result['final_stacks']) == 40000
            print(f'hand={index + 1} net_bb={result["payoffs_bb"][0]:.2f} '
                  f'cumulative_bb={game["cumulative_bb"]:.2f}')
            if index + 1 < args.hands:
                answer = client.post(f'/api/v1/hunl/sessions/{sid}/next-hand', json={
                    'expected_revision': game['revision']})
                answer.raise_for_status()
                game = answer.json()['state']
        print(f'GUI-1.4 PASS ({args.bot}, {args.hands} hands, mode={catalog["policy_mode"]})')


if __name__ == '__main__':
    main()
