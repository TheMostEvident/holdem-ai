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
