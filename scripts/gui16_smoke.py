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
