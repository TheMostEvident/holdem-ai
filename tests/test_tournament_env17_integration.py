"""Tournament ENV 1.7: release contract across both GUIs and durable replay.

No training/checkpoint is loaded here; offline and deterministic baseline bots.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend.main import create_app


def test_both_api_namespaces_are_served_without_training(tmp_path):
    db = tmp_path / 'combined.sqlite3'
    app = create_app(tournament_journal_path=str(db))
    with TestClient(app) as client:
        assert client.get('/api/v1/hunl/health').status_code == 200
        assert client.get('/api/v1/tournament/health').status_code == 200
        bots = client.get('/api/v1/hunl/bots')
        assert bots.status_code == 200
        assert any(x['id'] == 'random' for x in bots.json()['bots'])
        hunl = client.post('/api/v1/hunl/sessions', json={
            'bot': 'random', 'seed': 20261013,
        })
        assert hunl.status_code == 201, hunl.text
        assert hunl.json()['state']['game_type'] == 'HUNL200'
        tournament = client.post('/api/v1/tournament/sessions', json={
            'lineup': 'improved_4', 'seed': 20261013, 'starting_stack': 180,
            'hands_per_level': 2,
        })
        assert tournament.status_code == 201, tournament.text
        state = tournament.json()['state']
        assert state['game_type'] == 'TOURNAMENT_6MAX'
        assert len(state['seats']) == 6
        assert state['seats'][0]['role'] == 'human'
        assert state['seats'][1]['role'] == 'tournament_agent_placeholder'
        assert all(s['role'] == 'improved_bot' for s in state['seats'][2:])
        assert sum(s['stack'] for s in state['seats']) + state['pot'] == state['total_chips']
        assert 'hole_cards' not in state
        assert 'deck' not in state
        assert 'showdown_cards' not in state  # only settlement may reveal cards
        assert client.get('/api/v1/hunl/health').status_code == 200
        assert client.delete(f"/api/v1/hunl/sessions/{hunl.json()['session_id']}").status_code == 204


def test_journal_replay_settlement_and_export_survive_app_restart(tmp_path):
    db = tmp_path / 'replay.sqlite3'
    with TestClient(create_app(tournament_journal_path=str(db))) as client:
        created = client.post('/api/v1/tournament/sessions', json={
            'lineup': 'baseline_5', 'seed': 23, 'starting_stack': 180,
            'hands_per_level': 2,
        })
        assert created.status_code == 201, created.text
        sid = created.json()['session_id']
        state = created.json()['state']
        for _ in range(60):
            if state['hands_completed'] > 0:
                break
            assert state['your_turn']
            result = client.post(f'/api/v1/tournament/sessions/{sid}/actions', json={
                'action_index': 1, 'expected_revision': state['revision'],
            })
            assert result.status_code == 200, result.text
            state = result.json()['state']
        assert state['hands_completed'] >= 1
        assert sum(state['tournament_stacks']) == state['total_chips']
        current = client.get(f'/api/v1/tournament/sessions/{sid}/history')
        assert current.status_code == 200
        assert current.json()['hands'][0]['showdown_cards']
        replay = client.get(f'/api/v1/tournament/archive/{sid}/hands/1/replay')
        assert replay.status_code == 200
        snapshot = replay.json()
        assert snapshot['complete'] is True
        assert snapshot['settlement']['showdown_cards'] == current.json()['hands'][0]['showdown_cards']
        assert all('showdown_cards' not in a and 'actor_hole' not in a for a in snapshot['actions'])
        export = client.get(f'/api/v1/tournament/archive/{sid}/export')
        assert export.status_code == 200
        rows = [json.loads(line) for line in export.text.splitlines() if line.strip()]
        assert rows
        assert all(row['schema'] == 'tournament.actor_transition.v1' for row in rows)
        assert all(len(row['observation']['own_hole_cards']) == 2 for row in rows)
        assert all('showdown_cards' not in json.dumps(row) for row in rows)
        assert client.delete(f'/api/v1/tournament/sessions/{sid}').status_code == 204
    # A fresh application uses only the persisted journal; no live session resume.
    with TestClient(create_app(tournament_journal_path=str(db))) as client:
        assert client.get(f'/api/v1/tournament/sessions/{sid}').status_code == 404
        old = client.get(f'/api/v1/tournament/archive/{sid}/hands/1/replay')
        assert old.status_code == 200
        assert old.json()['settlement']['showdown_cards'] == snapshot['settlement']['showdown_cards']
        assert client.get('/api/v1/hunl/health').status_code == 200
