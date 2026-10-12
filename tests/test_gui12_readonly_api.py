"""GUI-1.2 read-only display contract against the real GUI-1.1 API."""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import create_app


def test_frontend_bootstrap_contract_and_cache_headers():
    with TestClient(create_app()) as client:
        response = client.post('/api/v1/hunl/sessions', json={
            'bot': 'random', 'seed': 42, 'big_blind': 100, 'small_blind': 50, 'stack_bb': 200,
        })
        assert response.status_code == 201
        body = response.json()
        state = body['state']
        assert response.headers['cache-control'] == 'no-store'
        assert isinstance(body['session_id'], str)
        assert state['game_type'] == 'HUNL200'
        assert len(state['hero_cards']) == 2
        assert state['phase'] == 'playing'
        assert len(state['seats']) == 2
        assert {s['role'] for s in state['seats']} == {'human', 'bot'}
        assert all(set(a) == {'index', 'label', 'kind', 'raise_to'} for a in state['actions'])
        assert state['hand_result'] is None
        assert 'hole_cards' not in str(state)
        assert 'opponent_cards' not in str(state)
        assert client.get(f"/api/v1/hunl/sessions/{body['session_id']}").json() == body
        assert client.delete(f"/api/v1/hunl/sessions/{body['session_id']}").status_code == 204


def test_browser_cors_is_restricted_to_local_development_origins():
    with TestClient(create_app()) as client:
        legit = client.options('/api/v1/hunl/sessions', headers={
            'Origin': 'http://127.0.0.1:5173',
            'Access-Control-Request-Method': 'POST',
        })
        assert legit.status_code == 200
        assert legit.headers['access-control-allow-origin'] == 'http://127.0.0.1:5173'
        remote = client.options('/api/v1/hunl/sessions', headers={
            'Origin': 'https://example.com',
            'Access-Control-Request-Method': 'POST',
        })
        assert 'access-control-allow-origin' not in remote.headers


def test_contract_hides_bot_cards_on_fold_finish():
    with TestClient(create_app()) as client:
        game = client.post('/api/v1/hunl/sessions', json={'bot': 'calling_station', 'seed': 1}).json()
        state = game['state']
        fold_index = next(a['index'] for a in state['actions'] if a['kind'] == 'fold')
        response = client.post(f"/api/v1/hunl/sessions/{game['session_id']}/actions", json={
            'action_index': fold_index, 'expected_revision': state['revision'],
        })
        assert response.status_code == 200
        state = response.json()['state']
        assert state['phase'] == 'complete'
        assert state['hand_result']['showdown'] is False
        assert state['hand_result']['opponent_cards'] is None
