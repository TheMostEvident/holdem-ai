"""GUI-1.4 local trained-agent selection, loading, privacy, and HTTP regression."""
from __future__ import annotations

import json
import sys
import types

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from holdem_ai.game_service.checkpoint_catalog import CheckpointCatalog

API = '/api/v1/hunl'


def _client(environment=None):
    return TestClient(create_app(catalog=CheckpointCatalog(environment=environment or {})))


def test_catalog_no_models_does_not_load_torch():
    catalog = CheckpointCatalog(environment={})
    opts = catalog.options()
    assert [o['id'] for o in opts] == [
        'random', 'calling_station', 'tight', 'pot_pressure', 'ppo', 'qboost']
    assert all(x['available'] for x in opts[:4])
    assert not any(x['available'] for x in opts[4:])
    with _client() as client:
        res = client.get(f'{API}/bots')
        assert res.status_code == 200
        assert res.headers['cache-control'] == 'no-store'
        assert res.json()['device'] == 'cpu'
        assert res.json()['policy_mode'] == 'sampled'
        assert res.json()['bots'] == opts
        assert client.post(f'{API}/sessions', json={'bot': 'random'}).status_code == 201


def test_unconfigured_checkpoint_never_falls_back_to_random():
    with _client() as client:
        for bot in ('ppo', 'qboost'):
            result = client.post(f'{API}/sessions', json={'bot': bot})
            assert result.status_code == 422
            assert 'checkpoint' in result.json()['detail'].lower()
            assert not client.app.state.gui11_registry._sessions


def test_invalid_neural_game_configuration_rejected():
    with _client() as client:
        for bad in ({'big_blind': 20, 'small_blind': 10},
                    {'stack_bb': 100}, {'big_blind': 100, 'small_blind': 25}):
            result = client.post(f'{API}/sessions', json={'bot': 'ppo', **bad})
            assert result.status_code == 422
            assert 'HUNL200' in result.json()['detail']


def test_catalog_config_validation():
    with pytest.raises(ValueError, match='DEVICE'):
        CheckpointCatalog(environment={'HOLDEM_AI_GUI_DEVICE': 'x'})
    with pytest.raises(ValueError, match='POLICY_MODE'):
        CheckpointCatalog(environment={'HOLDEM_AI_GUI_POLICY_MODE': 'test'})
    assert CheckpointCatalog(environment={'HOLDEM_AI_GUI_DEVICE': 'cuda',
                                          'HOLDEM_AI_GUI_POLICY_MODE': 'greedy'}).mode == 'greedy'


def _install_loader_stubs(monkeypatch, calls, seen, failure=False):
    """Stub Phase-16 loader in minimal Phase 1-4 test fixture.

    In the FULL Phase 0-16 repo the real weights-only Phase-16 loader is used.
    """
    fake_pkg = types.ModuleType('holdem_ai.evaluation')
    fake_pkg.__path__ = []
    fake_loader = types.ModuleType('holdem_ai.evaluation.checkpoint_loader')

    class Actor:
        config = types.SimpleNamespace(dropout=0.0)

    def load_actor(path, *, device):
        calls.append((str(path), device))
        if failure:
            raise ValueError('corrupt file / VERY SECRET path')
        return Actor()

    class Policy:
        def __init__(self, actor, *, device, big_blind, greedy):
            assert isinstance(actor, Actor)
            assert big_blind == 100
            seen.append(('constructed', device, greedy))

        def act(self, observation, rng):
            seen.append(observation)
            assert len(observation.private_cards) == 2
            # Bot must see ONLY its own information state, never both hands.
            return next((a for a in (1, 2, 0, 6, 3, 4, 5) if a in observation.legal_actions))

    fake_loader.load_actor = load_actor
    fake_policy = types.ModuleType('holdem_ai.agents.neural_policy')
    fake_policy.NeuralActorPolicy = Policy
    monkeypatch.setitem(sys.modules, 'holdem_ai.evaluation', fake_pkg)
    monkeypatch.setitem(sys.modules, 'holdem_ai.evaluation.checkpoint_loader', fake_loader)
    monkeypatch.setitem(sys.modules, 'holdem_ai.agents.neural_policy', fake_policy)


def test_trained_http_play_and_private_cards(tmp_path, monkeypatch):
    path = tmp_path / 'trusted.pt'
    path.write_bytes(b'fake-test-fixture-only')
    calls, seen = [], []
    _install_loader_stubs(monkeypatch, calls, seen)
    env = {'HOLDEM_AI_PPO_CHECKPOINT': str(path),
           'HOLDEM_AI_GUI_POLICY_MODE': 'greedy'}
    with _client(env) as client:
        info = client.get(f'{API}/bots').json()
        assert next(x for x in info['bots'] if x['id'] == 'ppo')['available']
        assert next(x for x in info['bots'] if x['id'] == 'qboost')['available'] is False
        assert str(path) not in json.dumps(info)
        game = client.post(f'{API}/sessions', json={'bot': 'ppo', 'seed': 42})
        assert game.status_code == 201, game.text
        sid = game.json()['session_id']
        state = game.json()['state']
        assert calls == [(str(path), 'cpu')]
        assert seen[0] == ('constructed', 'cpu', True)
        secret = client.app.state.gui11_registry.get(sid)._adapter.engine.hole_cards[1]
        for card in secret:
            assert '"' + card + '"' not in json.dumps(game.json())
        for _ in range(100):
            if state['phase'] == 'complete':
                break
            offer = next((a for a in state['actions'] if a['kind'] in ('check','call')), state['actions'][0])
            result = client.post(f'{API}/sessions/{sid}/actions', json={
                'action_index': offer['index'], 'expected_revision': state['revision']})
            assert result.status_code == 200, result.text
            state = result.json()['state']
        seen_bot = [o for o in seen if getattr(o, 'player_id', None) == 1]
        assert seen_bot
        assert all(set(o.private_cards) == set(secret) for o in seen_bot)
        assert all(not (set(o.private_cards) & set(game.json()['state']['hero_cards'])) for o in seen_bot)
        assert all('hole_cards' not in dict(o.public_state) for o in seen_bot)
        assert state['phase'] == 'complete'
        assert sum(state['hand_result']['payoffs_chips']) == 0
        next_res = client.post(f'{API}/sessions/{sid}/next-hand', json={
            'expected_revision': state['revision']})
        assert next_res.status_code == 200
        other = client.post(f'{API}/sessions', json={'bot': 'ppo', 'seed': 99})
        assert other.status_code == 201
        assert len(calls) == 1  # Frozen Actor cached; independent sessions have different RNGs.


def test_corrupt_trained_checkpoint_reports_safe_error(tmp_path, monkeypatch):
    path = tmp_path / 'secret_local_weights.pt'
    path.write_bytes(b'corrupt')
    calls, seen = [], []
    _install_loader_stubs(monkeypatch, calls, seen, failure=True)
    with _client({'HOLDEM_AI_QBOOST_CHECKPOINT': str(path)}) as client:
        result = client.post(f'{API}/sessions', json={'bot': 'qboost'})
        assert result.status_code == 422
        assert 'Could not load' in result.json()['detail']
        assert 'secret_local_weights' not in result.text
        assert 'VERY SECRET' not in result.text
        assert not client.app.state.gui11_registry._sessions


def test_checkpoint_reloaded_when_file_changes(tmp_path, monkeypatch):
    path = tmp_path / 'trusted.pt'
    path.write_bytes(b'aaa')
    calls, seen = [], []
    _install_loader_stubs(monkeypatch, calls, seen)
    catalog = CheckpointCatalog(environment={'HOLDEM_AI_PPO_CHECKPOINT': str(path)})
    first = catalog.build_policy('ppo')
    assert catalog.build_policy('ppo') is first
    path.write_bytes(b'aaaaaa')  # different size invalidates cache
    second = catalog.build_policy('ppo')
    assert second is not first
    assert len(calls) == 2
