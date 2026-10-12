"""GUI-1.5: completed-hand ledger, summary, privacy, pagination, API isolation."""
from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from holdem_ai.game_service.session_analytics import SessionAnalytics

API = '/api/v1/hunl'


def _finish(client, sid, state, *, fold=False):
    for _ in range(200):
        if state['phase'] == 'complete':
            return state
        action = next((a for a in state['actions'] if a['kind'] == ('fold' if fold else 'check')), None)
        if action is None:
            action = next((a for a in state['actions'] if a['kind'] == ('call' if not fold else 'fold')), None)
        if action is None:
            action = state['actions'][0]
        resp = client.post(f'{API}/sessions/{sid}/actions', json={
            'action_index': action['index'], 'expected_revision': state['revision']})
        assert resp.status_code == 200, resp.text
        state = resp.json()['state']
    pytest.fail('did not finish hand within 200 human decisions')


def _new(client, seed=42):
    r = client.post(f'{API}/sessions', json={'bot': 'calling_station', 'seed': seed})
    assert r.status_code == 201, r.text
    return r.json()['session_id'], r.json()['state']


def test_empty_stats_and_history_are_clean_and_cached_nowhere():
    with TestClient(create_app()) as c:
        sid, state = _new(c)
        for endpoint in ('stats', 'history'):
            r = c.get(f'{API}/sessions/{sid}/{endpoint}')
            assert r.status_code == 200
            assert r.headers['cache-control'] == 'no-store'
        summary = c.get(f'{API}/sessions/{sid}/stats').json()
        assert summary['completed_hands'] == 0
        assert summary['ci95_lower_bb100'] is None
        assert summary['net_bb'] == 0
        history = c.get(f'{API}/sessions/{sid}/history').json()
        assert history['items'] == [] and history['has_more'] is False


def test_folded_hands_never_reveal_opponent_cards_in_completed_journal():
    with TestClient(create_app()) as c:
        sid, state = _new(c, seed=42)
        secret = c.app.state.gui11_registry.get(sid)._adapter.engine.hole_cards[1]
        state = _finish(c, sid, state, fold=True)
        assert state['hand_result']['showdown'] is False
        history = c.get(f'{API}/sessions/{sid}/history').json()
        assert history['total'] == 1
        record = history['items'][0]
        assert record['hand_number'] == 1
        assert record['result']['opponent_cards'] is None
        assert record['result']['showdown'] is False
        for card in secret:
            assert '"' + card + '"' not in json.dumps(history)
        assert record['hero_cards'] == state['hero_cards']
        assert record['action_log'] == state['action_log']
        assert record['hero_net_chips'] == state['hand_result']['payoffs_chips'][0]
        assert math.isclose(record['hero_net_bb'], state['hand_result']['payoffs_bb'][0])


def test_two_hand_ledger_and_stats_match_session_and_payoffs():
    with TestClient(create_app()) as c:
        sid, state = _new(c, seed=44)
        results = []
        for index in range(3):
            state = _finish(c, sid, state)
            results.append(state['hand_result']['payoffs_bb'][0])
            assert c.get(f'{API}/sessions/{sid}/history').json()['completed_hands'] == len(results)
            if index < 2:
                r = c.post(f'{API}/sessions/{sid}/next-hand', json={'expected_revision': state['revision']})
                assert r.status_code == 200
                state = r.json()['state']
        stats = c.get(f'{API}/sessions/{sid}/stats').json()
        ledger = c.get(f'{API}/sessions/{sid}/history').json()['items']
        assert [r['hand_number'] for r in ledger] == [3, 2, 1]
        assert stats['completed_hands'] == 3
        assert stats['wins'] + stats['losses'] + stats['break_even'] == 3
        assert stats['retained_hand_records'] == 3
        assert math.isclose(stats['net_bb'], sum(results), abs_tol=1e-8)
        assert math.isclose(stats['bb_per_100'], 100 * sum(results)/3, abs_tol=1e-8)
        assert math.isclose(stats['net_bb'], state['cumulative_bb'], abs_tol=1e-8)
        assert stats['ci95_lower_bb100'] is not None
        assert stats['ci95_lower_bb100'] <= stats['ci95_upper_bb100']
        assert math.isclose(ledger[0]['cumulative_bb'], stats['net_bb'], abs_tol=1e-8)
        assert ledger[-1]['button'] == 0
        assert ledger[1]['button'] == 1
        assert not any('critic_state' in r or 'future_deck' in r for r in ledger)


def test_history_offset_limit_and_input_validation():
    with TestClient(create_app()) as c:
        sid, state = _new(c, seed=9)
        for _ in range(4):
            state = _finish(c, sid, state, fold=True)
            state = c.post(f'{API}/sessions/{sid}/next-hand',
                           json={'expected_revision': state['revision']}).json()['state']
        page1 = c.get(f'{API}/sessions/{sid}/history?offset=0&limit=2').json()
        page2 = c.get(f'{API}/sessions/{sid}/history?offset=2&limit=2').json()
        assert [h['hand_number'] for h in page1['items']] == [4, 3]
        assert [h['hand_number'] for h in page2['items']] == [2, 1]
        assert page1['has_more'] and not page2['has_more']
        assert c.get(f'{API}/sessions/{sid}/history?offset=-1').status_code == 422
        assert c.get(f'{API}/sessions/{sid}/history?limit=0').status_code == 422
        assert c.get(f'{API}/sessions/{sid}/history?limit=101').status_code == 422
        assert c.get(f'{API}/sessions/{sid}/history?offset=999').json()['items'] == []


def test_aggregate_survives_record_retention_and_history_is_defensive_copy():
    store = SessionAnalytics(history_limit=2)
    for n, chips in enumerate([100, -50, 75], 1):
        store.append_completed({
            'phase': 'complete', 'hand_number': n, 'hero_seat': 0,
            'button': (n-1) % 2, 'street': 'river',
            'big_blind': 100, 'small_blind': 50, 'hero_cards': ['sA', 'hK'],
            'board': [], 'action_log': [],
            'hand_result': {'payoffs_chips': [chips, -chips],
                            'payoffs_bb': [chips/100, -chips/100],
                            'final_stacks': [20000+chips, 20000-chips],
                            'awards': [0, 0], 'showdown': False,
                            'opponent_cards': None},
        })
    assert store.summary()['completed_hands'] == 3
    assert store.summary()['retained_hand_records'] == 2
    assert store.summary()['records_truncated']
    assert store.summary()['net_chips'] == 125
    page = store.history()
    assert [h['hand_number'] for h in page['items']] == [3, 2]
    page['items'][0]['hero_cards'][0] = 'INVALID'
    assert store.history()['items'][0]['hero_cards'][0] == 'sA'
    with pytest.raises(ValueError):
        store.history(limit=101)
    with pytest.raises(ValueError):
        SessionAnalytics(history_limit=0)


def test_session_isolation_and_deletion():
    with TestClient(create_app()) as c:
        one, first = _new(c, seed=42)
        two, other = _new(c, seed=43)
        _finish(c, one, first, fold=True)
        assert c.get(f'{API}/sessions/{one}/stats').json()['completed_hands'] == 1
        assert c.get(f'{API}/sessions/{two}/stats').json()['completed_hands'] == 0
        assert c.delete(f'{API}/sessions/{one}').status_code == 204
        assert c.get(f'{API}/sessions/{one}/stats').status_code == 404
        assert c.get(f'{API}/sessions/{one}/history').status_code == 404
        assert c.get(f'{API}/sessions/{two}/history').json()['total'] == 0


def test_statistics_do_not_update_mid_hand_or_on_invalid_action():
    with TestClient(create_app()) as c:
        sid, state = _new(c)
        before = c.get(f'{API}/sessions/{sid}/stats').json()
        bad = c.post(f'{API}/sessions/{sid}/actions', json={
            'action_index': 6 if 6 not in [x['index'] for x in state['actions']] else 99,
            'expected_revision': state['revision']})
        assert bad.status_code == 422
        assert c.get(f'{API}/sessions/{sid}/stats').json() == before
        assert c.get(f'{API}/sessions/{sid}/history').json()['items'] == []
