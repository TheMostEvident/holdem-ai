"""GUI-1.6 end-to-end stack carryover, cash accounting, bust and mode isolation.

Runs on the full repository after GUI-1.0~1.6 is installed.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient
import pytest

from backend.main import create_app
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.game_service.session_manager import HUNLGameSession
from holdem_ai.agents.diagnostic_opponents import CallingStation

API = '/api/v1/hunl'


def _create(client, **options):
    result = client.post(f'{API}/sessions', json={'bot': 'calling_station', 'seed': 42, **options})
    assert result.status_code == 201, result.text
    return result.json()['session_id'], result.json()['state']


def _action(client, sid, state, action_index):
    response = client.post(f'{API}/sessions/{sid}/actions', json={
        'action_index': action_index, 'expected_revision': state['revision']})
    assert response.status_code == 200, response.text
    return response.json()['state']


def _complete_by_folding(client, sid, state):
    for _ in range(80):
        if state['phase'] == 'complete':
            return state
        assert state['your_turn']
        fold = next(a for a in state['actions'] if a['kind'] == 'fold')
        state = _action(client, sid, state, fold['index'])
    pytest.fail('more than 80 fold decisions in one hand')


def _next(client, sid, state):
    response = client.post(f'{API}/sessions/{sid}/next-hand', json={
        'expected_revision': state['revision']})
    assert response.status_code == 200, response.text
    return response.json()['state']


def test_browser_default_is_continuous_and_chips_carried_exactly():
    with TestClient(create_app()) as client:
        sid, state = _create(client)
        assert state['stack_mode'] == 'continuous'
        assert state['starting_stacks'] == [20000, 20000]
        for h in range(1, 11):
            opening = tuple(state['starting_stacks'])
            assert tuple(s['stack'] + s['total_contribution'] for s in state['seats']) == opening
            assert state['button'] == (h - 1) % 2
            assert state['hand_number'] == h
            final = _complete_by_folding(client, sid, state)
            result = final['hand_result']
            assert sum(result['final_stacks']) == 40000
            assert tuple(x['stack'] for x in final['seats']) == tuple(result['final_stacks'])
            assert result['payoffs_chips'] == [result['final_stacks'][i] - opening[i] for i in (0, 1)]
            assert not final['match_over']
            # Never expose opponent hole cards when a player folds.
            assert result['opponent_cards'] is None
            assert 'hole_cards' not in json.dumps(final)
            journal = client.get(f'{API}/sessions/{sid}/history').json()
            assert journal['items'][0]['starting_stacks'] == list(opening)
            if h < 10:
                state = _next(client, sid, final)
                assert tuple(state['starting_stacks']) == tuple(result['final_stacks'])
                assert state['revision'] > final['revision']


def test_browser_explicit_independent_mode_keeps_original_behavior():
    with TestClient(create_app()) as client:
        sid, state = _create(client, stack_mode='independent')
        assert state['stack_mode'] == 'independent'
        ended = _complete_by_folding(client, sid, state)
        new_state = _next(client, sid, ended)
        assert new_state['starting_stacks'] == [20000, 20000]
        assert ended['hand_result']['final_stacks'] != [20000, 20000]


def test_continuous_all_in_bust_finishes_match_and_blocks_next_hand():
    with TestClient(create_app()) as client:
        sid, state = _create(client, seed=123)
        allin = next(a for a in state['actions'] if a['kind'] == 'all_in')
        ended = _action(client, sid, state, allin['index'])
        # CallingStation calls the all-in: all five board cards are run out.
        assert ended['phase'] == 'complete'
        assert ended['hand_result']['showdown']
        if ended['hand_result']['final_stacks'] == [20000, 20000]:
            pytest.skip('Extremely rare exact split with same seed; retry another fixed seed')
        assert ended['match_over']
        assert sum(ended['hand_result']['final_stacks']) == 40000
        assert min(ended['hand_result']['final_stacks']) == 0
        assert ended['winner_seat'] == ended['hand_result']['final_stacks'].index(40000)
        rejected = client.post(f'{API}/sessions/{sid}/next-hand', json={
            'expected_revision': ended['revision']})
        assert rejected.status_code == 409
        reread = client.get(f'{API}/sessions/{sid}').json()['state']
        assert reread == ended  # no revision/hand counter mutation on rejection
        stale = client.post(f'{API}/sessions/{sid}/next-hand', json={
            'expected_revision': ended['revision'] - 1})
        assert stale.status_code == 409


def test_training_reset_unchanged_and_direct_session_default_independent():
    adapter = HUNL200Adapter(big_blind=100, small_blind=50, stack_bb=200)
    adapter.reset(seed=90)
    assert tuple(adapter.engine.initial_stacks) == (20000, 20000)
    adapter.engine.stacks[0] -= 10  # simulate mutation of game before independent reset
    adapter.reset(seed=91)
    assert tuple(adapter.engine.initial_stacks) == (20000, 20000)
    session = HUNLGameSession(bot_policy=CallingStation(), seed=10)
    assert session.stack_mode == 'independent'
    assert session.start_hand()['stack_mode'] == 'independent'


def test_bad_stack_modes_rejected_safely():
    with pytest.raises(ValueError, match='stack_mode'):
        HUNLGameSession(stack_mode='unexpected')
    with TestClient(create_app()) as client:
        assert client.post(f'{API}/sessions', json={'stack_mode': 'x'}).status_code == 422


def test_automatic_short_blind_terminal_counts_and_finishes_match():
    """Regression: a hand may finish during reset without a human or bot action."""
    with TestClient(create_app()) as client:
        sid, state = _create(client, seed=42)
        auto_settlements = 0
        for _ in range(290):
            if state['phase'] == 'playing':
                state = _complete_by_folding(client, sid, state)
            if state['action_log'] == []:
                auto_settlements += 1
            assert state['completed_hands'] == state['hand_number']
            assert sum(state['hand_result']['final_stacks']) == 40000
            assert tuple(x['stack'] for x in state['seats']) == tuple(state['hand_result']['final_stacks'])
            if state['match_over']:
                break
            state = _next(client, sid, state)
        assert state['match_over']
        assert state['completed_hands'] == 267
        assert auto_settlements >= 1
        assert state['hand_result']['final_stacks'] == [0, 40000]
        assert client.get(f'{API}/sessions/{sid}/stats').json()['completed_hands'] == 267
