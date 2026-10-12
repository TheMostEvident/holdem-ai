"""GUI-1.3: actual engine transitions through browser-facing HTTP contracts."""
import pytest
from fastapi.testclient import TestClient
from backend.main import create_app

PREFIX = '/api/v1/hunl/sessions'

@pytest.fixture
def client():
    with TestClient(create_app(max_sessions=16)) as api:
        yield api


def create(client, seed=42, bot='calling_station'):
    response = client.post(PREFIX, json={'seed': seed, 'bot': bot})
    assert response.status_code == 201, response.text
    return response.json()


def test_human_exact_raise_and_bot_response(client):
    game = create(client)
    sid, state = game['session_id'], game['state']
    assert state['your_turn'] and state['can_raise_to']
    target = state['min_raise_to'] + 1
    assert target < state['max_raise_to']
    r = client.post(f'{PREFIX}/{sid}/raise-to', json={
        'raise_to': target, 'expected_revision': state['revision'],
    })
    assert r.status_code == 200, r.text
    nxt = r.json()['state']
    assert nxt['revision'] > state['revision']
    assert nxt['action_log'][0]['index'] == -1
    assert nxt['action_log'][0]['kind'] == 'raise_to'
    assert nxt['action_log'][0]['raise_to'] == target
    assert any(event['seat'] == 1 for event in nxt['action_log'])
    assert not any(k in str(nxt).lower() for k in ('_deck', 'criticstate', 'burned'))
    assert nxt['hand_result'] is None or nxt['hand_result']['opponent_cards'] is None or nxt['hand_result']['showdown']


def test_invalid_raise_is_atomic_and_revision_safe(client):
    game = create(client)
    sid, state = game['session_id'], game['state']
    endpoint = f'{PREFIX}/{sid}/raise-to'
    for target in (0, state['min_raise_to'] - 1, state['max_raise_to'] + 1):
        response = client.post(endpoint, json={'raise_to': target, 'expected_revision': state['revision']})
        assert response.status_code == 422, response.text
        unchanged = client.get(f'{PREFIX}/{sid}').json()['state']
        assert unchanged == state
    first = client.post(endpoint, json={'raise_to': state['min_raise_to'], 'expected_revision': state['revision']})
    assert first.status_code == 200, first.text
    stale = client.post(endpoint, json={'raise_to': state['min_raise_to'], 'expected_revision': state['revision']})
    assert stale.status_code == 409
    assert 'stale' in stale.json()['detail']


def test_browser_regular_action_and_next_hand(client):
    game = create(client, seed=7)
    sid, state = game['session_id'], game['state']
    assert state['your_turn']
    fold = next(a for a in state['actions'] if a['kind'] == 'fold')
    action = client.post(f'{PREFIX}/{sid}/actions', json={
        'action_index': fold['index'], 'expected_revision': state['revision']})
    assert action.status_code == 200, action.text
    finished = action.json()['state']
    assert finished['phase'] == 'complete'
    assert finished['hand_result']['opponent_cards'] is None
    following = client.post(f'{PREFIX}/{sid}/next-hand', json={'expected_revision': finished['revision']})
    assert following.status_code == 200, following.text
    follow_state = following.json()['state']
    assert follow_state['hand_number'] == 2
    assert follow_state['button'] == 1
    assert follow_state['completed_hands'] == 1
    assert follow_state['phase'] == 'playing'


def test_reject_bool_fraction_unknown_json_and_nonhuman_turn(client):
    game = create(client)
    sid, state = game['session_id'], game['state']
    url = f'{PREFIX}/{sid}/raise-to'
    for target in (True, 250.5, '250', None):
        response = client.post(url, json={'raise_to': target, 'expected_revision': state['revision']})
        assert response.status_code == 422, response.text
    extra = client.post(url, json={'raise_to': 300, 'expected_revision': state['revision'], 'opponent_cards': ['sA', 'hA']})
    assert extra.status_code == 422
    assert client.get(f'{PREFIX}/{sid}').json()['state'] == state


def test_fold_check_call_loop_and_chip_conservation(client):
    game = create(client, seed=19)
    sid, state = game['session_id'], game['state']
    for _ in range(80):
        if state['phase'] == 'complete':
            break
        offer = next((o for o in state['actions'] if o['kind'] in ('check', 'call')), None)
        if offer is None:
            offer = state['actions'][0]
        response = client.post(f'{PREFIX}/{sid}/actions', json={
            'action_index': offer['index'], 'expected_revision': state['revision']})
        assert response.status_code == 200, response.text
        state = response.json()['state']
    else:
        pytest.fail('hand did not terminate')
    assert state['hand_result'] is not None
    assert sum(state['hand_result']['final_stacks']) == 40000
    assert sum(state['hand_result']['payoffs_chips']) == 0
    if state['hand_result']['showdown']:
        assert len(state['hand_result']['opponent_cards']) == 2
