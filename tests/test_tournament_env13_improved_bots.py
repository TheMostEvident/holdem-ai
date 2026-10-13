"""Tournament ENV 1.3: explicit six-seat lineup with four heuristic policies."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from holdem_ai.tournament_service.improved_bot import (
    PROFILES, ImprovedTournamentBot, _preflop_strength,
)
from holdem_ai.tournament_service.session import TournamentGameSession


SHORT = dict(seed=23, starting_stack=110, hands_per_level=2,
             blind_levels=[(5, 10), (10, 20), (20, 40), (40, 80), (80, 160)])


def test_lineup_is_opt_in_and_default_backward_compatible():
    legacy = TournamentGameSession(**SHORT)
    assert legacy.view()['lineup'] == 'baseline_5'
    assert [seat['role'] for seat in legacy.view()['seats']] == ['human'] + ['baseline_bot'] * 5


def test_new_roster_with_no_fake_trained_model():
    game = TournamentGameSession(**SHORT, lineup='improved_4')
    view = game.view()
    assert view['lineup'] == 'improved_4'
    assert [seat['role'] for seat in view['seats']] == [
        'human', 'tournament_agent_placeholder', *(['improved_bot'] * 4)
    ]
    assert view['seats'][1]['policy_name'] == 'controlled_pressure'
    assert [s['policy_name'] for s in view['seats'][2:]] == list(PROFILES)
    assert 'ppo' not in str(view).lower()
    assert sum(s['stack'] for s in view['seats']) + view['pot'] == view['total_chips']


def test_validate_lineup_and_four_profiles():
    with pytest.raises(ValueError, match='lineup'):
        TournamentGameSession(lineup='untrusted_ppo')
    with pytest.raises(ValueError, match='profile'):
        ImprovedTournamentBot('experimental_magic')
    assert len(PROFILES) == len(set(PROFILES)) == 4


def test_preflop_values_follow_observed_hole_cards():
    assert _preflop_strength(('sA', 'hA')) > _preflop_strength(('sK','hQ'))
    assert _preflop_strength(('sK', 'hQ')) > _preflop_strength(('s7', 'd2'))


def test_all_profiles_only_output_engine_legal_actions():
    for profile in PROFILES:
        policy = ImprovedTournamentBot(profile, seed=11)
        for legal in ((0, 1, 2, 3, 4), (0, 1), (1,), (0,), (4,), (1, 4)):
            raw = dict(hand=['sA','hK'], public_cards=[], player_id=2, to_call=5,
                       pot=40, stacks=[100]*6, status=['active']*6)
            state = {'legal_actions': {i: None for i in legal}, 'raw_obs': raw}
            for _ in range(6):
                assert policy.select_action(state) in legal


def test_missing_hole_cards_are_not_accepted():
    state = dict(legal_actions={0:None,1:None}, raw_obs=dict(
        hand=[], public_cards=[], player_id=2, to_call=5,
        pot=40, stacks=[100]*6, status=['active']*6))
    with pytest.raises(ValueError, match='malformed'):
        ImprovedTournamentBot('balanced_value').select_action(state)


def test_new_lineup_seed_reproducible_and_chips_invariant():
    a = TournamentGameSession(**SHORT, lineup='improved_4')
    b = TournamentGameSession(**SHORT, lineup='improved_4')
    for _ in range(350):
        av, bv = a.view(), b.view()
        assert av == bv
        assert sum(s['stack'] for s in av['seats']) + av['pot'] == av['total_chips']
        if av['tournament_over']:
            break
        assert av['your_turn']
        a.play_action(1, expected_revision=av['revision'])
        b.play_action(1, expected_revision=bv['revision'])
    else:
        pytest.fail('tournament stuck')
    assert a.view()['winner_seat'] is not None
    assert len(a.env.hand_history) == a.view()['hands_completed']


def test_api_lineup_creation_and_projection_redacts_private_cards():
    with TestClient(create_app()) as client:
        invalid = client.post('/api/v1/tournament/sessions', json={'lineup':'ppo_sixmax'})
        assert invalid.status_code == 422
        response = client.post('/api/v1/tournament/sessions',json={**SHORT,'lineup':'improved_4'})
        assert response.status_code == 201, response.text
        payload = response.json(); state=payload['state']
        assert state['lineup']=='improved_4'
        assert state['seats'][1]['role']=='tournament_agent_placeholder'
        assert sum(s['role']=='improved_bot' for s in state['seats']) == 4
        session = client.app.state.tournament_env_p1_registry.get(payload['session_id'])
        for i in range(1,6):
            for card in session.env._hand.hole_cards[i]:
                if card not in state['hero_cards'] and card not in state['board']:
                    assert json.dumps(card) not in json.dumps(state)
        assert 'hole_cards' not in json.dumps(state)
        nxt = client.post(f"/api/v1/tournament/sessions/{payload['session_id']}/actions", json={
            'expected_revision':state['revision'],'action_index':state['actions'][0]['index']})
        assert nxt.status_code == 200,nxt.text
        assert nxt.json()['state']['revision'] > state['revision']


def test_policy_only_sees_the_given_legal_observation():
    # no access to env, deck, opponent private card attributes
    bot=ImprovedTournamentBot('controlled_pressure',seed=1)
    view={'legal_actions':{1:None},'raw_obs':dict(
        hand=['s2','h2'], public_cards=['d2','h9','cK'], player_id=5,
        to_call=0, pot=70, stacks=[100]*6, status=['active']*6)}
    assert bot.select_action(view)==1
    assert not hasattr(bot,'env') and not hasattr(bot,'train')
