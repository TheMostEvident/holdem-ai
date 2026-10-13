"""ENV 1.4: checkpoint gating, action contract and no silent baseline fallback."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from holdem_ai.tournament_service.session import TournamentGameSession
from holdem_ai.tournament_service.trained_actor import (
    TrainedTournamentActor, _prepare_public_state, status, EXPAND_TO_NATIVE,
)

SHORT = dict(seed=23, starting_stack=110, hands_per_level=2,
             blind_levels=[(5, 10), (10, 20), (20, 40), (40, 80), (80, 160)])


def test_unconfigured_status_and_fail_closed(monkeypatch):
    monkeypatch.delenv('HOLDEM_AI_TOURNAMENT_CHECKPOINT', raising=False)
    assert not status()['configured']
    with pytest.raises(ValueError, match='HOLDEM_AI_TOURNAMENT_CHECKPOINT'):
        TournamentGameSession(**SHORT, lineup='trained_1_improved_4')
    with TestClient(create_app()) as client:
        status_resp = client.get('/api/v1/tournament/agent-status')
        assert status_resp.status_code == 200
        assert status_resp.json()['configured'] is False
        response = client.post('/api/v1/tournament/sessions',json={**SHORT,
                            'lineup':'trained_1_improved_4'})
        assert response.status_code == 422
        assert 'checkpoint' in response.text.lower()
        # Existing explicit heuristic path stays intact.
        baseline = client.post('/api/v1/tournament/sessions',json={**SHORT,
                               'lineup':'improved_4'})
        assert baseline.status_code == 201
        assert baseline.json()['state']['seats'][1]['role'] == 'tournament_agent_placeholder'


def test_missing_checkpoint_does_not_fallback(monkeypatch,tmp_path):
    monkeypatch.setenv('HOLDEM_AI_TOURNAMENT_CHECKPOINT',str(tmp_path/'not-found.pt'))
    assert status()['configured'] and not status()['file_found']
    with pytest.raises(ValueError,match='missing'):
        TournamentGameSession(**SHORT,lineup='trained_1_improved_4')


def test_checkpoint_is_chosen_server_side_only(monkeypatch):
    monkeypatch.delenv('HOLDEM_AI_TOURNAMENT_CHECKPOINT',raising=False)
    with TestClient(create_app()) as client:
        response=client.post('/api/v1/tournament/sessions',json={**SHORT,
            'lineup':'trained_1_improved_4','checkpoint':'C:/untrusted.pt'})
        assert response.status_code==422   # extra fields rejected


def test_explicit_five_to_seven_action_mapping():
    assert EXPAND_TO_NATIVE == (0, 1, 1, 2, 3, None, 4)
    assert EXPAND_TO_NATIVE[5] is None  # no double-pot action in Phase4 engine


def test_expand_actor_observation_and_mask(monkeypatch):
    pytest.importorskip('holdem_ai.expand.information.schema')
    from holdem_ai.environment.tournament_env import TournamentEnv
    env = TournamentEnv(num_players=6, starting_stack=200, seed=41)
    env.reset()
    # Advance by legal checks/calls until the acting seat is 1.
    for _ in range(30):
        if env.current_player == 1: break
        env.step(1)
    assert env.current_player == 1
    observed = dict(env.observe(1))
    observed['public_action_log'] = ({'seat':3,'street':'preflop','index':1,
        'raise_to':None,'to_call_before':10,'pot_before':15},)
    state = _prepare_public_state(observed)
    assert state.hero_seat == 1
    assert len(state.seats)==6 and state.hero_hole == tuple(observed['raw_obs']['hand'])
    assert state.action_history[0].kind == 'call'
    assert state.legal_actions.mask[5] is False
    assert len(state.legal_actions.mask)==7
    assert all(isinstance(x,bool) for x in state.legal_actions.mask)
    assert state.big_blind_chips > 0


def test_actor_greedy_never_emits_illegal_native_action(monkeypatch):
    pytest.importorskip('holdem_ai.expand.information.encoder')
    from holdem_ai.environment.tournament_env import TournamentEnv
    env=TournamentEnv(num_players=6,starting_stack=200,seed=43)
    env.reset()
    for _ in range(30):
        if env.current_player==1:break
        env.step(1)
    assert env.current_player==1
    obs=env.observe(1)
    class SpyActor:
        def action_probabilities(self, encoded):
            assert len(encoded.legal_mask)==7
            return (0.001,0.001,0.001,0.3,0.3,0.997,0.001)
    model=SimpleNamespace(actor=SpyActor())
    policy=TrainedTournamentActor(model,fingerprint='a'*64,seed=4,mode='greedy')
    action=policy.select_action(obs)
    assert action in obs['legal_actions']
    assert action != 5


def test_trained_lineup_uses_configured_model_seat_without_touching_others(monkeypatch):
    # Test wiring with explicit stub to isolate loader from unavailable checkpoint.
    from holdem_ai.tournament_service.improved_bot import ImprovedTournamentBot
    class FakeTrainedPolicy:
        name = 'test-trained-interface'
        def select_action(self, observation):
            return 1 if 1 in observation['legal_actions'] else next(iter(observation['legal_actions']))
    monkeypatch.setattr(TrainedTournamentActor,'from_environment',
                        classmethod(lambda cls,seed=None: FakeTrainedPolicy()))
    game=TournamentGameSession(**SHORT,lineup='trained_1_improved_4')
    view=game.view()
    assert view['seats'][1]['role']=='tournament_agent'
    assert view['seats'][1]['policy_name']=='test-trained-interface'
    assert [s['role'] for s in view['seats'][2:]]==['improved_bot']*4
    assert all(isinstance(game._policies[i],ImprovedTournamentBot) for i in range(2,6))
    assert sum(s['stack'] for s in view['seats'])+view['pot']==view['total_chips']


def test_full_live_tournament_with_a_frozen_actor_stub(monkeypatch):
    pytest.importorskip('holdem_ai.expand.information.encoder')
    class LegalBiasedActor:
        def action_probabilities(self, encoded):
            # High likelihoods on all available bets. The adapter must apply
            # its converted tournament action mask before sampling.
            assert len(encoded.legal_mask)==7
            return (0.02, 0.07, 0.16, 0.30, 0.40, 0.99, 0.05)
    model = SimpleNamespace(actor=LegalBiasedActor())
    monkeypatch.setattr(TrainedTournamentActor,'from_environment',
        classmethod(lambda cls,seed=None: cls(model,fingerprint='d'*64,seed=seed)))
    game=TournamentGameSession(seed=42,starting_stack=250,hands_per_level=2,
                               lineup='trained_1_improved_4')
    for _ in range(250):
        view=game.view()
        assert sum(s['stack'] for s in view['seats']) + view['pot']==view['total_chips']
        if view['tournament_over']:
            break
        assert view['your_turn']
        choices={a['index'] for a in view['actions']}
        game.play_action(1 if 1 in choices else next(iter(choices)),expected_revision=view['revision'])
    else:
        pytest.fail('trained-agent test tournament failed to terminate')
    assert game.view()['tournament_over']
    assert len(game.env.hand_history)>0


def test_wrong_format_checkpoint_is_rejected(monkeypatch,tmp_path):
    pytest.importorskip('holdem_ai.expand.rl.trainer')
    torch=pytest.importorskip('torch')
    filename=tmp_path/'hunl200-not-expand.pt'
    torch.save({'format':'not-an-expand-model','model':{}},filename)
    monkeypatch.setenv('HOLDEM_AI_TOURNAMENT_CHECKPOINT',str(filename))
    with pytest.raises(ValueError,match='checkpoint rejected'):
        TrainedTournamentActor.from_environment(seed=9)
