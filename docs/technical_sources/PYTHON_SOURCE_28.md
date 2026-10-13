# Full Python implementation — Volume 28 / 28

[최종 통합 기술보고서](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md)의 코드 부록. 원본 커밋 `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`에 포함된 테스트 파일의 전체 원문이다. 구분용 LF는 원문 바이트 수와 Git Blob SHA 계산에서 제외한다.

## `tests/test_tournament_env131_aggression.py`

Original blob `64898cf766c4023f1268c14686e7af8952e993a3` (2720 bytes).

<a name="py-319"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_tournament_env131_aggression.py" sha="64898cf766c4023f1268c14686e7af8952e993a3" -->

````````python
"""ENV 1.3.1 regression tests for multiway bot aggression and action safety."""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import pytest

from holdem_ai.tournament_service.improved_bot import ImprovedTournamentBot, PROFILES
from holdem_ai.tournament_service.session import TournamentGameSession


def observation(hole, *, legal=(0, 1, 2, 3, 4), board=(), call=10, pot=45, stack=20000):
    return {
        'legal_actions': {int(action): None for action in legal},
        'raw_obs': {
            'hand': list(hole), 'public_cards': list(board),
            'player_id': 2, 'to_call': call, 'pot': pot,
            'stacks': [stack] * 6, 'status': ['active'] * 6,
        },
    }


@pytest.mark.parametrize('profile', PROFILES)
def test_pot_raise_remains_possible_when_half_pot_is_illegal(profile):
    # Under old code the guard `and 2 in legal` wrongly downgraded
    # a legal pot raise to a call for even AA.
    bot = ImprovedTournamentBot(profile, seed=42)
    for _ in range(5):
        assert bot.select_action(observation(('sA','hA'), legal=(0,1,3,4))) == 3


@pytest.mark.parametrize('profile,min_raises', [
    ('tight_aggressive', 45),
    ('balanced_value', 150),
    ('pot_odds', 75),
    ('controlled_pressure', 240),
])
def test_policies_reach_value_raise_on_preflop_hand_distribution(profile, min_raises):
    """Synthetic 1,326 two-card combos, not an estimate of real match raise %.

    Fixed pot/stack/legal actions deliberately isolate the policy from engine
    and strategy effects. Real-game action frequencies require match logging.
    """
    deck = [f'{s}{r}' for s in 'shdc' for r in '23456789TJQKA']
    bot = ImprovedTournamentBot(profile, seed=2026)
    counts = Counter(bot.select_action(observation(cards)) for cards in combinations(deck, 2))
    assert sum(counts.values()) == 1326
    assert counts[2] + counts[3] >= min_raises, counts
    assert counts[4] == 0  # deep stack: no blind automatic all-in
    assert set(counts) <= {0,1,2,3,4}


def test_agent_slot_is_honestly_marked_as_untrained_heuristic():
    game = TournamentGameSession(seed=31, starting_stack=400,
                                 hands_per_level=2, lineup='improved_4')
    seat = game.view()['seats'][1]
    assert seat['role'] == 'tournament_agent_placeholder'
    assert seat['policy_name'] == 'controlled_pressure'
    assert 'PPO' not in str(seat) and 'trained' not in str(seat)


def test_no_raise_if_engine_does_not_offer_any_raise_action():
    for profile in PROFILES:
        bot = ImprovedTournamentBot(profile, seed=20)
        for _ in range(10):
            assert bot.select_action(observation(('sA', 'hA'), legal=(0,1))) in (0, 1)

````````

<!-- END_PY_SOURCE path="tests/test_tournament_env131_aggression.py" -->

## `tests/test_tournament_env13_improved_bots.py`

Original blob `17824f1e1f225e4a96b44209b6ca456be26550a3` (5331 bytes).

<a name="py-320"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_tournament_env13_improved_bots.py" sha="17824f1e1f225e4a96b44209b6ca456be26550a3" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/test_tournament_env13_improved_bots.py" -->

## `tests/test_tournament_env14_trained_actor.py`

Original blob `39b95826af5e6e1d3fbb5fadb1b0b8c93bc7df65` (7407 bytes).

<a name="py-321"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_tournament_env14_trained_actor.py" sha="39b95826af5e6e1d3fbb5fadb1b0b8c93bc7df65" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/test_tournament_env14_trained_actor.py" -->

## `tests/test_tournament_env15_progress.py`

Original blob `680a2eca93be0d1b8317d8a8e1984a4b0c121e07` (4064 bytes).

<a name="py-322"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_tournament_env15_progress.py" sha="680a2eca93be0d1b8317d8a8e1984a4b0c121e07" -->

````````python
"""Tournament ENV 1.5 public, read-only standings and blind level contract."""
from __future__ import annotations

import json
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.tournament_router import build_tournament_router
from backend.tournament_schemas import TournamentView
from holdem_ai.tournament_service.session import TournamentGameSession, TournamentSessionRegistry


def test_state_has_authoritative_settled_stacks_and_alive_seats():
    session = TournamentGameSession(seed=43, starting_stack=2000, hands_per_level=3)
    view = session.view()
    assert sum(view['tournament_stacks']) == view['total_chips'] == 12_000
    assert len(view['tournament_stacks']) == 6
    assert view['alive_seats'] == sorted(set(view['alive_seats']))
    assert all(view['tournament_stacks'][i] > 0 for i in view['alive_seats'])
    assert all(view['tournament_stacks'][i] == 0 for i in set(range(6)) - set(view['alive_seats']))
    assert sum(seat['stack'] for seat in view['seats']) + view['pot'] == view['total_chips']
    assert TournamentView.model_validate(view).model_dump()['blind_progress'] == view['blind_progress']
    payload = json.dumps(view)
    for forbidden in ('hole_cards', 'deck', 'critic', 'optimizer', 'torch_rng'):
        assert f'"{forbidden}"' not in payload


def test_correct_blind_countdown_uses_actual_engine_schedule():
    game = TournamentGameSession(seed=13, starting_stack=2000, hands_per_level=2,
                                 blind_levels=((5, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 320), (320, 640), (640, 1280), (1280, 2560)))
    for _ in range(200):
        view = game.view()
        p = view['blind_progress']
        assert p['hands_per_level'] == 2
        if view['tournament_over']:
            assert p['hands_to_next'] is None
            break
        assert 1 <= p['hand_in_level'] <= 2
        assert view['level_index'] == min((view['hand_number'] - 1) // 2, 8)
        assert p['final_level'] == (view['level_index'] == 8)
        if p['final_level']:
            assert p['next_small_blind'] is None
            assert p['next_big_blind'] is None
            assert p['hands_to_next'] is None
        else:
            assert p['next_small_blind'] == (10, 20, 40, 80, 160, 320, 640, 1280)[view['level_index']]
            assert p['next_big_blind'] == (20, 40, 80, 160, 320, 640, 1280, 2560)[view['level_index']]
            assert p['hands_to_next'] == 2 - ((view['hand_number'] - 1) % 2)
        assert view['your_turn']
        game.play_action(1, expected_revision=view['revision'])
    else:
        raise AssertionError('Tournament failed to terminate')


def test_http_roundtrip_and_elimination_placement_are_consistent():
    app = FastAPI()
    registry = TournamentSessionRegistry()
    app.include_router(build_tournament_router(registry))
    with TestClient(app) as client:
        response = client.post('/api/v1/tournament/sessions', json={
            'seed': 13, 'starting_stack': 100, 'hands_per_level': 2, 'max_hands': 200,
        })
        assert response.status_code == 201, response.text
        response_obj = response.json()
        session_id = response_obj['session_id']
        for _ in range(500):
            view = response_obj['state']
            assert sum(view['tournament_stacks']) == 600
            assert len(view['alive_seats']) == sum(s > 0 for s in registry.get(session_id).env.players.stacks)
            if view['tournament_over']:
                assert all(s['rank'] is not None for s in view['seats'])
                assert view['seats'][view['winner_seat']]['rank'] == 1
                assert len(view['alive_seats']) == 1
                break
            response = client.post(f'/api/v1/tournament/sessions/{session_id}/actions', json={
                'action_index': 1, 'expected_revision': view['revision'],
            })
            assert response.status_code == 200, response.text
            response_obj = response.json()
        else:
            raise AssertionError('Tournament failed to finish')

````````

<!-- END_PY_SOURCE path="tests/test_tournament_env15_progress.py" -->

## `tests/test_tournament_env161_showdown.py`

Original blob `d7c8c1e343f1c46b541a29def1668b58feb36dbe` (7741 bytes).

<a name="py-323"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_tournament_env161_showdown.py" sha="d7c8c1e343f1c46b541a29def1668b58feb36dbe" -->

````````python
"""Phase 1.6.1: show only legitimate showdown hands, and only after settlement."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.tournament_router import build_tournament_router
from holdem_ai.environment.tournament_env import TournamentEnv
from holdem_ai.tournament_service.journal import TournamentJournal
from holdem_ai.tournament_service.session import TournamentSessionRegistry


class CallPolicy:
    def select_action(self, observation):
        return 1


class FoldPolicy:
    def select_action(self, observation):
        return 0


def _create(registry, policy, *, seed=21):
    return registry.create(seed=seed, starting_stack=1000,
                           policies={seat: policy for seat in range(1, 6)})


def test_showdown_reveals_exactly_nonfolded_contenders_and_no_early_leak(tmp_path):
    journal = TournamentJournal(tmp_path / 'showdown.sqlite3')
    registry = TournamentSessionRegistry(journal=journal)
    sid, state = _create(registry, CallPolicy())
    assert state['last_hand'] is None
    assert 'showdown_cards' not in state
    assert 'hole_cards' not in str(state)  # no opponent holes in live view
    pending = journal.replay(sid, 1)
    assert pending['settlement'] is None
    assert pending['complete'] is False
    assert 'showdown_cards' not in str(pending['actions'])
    # Human always checks/calls until first hand settles; six seats reach showdown.
    for _ in range(20):
        if state['hands_completed']:
            break
        state = registry.get(sid).play_action(1, expected_revision=state['revision'])
    assert state['hands_completed'] > 0
    hand = registry.get(sid).history()['hands'][0]
    reveals = hand['showdown_cards']
    engine_record = registry.get(sid).env.hand_history[0]
    assert len(reveals) == 6
    assert sorted(entry['seat'] for entry in reveals) == list(range(6))
    assert all(entry['cards'] == list(engine_record.showdown_hole_cards[entry['seat']]) for entry in reveals)
    assert len(set(hand['board'] + [c for entry in reveals for c in entry['cards']])) == 17
    assert state['last_hand']['showdown_cards'] == reveals
    archived = TournamentJournal(journal.path).replay(sid, 1)
    assert archived['settlement']['showdown_cards'] == reveals
    assert all('showdown_cards' not in action and 'actor_hole' not in action for action in archived['actions'])
    exported = journal.actor_export_rows(sid)
    assert exported
    assert all('showdown_cards' not in str(row) for row in exported)
    assert all('opponent_hole' not in str(row) for row in exported)


def test_uncontested_pot_never_opens_winners_or_folders_hole_cards(tmp_path):
    registry = TournamentSessionRegistry(journal=TournamentJournal(tmp_path / 'folds.sqlite3'))
    sid, state = _create(registry, FoldPolicy())
    # Bot seats after the blinds fold; hero folds as well, so the BB wins uncontested.
    if not state['hands_completed']:
        state = registry.get(sid).play_action(0, expected_revision=state['revision'])
    assert state['hands_completed'] >= 1
    original = registry.get(sid).env.hand_history[0]
    assert sum(not f for f in original.hand_result.folded) == 1
    assert original.showdown_hole_cards == ()
    assert registry.get(sid).history()['hands'][0]['showdown_cards'] == []
    assert registry.journal.replay(sid, 1)['settlement']['showdown_cards'] == []


def test_only_surviving_showdown_seats_are_exposed_when_others_fold(tmp_path):
    journal = TournamentJournal(tmp_path / 'mixed.sqlite3')
    registry = TournamentSessionRegistry(journal=journal)
    policy_map = {1: CallPolicy(), 2: CallPolicy(), 3: FoldPolicy(), 4: FoldPolicy(), 5: FoldPolicy()}
    sid, state = registry.create(seed=22, starting_stack=1000, policies=policy_map)
    for _ in range(20):
        if state['hands_completed']:
            break
        state = registry.get(sid).play_action(1, expected_revision=state['revision'])
    assert state['hands_completed'] >= 1
    rec = registry.get(sid).env.hand_history[0]
    shown = registry.get(sid).history()['hands'][0]['showdown_cards']
    expected = {i for i,folded in enumerate(rec.hand_result.folded) if not folded}
    assert len(expected) >= 2
    assert {row['seat'] for row in shown} == expected
    assert all(len(row['cards']) == 2 for row in shown)


def test_zero_action_all_in_hand_still_saves_showdown_cards():
    # Both blinds consume both 1-chip stacks, so the board runs out with no actions.
    env = TournamentEnv(num_players=2, starting_stack=1, seed=9, max_hands=30)
    env.reset()
    assert env.hand_history
    rec = env.hand_history[0]
    assert rec.hand_result.action_history == ()
    assert len(rec.hand_result.board) == 5
    assert len(rec.showdown_hole_cards) == 2
    assert all(len(cards) == 2 for cards in rec.showdown_hole_cards)


def test_http_response_and_durable_replay_expose_only_post_settlement(tmp_path):
    journal = TournamentJournal(tmp_path / 'api.sqlite3')
    registry = TournamentSessionRegistry(journal=journal)
    app = FastAPI()
    app.include_router(build_tournament_router(registry))
    with TestClient(app) as client:
        sid, state = _create(registry, CallPolicy(), seed=31)
        for _ in range(20):
            if state['hands_completed']:
                break
            response = client.post(f'/api/v1/tournament/sessions/{sid}/actions',
                                   json={'action_index': 1, 'expected_revision': state['revision']})
            assert response.status_code == 200, response.text
            state = response.json()['state']
        assert state['hands_completed'] >= 1
        reveals = state['last_hand']['showdown_cards']
        assert len(reveals) == 6
        history = client.get(f'/api/v1/tournament/sessions/{sid}/history')
        assert history.status_code == 200
        assert history.json()['hands'][0]['showdown_cards'] == reveals
        replay = client.get(f'/api/v1/tournament/archive/{sid}/hands/1/replay')
        assert replay.status_code == 200
        assert replay.json()['settlement']['showdown_cards'] == reveals
        assert all('showdown_cards' not in action for action in replay.json()['actions'])
    reopened = TournamentJournal(journal.path)
    assert reopened.replay(sid, 1)['settlement']['showdown_cards'] == reveals


def test_legacy_archive_without_showdown_cards_remains_readable(tmp_path):
    journal = TournamentJournal(tmp_path / 'legacy.sqlite3')
    registry = TournamentSessionRegistry(journal=journal)
    sid, st = _create(registry, CallPolicy())
    for _ in range(20):
        if st['hands_completed']:
            break
        st = registry.get(sid).play_action(1, expected_revision=st['revision'])
    # Legacy records omit the new optional field. The existing API can still read them.
    import json, sqlite3
    with sqlite3.connect(journal.path) as db:
        raw = db.execute('SELECT data_json FROM tournament_hands WHERE session_id=? AND hand_number=1', (sid,)).fetchone()[0]
        payload = json.loads(raw)
        payload.pop('showdown_cards')
        db.execute('UPDATE tournament_hands SET data_json=? WHERE session_id=? AND hand_number=1',
                   (json.dumps(payload), sid))
    assert 'showdown_cards' not in TournamentJournal(journal.path).replay(sid, 1)['settlement']
    # The strict response schema defaults the new field for older journals.
    app = FastAPI()
    app.include_router(build_tournament_router(registry))
    with TestClient(app) as client:
        response = client.get(f'/api/v1/tournament/sessions/{sid}/history')
        assert response.status_code == 200
        assert response.json()['hands'][0]['showdown_cards']  # in-memory live state is modern

````````

<!-- END_PY_SOURCE path="tests/test_tournament_env161_showdown.py" -->

## `tests/test_tournament_env16_persistence.py`

Original blob `f0655533d21cd7fa537b46f7e20756328c58f41c` (6901 bytes).

<a name="py-324"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_tournament_env16_persistence.py" sha="f0655533d21cd7fa537b46f7e20756328c58f41c" -->

````````python
"""Phase 1.6: durable actor-visible transitions and public replay."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import json
import sqlite3

from backend.tournament_router import build_tournament_router
from holdem_ai.tournament_service.journal import TournamentJournal
from holdem_ai.tournament_service.session import TournamentSessionRegistry


def test_durable_journal_and_archive_survive_new_process(tmp_path):
    path = tmp_path / 'tournament.sqlite3'
    journal = TournamentJournal(path)
    registry = TournamentSessionRegistry(journal=journal)
    sid, state = registry.create(seed=7, starting_stack=140, hands_per_level=3,
                                 blind_levels=((1,2),(2,4),(4,8),(8,16),(16,32)), max_hands=300)
    for _ in range(180):
        if state['tournament_over'] or state['hands_completed'] >= 3:
            break
        state = registry.get(sid).play_action(1,expected_revision=state['revision'])
    else:
        raise AssertionError('Expected at least 3 hands')
    assert state['hands_completed'] >= 3
    assert sum(state['tournament_stacks']) == 840
    archive = TournamentJournal(path)  # simulate Backend restart, not a live resume
    items = archive.list_sessions()['sessions']
    assert len(items) == 1 and items[0]['session_id'] == sid
    assert archive.hands(sid)['total'] == state['hands_completed']
    hand1 = archive.replay(sid, 1)
    assert hand1['complete'] and hand1['settlement']['hand_number'] == 1
    assert hand1['actions']
    assert [a['action_no'] for a in hand1['actions']] == list(range(1,len(hand1['actions'])+1))
    assert [a['seat'] for a in hand1['actions']] == [a['seat'] for a in hand1['settlement']['action_history']]
    assert sum(hand1['settlement']['stacks_after']) == 840
    for a in hand1['actions']:
        assert 'actor_hole' not in a
        assert len(a['public_before']['stacks']) == 6
        assert 'deck' not in json.dumps(a)
        if a['seat'] != 0: assert a['hero_hole'] == []
    registry.delete(sid)
    assert archive.list_sessions()['sessions'][0]['status'] in ('closed','complete')
    assert archive.hands(sid)['total'] >= 3


def test_private_actor_hole_is_only_own_hole_and_public_replay_redacts(tmp_path):
    journal = TournamentJournal(tmp_path / 'private.sqlite3')
    registry = TournamentSessionRegistry(journal=journal)
    sid, state = registry.create(seed=12, starting_stack=300)
    state = registry.get(sid).play_action(1,expected_revision=state['revision'])
    rows = journal.actor_export_rows(sid)
    assert rows
    hero_rows = [r for r in rows if r['acting_seat'] == 0]
    assert hero_rows
    for r in rows:
        assert len(r['observation']['own_hole_cards']) == 2
        assert r['action']['index'] in r['observation']['legal_actions'] or r['action']['index'] == 5
        assert 'opponent_hole' not in json.dumps(r)
        assert 'deck' not in json.dumps(r)
        assert r['schema'] == 'tournament.actor_transition.v1'
    public = journal.replay(sid,1)
    for a in public['actions']:
        if a['seat'] == 0: assert len(a['hero_hole']) == 2
        else: assert a['hero_hole'] == []
    assert all('actor_hole' not in a for a in public['actions'])


def test_invalid_action_does_not_persist(tmp_path):
    journal=TournamentJournal(tmp_path/'invalid.sqlite3')
    registry=TournamentSessionRegistry(journal=journal)
    sid,st=registry.create(seed=15)
    before=len(journal.actor_export_rows(sid))
    try:
        registry.get(sid).play_action(1,expected_revision=st['revision']-1)
    except ValueError as e:
        assert 'stale' in str(e)
    else: raise AssertionError('Expected stale revision rejection')
    assert len(journal.actor_export_rows(sid)) == before


def test_http_archive_replay_and_export_after_registry_close(tmp_path):
    path=tmp_path/'api.sqlite3'
    registry=TournamentSessionRegistry(journal=TournamentJournal(path))
    app=FastAPI()
    app.include_router(build_tournament_router(registry))
    with TestClient(app) as client:
        created=client.post('/api/v1/tournament/sessions',json={'seed':3,'starting_stack':100,'hands_per_level':2,'max_hands':250})
        assert created.status_code == 201,created.text
        sid=created.json()['session_id']
        state=created.json()['state']
        for _ in range(150):
            if state['hands_completed']>=1:break
            response=client.post(f'/api/v1/tournament/sessions/{sid}/actions',json={'action_index':1,'expected_revision':state['revision']})
            assert response.status_code==200,response.text
            state=response.json()['state']
        assert state['hands_completed']>=1
        result=client.get('/api/v1/tournament/archive')
        assert result.status_code==200
        assert any(s['session_id']==sid for s in result.json()['sessions'])
        hist=client.get(f'/api/v1/tournament/archive/{sid}/hands')
        assert hist.status_code==200 and hist.json()['total']>=1
        replay=client.get(f'/api/v1/tournament/archive/{sid}/hands/1/replay')
        assert replay.status_code==200 and replay.json()['complete']
        resp=client.get(f'/api/v1/tournament/archive/{sid}/export')
        assert resp.status_code==200
        assert resp.headers['content-type'].startswith('application/x-ndjson')
        rows=[json.loads(line) for line in resp.text.splitlines()]
        assert rows and all(len(row['observation']['own_hole_cards'])==2 for row in rows)
        assert all('deck' not in json.dumps(row) for row in rows)
        response=client.delete(f'/api/v1/tournament/sessions/{sid}')
        assert response.status_code==204
        assert client.get(f'/api/v1/tournament/sessions/{sid}').status_code==404
        assert client.get(f'/api/v1/tournament/archive/{sid}/hands/1/replay').status_code==200
        assert client.get('/api/v1/tournament/archive/missing/export').status_code==404
    restarted = TournamentJournal(path)
    assert restarted.hands(sid)['total']>=1


def test_invalid_checkpoint_does_not_leave_archived_session(tmp_path):
    journal=TournamentJournal(tmp_path/'failed.sqlite3')
    reg=TournamentSessionRegistry(journal=journal)
    try:
        reg.create(lineup='trained_1_improved_4',seed=3)
    except (FileNotFoundError,ValueError,RuntimeError):
        pass
    else:
        # If a real trusted multi-seat checkpoint exists in the environment,
        # successful creation is valid; this is a rollback-only test elsewhere.
        return
    assert journal.list_sessions()['total']==0


def test_no_privileged_tables_or_fields(tmp_path):
    journal=TournamentJournal(tmp_path/'schema.sqlite3')
    with sqlite3.connect(journal.path) as db:
        tables={x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'deck' not in tables
    assert not any('optimizer' in t or 'weights' in t for t in tables)

````````

<!-- END_PY_SOURCE path="tests/test_tournament_env16_persistence.py" -->

## `tests/test_tournament_env17_integration.py`

Original blob `8b2d67739f8b9a9d047967e9167a8a0c2f954a3a` (4731 bytes).

<a name="py-325"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_tournament_env17_integration.py" sha="8b2d67739f8b9a9d047967e9167a8a0c2f954a3a" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/test_tournament_env17_integration.py" -->

## `tests/test_tournament_env17_release_audit.py`

Original blob `b43f19dd17020a68391721429e3dc5185b0863e0` (2253 bytes).

<a name="py-326"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_tournament_env17_release_audit.py" sha="b43f19dd17020a68391721429e3dc5185b0863e0" -->

````````python
"""Unit tests for the privacy/staging release gate, no project game imports."""
from __future__ import annotations

from pathlib import Path
import subprocess

from scripts.tournament_env17_release_audit import check_release, unsafe_reason


def _git(root: Path, *args: str) -> None:
    subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)


def _project(tmp_path: Path) -> Path:
    (tmp_path / 'frontend').mkdir()
    (tmp_path / 'backend').mkdir()
    (tmp_path / 'frontend' / 'package.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'backend' / 'main.py').write_text('', encoding='utf-8')
    (tmp_path / '.gitignore').write_text(
        '.local/\nruns/\ncheckpoints/\nfrontend/node_modules/\n'
        '*.pt\n*.pth\n*.jsonl\n*.sqlite3\n', encoding='utf-8',
    )
    _git(tmp_path, 'init', '-q')
    _git(tmp_path, 'add', '--', '.gitignore', 'backend/main.py', 'frontend/package.json')
    return tmp_path


def test_unsafe_patterns_block_private_artifacts():
    assert unsafe_reason('runs/expand_phase03_pilot/best.pt')
    assert unsafe_reason('.local/tournament_history.sqlite3')
    assert unsafe_reason('frontend/node_modules/react/index.js')
    assert unsafe_reason('checkpoints/model.pt')
    assert unsafe_reason('.env.local')
    assert unsafe_reason('data/hand_histories/hand.jsonl')
    assert unsafe_reason('frontend/src/tournament/TournamentApp.tsx') is None
    assert unsafe_reason('tests/test_tournament_env17_release_audit.py') is None
    assert unsafe_reason('data/hand_histories/.gitkeep') is None


def test_read_only_audit_works_for_safe_staging(tmp_path):
    root = _project(tmp_path)
    status, lines = check_release(root, strict=True)
    assert status == 0, '\n'.join(lines)
    assert not (root / '.local').exists()


def test_audit_blocks_explicitly_force_staged_checkpoint(tmp_path):
    root = _project(tmp_path)
    folder = root / 'runs' / 'expand_phase03_pilot'
    folder.mkdir(parents=True)
    (folder / 'best.pt').write_bytes(b'fake-checkpoint')
    _git(root, 'add', '-f', '--', 'runs/expand_phase03_pilot/best.pt')
    status, lines = check_release(root, strict=True)
    assert status != 0
    assert any('staged private/generated file' in line for line in lines)

````````

<!-- END_PY_SOURCE path="tests/test_tournament_env17_release_audit.py" -->

## `tests/test_tournament_env1_api.py`

Original blob `342cd89e5a4dd719d99a5a1ff0f5d897ae6a6d30` (10183 bytes).

<a name="py-327"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_tournament_env1_api.py" sha="342cd89e5a4dd719d99a5a1ff0f5d897ae6a6d30" -->

````````python
"""Tournament ENV 1.1 -- API/engine integration and privacy boundary."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from holdem_ai.tournament_service.session import TournamentGameSession, TournamentSessionRegistry


SHORT = {
    "seed": 13,
    "starting_stack": 100,
    "hands_per_level": 2,
    "blind_levels": [[5, 10], [10, 20], [20, 40], [40, 80], [80, 160]],
}


def test_health_and_hunl_coexist():
    with TestClient(create_app()) as client:
        assert client.get("/api/v1/tournament/health").json() == {
            "status": "ok", "service": "tournament-env-local", "phase": "1.1"
        }
        assert client.get("/api/v1/hunl/health").status_code == 200


def test_create_projection_and_no_opponent_cards():
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/tournament/sessions", json=SHORT)
        assert response.status_code == 201, response.text
        response_data = response.json()
        session = client.app.state.tournament_env_p1_registry.get(response_data["session_id"])
        state = response_data["state"]
        assert response.headers["Cache-Control"] == "no-store"
        assert state["game_type"] == "TOURNAMENT_6MAX"
        assert state["phase"] == "playing"
        assert state["your_turn"] is True
        assert state["to_act"] == 0
        assert [s["seat"] for s in state["seats"]] == list(range(6))
        assert [s["role"] for s in state["seats"]] == ["human"] + ["baseline_bot"] * 5
        assert len(state["hero_cards"]) == 2
        assert sum(s["stack"] for s in state["seats"]) + state["pot"] == 600
        for seat in range(1, 6):
            private = session.env._hand.hole_cards[seat]
            for card in private:
                if card not in state["board"] and card not in state["hero_cards"]:
                    assert json.dumps(card) not in json.dumps(state)
        assert "hole_cards" not in json.dumps(state)
        assert "deck" not in json.dumps(state)
        assert "burned" not in json.dumps(state)
        assert "critic" not in json.dumps(state)
        assert client.get(f'/api/v1/tournament/sessions/{response_data["session_id"]}').json() == response_data


def test_stale_revision_illegal_action_and_unknown_id():
    with TestClient(create_app()) as client:
        payload = client.post("/api/v1/tournament/sessions", json=SHORT).json()
        sid, state = payload["session_id"], payload["state"]
        action_path = f"/api/v1/tournament/sessions/{sid}/actions"
        assert client.get("/api/v1/tournament/sessions/unknown").status_code == 404
        assert client.post(action_path, json={"action_index": 1, "expected_revision": 0}).status_code == 422
        assert client.post(action_path, json={"action_index": 1, "expected_revision": state["revision"] + 1}).status_code == 409
        assert client.post(action_path, json={"action_index": 9, "expected_revision": state["revision"]}).status_code == 422
        assert client.post(action_path, json={"action_index": 1, "expected_revision": state["revision"], "private": "oops"}).status_code == 422
        valid_action = state["actions"][0]["index"]
        response = client.post(action_path, json={"action_index": valid_action, "expected_revision": state["revision"]})
        assert response.status_code == 200, response.text
        assert response.json()["state"]["revision"] > state["revision"]
        assert client.post(action_path, json={"action_index": valid_action, "expected_revision": state["revision"]}).status_code == 409


def test_invalid_raise_is_atomic():
    with TestClient(create_app()) as client:
        payload = client.post("/api/v1/tournament/sessions", json=SHORT).json()
        sid, state = payload["session_id"], payload["state"]
        invalid = client.post(f"/api/v1/tournament/sessions/{sid}/raise-to", json={
            "raise_to": 999999, "expected_revision": state["revision"]
        })
        assert invalid.status_code == 422, invalid.text
        after = client.get(f"/api/v1/tournament/sessions/{sid}").json()["state"]
        assert state == after


def test_tournament_completion_and_history():
    with TestClient(create_app()) as client:
        payload = client.post("/api/v1/tournament/sessions", json=SHORT).json()
        sid = payload["session_id"]
        last_hands = 0
        for iteration in range(2000):
            state = payload["state"]
            assert sum(s["stack"] for s in state["seats"]) + state["pot"] == state["total_chips"]
            assert state["hands_completed"] >= last_hands
            last_hands = state["hands_completed"]
            if state["tournament_over"]:
                break
            assert state["your_turn"]
            payload = client.post(f"/api/v1/tournament/sessions/{sid}/actions", json={
                "action_index": 1, "expected_revision": state["revision"]
            }).json()
        else:
            pytest.fail("tournament did not terminate")
        state = payload["state"]
        assert state["winner_seat"] in range(6)
        assert sum(s["stack"] > 0 for s in state["seats"]) == 1
        assert sum(s["stack"] for s in state["seats"]) == 600
        assert state["hands_completed"] >= 1
        history = client.get(f"/api/v1/tournament/sessions/{sid}/history?offset=0&limit=100")
        assert history.status_code == 200, history.text
        data = history.json()
        assert data["total"] == state["hands_completed"]
        assert len(data["hands"]) == data["total"]
        for i, hand in enumerate(data["hands"]):
            assert hand["hand_number"] == i + 1
            assert sum(hand["stacks_before"]) == sum(hand["stacks_after"]) == 600
            if i:
                assert hand["stacks_before"] == data["hands"][i - 1]["stacks_after"]
            for knocked in hand["eliminated"]:
                assert hand["stacks_before"][knocked] > 0
                assert hand["stacks_after"][knocked] == 0
        assert client.post(f"/api/v1/tournament/sessions/{sid}/actions", json={
            "action_index": 1, "expected_revision": state["revision"]
        }).status_code == 409
        assert client.delete(f"/api/v1/tournament/sessions/{sid}").status_code == 204
        assert client.get(f"/api/v1/tournament/sessions/{sid}").status_code == 404


def test_invalid_creation_and_registry_cap():
    with TestClient(create_app(max_sessions=1)) as client:
        for data in (
            {"starting_stack": -5},
            {"starting_stack": True},
            {"blind_levels": [[20, 10]]},
            {"blind_levels": [[5, 10], [1, 2]]},
            {"hands_per_level": 0},
            {"button": 6},
        ):
            response = client.post("/api/v1/tournament/sessions", json=data)
            assert response.status_code == 422, (data, response.text)
        first = client.post("/api/v1/tournament/sessions", json=SHORT)
        assert first.status_code == 201
        second = client.post("/api/v1/tournament/sessions", json=SHORT)
        assert second.status_code == 503
        assert client.delete(f'/api/v1/tournament/sessions/{first.json()["session_id"]}').status_code == 204
        assert client.post("/api/v1/tournament/sessions", json=SHORT).status_code == 201


def test_seed_reproducibility_and_baseline_legal():
    a = TournamentGameSession(seed=89, starting_stack=100, hands_per_level=2,
                              blind_levels=[(5, 10), (10, 20), (20, 40), (40, 80)])
    b = TournamentGameSession(seed=89, starting_stack=100, hands_per_level=2,
                              blind_levels=[(5, 10), (10, 20), (20, 40), (40, 80)])
    for _ in range(300):
        v1, v2 = a.view(), b.view()
        assert v1 == v2
        if v1["tournament_over"]:
            break
        a.play_action(1, expected_revision=v1["revision"])
        b.play_action(1, expected_revision=v2["revision"])
    else:
        pytest.fail("not completed in time")


def test_no_hunl_agent_policy_is_silently_used_for_6max():
    # Incompatible HUNL200 masks are never automatically converted here.
    game = TournamentGameSession(seed=17, starting_stack=100)
    for seat in game.view()["seats"][1:]:
        assert seat["role"] == "baseline_bot"
    assert not hasattr(game, "train")


def test_successful_raise_to_with_engine_boundaries():
    with TestClient(create_app()) as client:
        payload = client.post("/api/v1/tournament/sessions", json=SHORT).json()
        sid, state = payload["session_id"], payload["state"]
        assert state["your_turn"]
        if not state["can_raise_to"]:
            pytest.skip("initial hand cannot raise for this seed")
        target = min(state["min_raise_to"], state["max_raise_to"])
        response = client.post(f"/api/v1/tournament/sessions/{sid}/raise-to", json={
            "raise_to": target, "expected_revision": state["revision"]
        })
        assert response.status_code == 200, response.text
        assert response.json()["state"]["revision"] > state["revision"]


def test_history_is_paginated_and_no_private_information():
    game = TournamentGameSession(seed=47, starting_stack=100, hands_per_level=2,
                                 blind_levels=[(5, 10), (10, 20), (20, 40), (40, 80)])
    for _ in range(150):
        state = game.view()
        if state["tournament_over"]:
            break
        game.play_action(0, expected_revision=state["revision"])
    data = game.history(limit=2)
    assert data["total"] == len(game.env.hand_history)
    assert len(data["hands"]) <= 2
    if len(data["hands"]) == 2:
        assert game.history(offset=1, limit=1)["hands"][0] == data["hands"][1]
    assert '"hole_cards"' not in json.dumps(data)
    assert '"burned"' not in json.dumps(data)


def test_bad_bot_action_refused_instead_of_mask_guessing():
    class IllegalPolicy:
        def select_action(self, state):
            return 6  # HUNL200 action indices cannot be passed as Phase-4 indices.

    with pytest.raises(RuntimeError, match="illegal tournament action"):
        TournamentGameSession(seed=71, policies={i: IllegalPolicy() for i in range(1, 6)})

````````

<!-- END_PY_SOURCE path="tests/test_tournament_env1_api.py" -->

