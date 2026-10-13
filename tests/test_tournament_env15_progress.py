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
