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
