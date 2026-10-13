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
