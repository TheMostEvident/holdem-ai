"""GUI-1.1 HTTP contract tests (requires GUI-1.0 + FastAPI/httpx)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


@pytest.fixture
def client():
    with TestClient(create_app(max_sessions=16)) as c:
        yield c


def new_game(client, **kwargs):
    res = client.post("/api/v1/hunl/sessions", json={"bot": "calling_station", "seed": 42, **kwargs})
    assert res.status_code == 201, res.text
    assert res.headers["cache-control"] == "no-store"
    return res.json()


def act(client, sid, state, kind=None):
    assert state["phase"] == "playing" and state["your_turn"]
    offers = state["actions"]
    if kind is None:
        offer = next((a for a in offers if a["kind"] in ("check", "call")), offers[0])
    else:
        offer = next(a for a in offers if a["kind"] == kind)
    response = client.post(
        f"/api/v1/hunl/sessions/{sid}/actions",
        json={"action_index": offer["index"], "expected_revision": state["revision"]},
    )
    assert response.status_code == 200, response.text
    return response.json()["state"]


def finish_hand(client, sid, state):
    for _ in range(100):
        if state["phase"] == "complete":
            return state
        state = act(client, sid, state)
    pytest.fail("hand did not terminate after 100 human actions")


def test_health_and_openapi(client):
    r = client.get("/api/v1/hunl/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "hunl-gui-local"}
    assert client.get("/openapi.json").status_code == 200


def test_create_get_state_no_private_leaks(client):
    item = new_game(client)
    sid, state = item["session_id"], item["state"]
    assert state["hero_seat"] == 0
    assert state["game_type"] == "HUNL200"
    assert state["your_turn"] and state["to_act"] == 0
    assert len(state["hero_cards"]) == 2
    assert sum(s["stack"] + s["total_contribution"] for s in state["seats"]) == 40_000
    assert client.get(f"/api/v1/hunl/sessions/{sid}").json() == item
    # The trusted engine has both hands; the browser must not have opponent's.
    session = client.app.state.gui11_registry.get(sid)
    secret = session._adapter.engine.hole_cards[1]  # only test accesses privileged private member
    visible = json.dumps(item)
    for card in secret:
        assert card not in state["hero_cards"]
        assert '"' + card + '"' not in visible
    assert "hole_cards" not in visible
    assert "critic" not in visible.lower()
    assert "_deck" not in visible
    assert "training_state" not in visible
    assert state["hand_result"] is None


def test_human_action_rev_increments_and_stale_request_fails(client):
    item = new_game(client)
    sid, before = item["session_id"], item["state"]
    after = act(client, sid, before)
    assert after["revision"] > before["revision"]
    stale = client.post(f"/api/v1/hunl/sessions/{sid}/actions", json={
        "action_index": before["actions"][0]["index"], "expected_revision": before["revision"]
    })
    assert stale.status_code == 409
    assert client.get(f"/api/v1/hunl/sessions/{sid}").json()["state"] == after


def test_invalid_action_rejected_without_mutation(client):
    item = new_game(client)
    sid, state = item["session_id"], item["state"]
    illegal = next(i for i in range(7) if i not in {a["index"] for a in state["actions"]})
    response = client.post(f"/api/v1/hunl/sessions/{sid}/actions", json={
        "action_index": illegal, "expected_revision": state["revision"]
    })
    assert response.status_code == 422
    assert client.get(f"/api/v1/hunl/sessions/{sid}").json()["state"] == state


def test_strict_http_payloads(client):
    for invalid in (
        {"bot": "unknown"}, {"bot": "random", "seed": "42"},
        {"big_blind": 10, "small_blind": 10}, {"surprise": 3},
        {"stack_bb": 1}, {"big_blind": True},
    ):
        assert client.post("/api/v1/hunl/sessions", json=invalid).status_code == 422
    item = new_game(client)
    sid, state = item["session_id"], item["state"]
    for invalid in (
        {"action_index": "2", "expected_revision": state["revision"]},
        {"action_index": True, "expected_revision": state["revision"]},
        {"action_index": 10, "expected_revision": state["revision"]},
        {"action_index": 2, "expected_revision": 0},
        {"action_index": 2, "expected_revision": state["revision"], "unexpected": 1},
    ):
        assert client.post(f"/api/v1/hunl/sessions/{sid}/actions", json=invalid).status_code == 422
    assert client.get(f"/api/v1/hunl/sessions/{sid}").json()["state"] == state


def test_complete_hand_showdown_and_next_hand_revision(client):
    item = new_game(client)
    sid, state = item["session_id"], item["state"]
    assert client.post(f"/api/v1/hunl/sessions/{sid}/next-hand", json={
        "expected_revision": state["revision"]
    }).status_code == 409
    finished = finish_hand(client, sid, state)
    assert finished["hand_result"]["showdown"] is True
    assert len(finished["hand_result"]["opponent_cards"]) == 2
    assert finished["completed_hands"] == 1
    assert finished["hand_result"]["payoffs_chips"][0] == -finished["hand_result"]["payoffs_chips"][1]
    assert finished["hand_result"]["payoffs_bb"][0] == finished["cumulative_bb"]
    res = client.post(f"/api/v1/hunl/sessions/{sid}/next-hand", json={
        "expected_revision": finished["revision"]
    })
    assert res.status_code == 200, res.text
    next_state = res.json()["state"]
    assert next_state["hand_number"] == 2
    assert next_state["button"] == 1
    assert next_state["completed_hands"] == 1
    assert next_state["revision"] > finished["revision"]
    assert next_state["hero_seat"] == 0
    assert next_state["hand_result"] is None
    assert len(next_state["hero_cards"]) == 2
    assert sum(s["stack"] + s["total_contribution"] for s in next_state["seats"]) == 40_000
    again = client.post(f"/api/v1/hunl/sessions/{sid}/next-hand", json={
        "expected_revision": finished["revision"]
    })
    assert again.status_code == 409


def test_multiple_hands_stats_and_fold_hidden(client):
    game = new_game(client)
    sid, state = game["session_id"], game["state"]
    fold = act(client, sid, state, kind="fold")
    assert fold["phase"] == "complete"
    assert fold["hand_result"]["showdown"] is False
    assert fold["hand_result"]["opponent_cards"] is None
    assert fold["completed_hands"] == 1
    assert fold["bb_per_100"] == pytest.approx(100 * fold["cumulative_bb"])
    next_res = client.post(f"/api/v1/hunl/sessions/{sid}/next-hand", json={
        "expected_revision": fold["revision"]
    })
    assert next_res.status_code == 200
    second = finish_hand(client, sid, next_res.json()["state"])
    assert second["completed_hands"] == 2
    assert second["bb_per_100"] == pytest.approx(50 * second["cumulative_bb"])


def test_missing_and_deleted_session(client):
    assert client.get("/api/v1/hunl/sessions/not-real").status_code == 404
    item = new_game(client)
    sid = item["session_id"]
    r = client.delete(f"/api/v1/hunl/sessions/{sid}")
    assert r.status_code == 204
    assert client.get(f"/api/v1/hunl/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/v1/hunl/sessions/{sid}").status_code == 404


def test_sessions_isolated_and_capacity():
    with TestClient(create_app(max_sessions=1)) as c:
        first = new_game(c)
        assert c.post("/api/v1/hunl/sessions", json={}).status_code == 503
        assert c.delete(f"/api/v1/hunl/sessions/{first['session_id']}").status_code == 204
        assert c.post("/api/v1/hunl/sessions", json={}).status_code == 201
    with TestClient(create_app()) as c:
        a = new_game(c, seed=11)
        b = new_game(c, seed=12)
        assert a["session_id"] != b["session_id"]
        state_a = act(c, a["session_id"], a["state"])
        assert c.get(f"/api/v1/hunl/sessions/{b['session_id']}").json()["state"] == b["state"]
        assert state_a["revision"] > a["state"]["revision"]


def test_cors_is_only_local_frontend(client):
    allow = client.options(
        "/api/v1/hunl/sessions", headers={"Origin": "http://localhost:5173",
                                           "Access-Control-Request-Method": "POST"}
    )
    assert allow.status_code == 200
    assert allow.headers["access-control-allow-origin"] == "http://localhost:5173"
    denied = client.options(
        "/api/v1/hunl/sessions", headers={"Origin": "https://untrusted.example",
                                           "Access-Control-Request-Method": "POST"}
    )
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_builtin_bot_variants(client):
    for bot in ("random", "calling_station", "tight", "pot_pressure"):
        res = client.post("/api/v1/hunl/sessions", json={"bot": bot, "seed": 1})
        assert res.status_code == 201, (bot, res.text)
        s = res.json()["state"]
        assert s["your_turn"] or s["phase"] == "complete"
