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
