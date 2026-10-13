"""API smoke: uv run --extra dev --extra gui python -m scripts.tournament_env1_smoke"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import create_app


def main() -> None:
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/tournament/sessions", json={
            "seed": 19, "starting_stack": 100,
            "blind_levels": [[5, 10], [10, 20], [20, 40], [40, 80], [80, 160]],
            "hands_per_level": 2,
        })
        response.raise_for_status()
        payload = response.json()
        sid = payload["session_id"]
        total = payload["state"]["total_chips"]
        actions = 0
        while not payload["state"]["tournament_over"]:
            state = payload["state"]
            assert state["your_turn"] and state["to_act"] == 0
            assert sum(s["stack"] for s in state["seats"]) + state["pot"] == total
            assert len(state["seats"]) == 6
            response = client.post(f"/api/v1/tournament/sessions/{sid}/actions", json={
                "expected_revision": state["revision"], "action_index": 1,
            })
            response.raise_for_status()
            payload = response.json()
            actions += 1
            if actions > 1000:
                raise RuntimeError("smoke exceeded 1000 human decisions")
        final = payload["state"]
        assert final["winner_seat"] in range(6)
        assert sum(s["stack"] for s in final["seats"]) == total
        history = client.get(f"/api/v1/tournament/sessions/{sid}/history").json()
        assert history["total"] == final["hands_completed"]
        print({"phase": "Tournament ENV 1.1", "human_actions": actions,
               "hands": final["hands_completed"], "winner_seat": final["winner_seat"],
               "chips": total, "result": "passed"})


if __name__ == "__main__":
    main()
