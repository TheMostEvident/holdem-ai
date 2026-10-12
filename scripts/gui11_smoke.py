"""HTTP smoke exercise without starting an external server.

Run: uv run --extra dev --extra gui python -m scripts.gui11_smoke --hands 10
"""
from __future__ import annotations

import argparse

from fastapi.testclient import TestClient

from backend.main import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="GUI-1.1 local HTTP smoke exercise")
    ap.add_argument("--hands", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.hands < 1:
        ap.error("--hands must be positive")
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/hunl/sessions", json={"bot": "calling_station", "seed": args.seed})
        response.raise_for_status()
        item = response.json()
        sid, state = item["session_id"], item["state"]
        for hand in range(1, args.hands + 1):
            while state["phase"] != "complete":
                assert state["your_turn"]
                assert state["actions"]
                offer = next((a for a in state["actions"] if a["kind"] in ("check", "call")), state["actions"][0])
                res = client.post(
                    f"/api/v1/hunl/sessions/{sid}/actions",
                    json={"action_index": offer["index"], "expected_revision": state["revision"]},
                )
                res.raise_for_status()
                state = res.json()["state"]
            print(f"hand={hand} bb={state['cumulative_bb']:.2f} bb/100={state['bb_per_100']:.2f} "
                  f"showdown={state['hand_result']['showdown']}")
            if hand < args.hands:
                res = client.post(
                    f"/api/v1/hunl/sessions/{sid}/next-hand",
                    json={"expected_revision": state["revision"]},
                )
                res.raise_for_status()
                state = res.json()["state"]
        deleted = client.delete(f"/api/v1/hunl/sessions/{sid}")
        assert deleted.status_code == 204
        print(f"PASS: {args.hands} complete HUNL200 hands over HTTP interface")


if __name__ == "__main__":
    main()
