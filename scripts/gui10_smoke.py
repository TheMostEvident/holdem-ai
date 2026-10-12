"""Transport-free GUI-1.0 HUNL smoke run; human plays check/call by default.

Run from repo root: uv run --extra dev python scripts/gui10_smoke.py --hands 20
"""
from __future__ import annotations

import argparse
import json

from holdem_ai.agents.diagnostic_opponents import CallingStation
from holdem_ai.game_service import HUNLGameSession


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI-1.0 human-agent integration smoke")
    parser.add_argument("--hands", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    if options.hands < 1:
        parser.error("--hands must be positive")
    session = HUNLGameSession(bot_policy=CallingStation(), seed=options.seed)
    for hand in range(options.hands):
        view = session.start_hand()
        for _ in range(300):
            if view["phase"] == "complete":
                break
            legal = {action["label"]: action["index"] for action in view["actions"]}
            action = legal.get("check", legal.get("call"))
            if action is None:
                raise AssertionError("human has neither check nor call")
            view = session.play_human_action(action, expected_revision=view["revision"])
        else:
            raise RuntimeError("hand exceeded action limit")
        if sum(view["hand_result"]["payoffs_chips"]) != 0:
            raise AssertionError("non-zero-sum hand")
        print(f"hand={hand+1:3} button={view['button']} hero_bb={view['hand_result']['payoffs_bb'][0]:+8.2f} board={' '.join(view['board'])}")
    print("PASS:", json.dumps({
        "hands": view["completed_hands"],
        "cumulative_bb": view["cumulative_bb"],
        "bb_per_100": view["bb_per_100"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
