"""Run from repository root: uv run python scripts/phase5_rlcard_smoke.py"""
from __future__ import annotations

import rlcard
from holdem_agent_phase5 import RLCardTurnDriver, RandomLegalPolicy, run_self_play


def main() -> None:
    env = rlcard.make("no-limit-holdem", config={"seed": 19})
    driver = RLCardTurnDriver(env, tuple(f"action_{i}" for i in range(env.num_actions)))
    baseline = {i: RandomLegalPolicy() for i in range(env.num_players)}
    result = run_self_play(driver, baseline, seed=19, max_decisions=500)
    print(f"players={len(result.rewards)} decisions={result.num_decisions} rewards={result.rewards}")
    for seat in range(len(result.rewards)):
        print(f"seat={seat} actions={len(result.for_seat(seat))}")


if __name__ == "__main__":
    main()
