"""Temporary, engine-legal six-seat baseline policies.

These are NOT the Tournament Agent or the proposed Improved Bots.  The
Tournament Policy Adapter milestone must replace/inject those policies after
an observation/action compatibility audit.
"""
from __future__ import annotations

import random
from typing import Any, Mapping


BASELINES = ("random", "calling_station", "tight", "pot_pressure")


class TournamentBaseline:
    def __init__(self, name: str = "calling_station", *, seed: int | None = None):
        if name not in BASELINES:
            raise ValueError("unknown tournament baseline policy")
        self.name = name
        self.rng = random.Random(seed)

    def select_action(self, observation: Mapping[str, Any]) -> int:
        legal = tuple(int(i) for i in observation["legal_actions"])
        if not legal:
            raise RuntimeError("no legal tournament actions")
        if self.name == "random":
            return self.rng.choice(legal)
        if self.name == "calling_station":
            priorities = (1, 0, 2, 3, 4)
        elif self.name == "tight":
            to_call = int(observation["raw_obs"]["to_call"])
            priorities = (1, 0, 2, 3, 4) if to_call == 0 else (0, 1, 2, 3, 4)
        else:
            priorities = (3, 2, 4, 1, 0)
        return next(i for i in priorities if i in legal)
