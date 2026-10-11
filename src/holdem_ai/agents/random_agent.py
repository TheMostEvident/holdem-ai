"""Random legal-action baseline, for testing the environment/agent boundary."""

import random
from typing import Any, Mapping


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: Mapping[str, Any]) -> int:
        legal_actions = tuple(state['legal_actions'])
        if not legal_actions:
            raise ValueError('There are no legal actions in this state.')
        return int(self._rng.choice(legal_actions))
