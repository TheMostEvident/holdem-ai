"""Isolated stub for table-management tests; real-engine tests are separate."""
from __future__ import annotations
from pathlib import Path
import sys
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[2] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeHandEngine:
    """Not a poker simulator. Tests table events independent of Base Phase 3."""
    def __init__(self, *, stacks, button, small_blind, big_blind, seed):
        self.initial_stacks = tuple(stacks)
        self.stacks = list(stacks)
        self.contributions = [0] * 6
        participants = [i for i, stack in enumerate(stacks) if stack > 0]
        self.button = button
        def nxt(i):
            return next((i + x) % 6 for x in range(1, 7) if (i + x) % 6 in participants)
        if len(participants) == 2:
            sb, bb = button, nxt(button)
        else:
            sb = nxt(button)
            bb = nxt(sb)
        self.seats = SimpleNamespace(small_blind=sb, big_blind=bb)
        self.current_player = participants[0]
        self._done = False
        self._actions = []
        self._result = None

    def reset(self):
        return self.observe(self.current_player), self.current_player

    def is_over(self):
        return self._done

    def observe(self, player_id):
        assert player_id == self.current_player and not self._done
        return {'legal_actions': {0: None, 1: None},
                'raw_obs': {'to_call': 0, 'hand': ['cA', 'dK'], 'pot': 0}}

    def step(self, action_id):
        self._actions.append(action_id)
        self._done = True
        self._result = SimpleNamespace(final_stacks=tuple(self.stacks),
                                       payoffs_chips=(0,) * 6,
                                       board=(), action_history=((self.current_player, action_id),),
                                       action_raise_to=(None,))

    def step_raise_to(self, amount):
        self._actions.append((5, amount))
        self.step(5)

    def get_result(self):
        assert self._done
        return self._result


class BustHandEngine(FakeHandEngine):
    def step(self, action_id):
        super().step(action_id)
        occupied = [i for i, x in enumerate(self.initial_stacks) if x]
        loser, winner = occupied[:2]
        transferred = self.initial_stacks[loser]
        out = list(self.initial_stacks)
        out[loser] = 0
        out[winner] += transferred
        payoffs = [0] * 6
        payoffs[loser] = -transferred
        payoffs[winner] = transferred
        self._result = SimpleNamespace(final_stacks=tuple(out), payoffs_chips=tuple(payoffs),
                                       board=(), action_history=((self.current_player, action_id),),
                                       action_raise_to=(None,))
