"""Initial RLCard adapter for *one hand* of heads-up no-limit Hold'em.

Scope: HUNL 2 players, the action abstraction and blinds defined by RLCard.
Not yet a verified tournament or production poker-rules engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping, Sequence

from holdem_ai.agents.base_agent import Agent


@dataclass(frozen=True)
class HandResult:
    payoffs_chips: tuple[float, float]
    payoffs_bb: tuple[float, float]
    action_history: tuple[tuple[int, int], ...]  # (player_id, RLCard action_id)


class HandEngine:
    """RLCard HUNL wrapper with validation, observation, and payoff access."""

    def __init__(
        self,
        starting_stack: int = 200,
        seed: int | None = 42,
        allow_step_back: bool = False,
    ) -> None:
        if starting_stack < 2:
            raise ValueError('starting_stack must be at least 2 chips.')

        try:
            import rlcard
        except ImportError as exc:
            raise ImportError(
                'RLCard is required. Install the project with: python -m pip install -e ".[dev]"'
            ) from exc

        self.env = rlcard.make(
            'no-limit-holdem',
            config={
                'seed': seed,
                'game_num_players': 2,
                'chips_for_each': starting_stack,
                'allow_step_back': allow_step_back,
            },
        )
        self.num_players = 2
        self.big_blind = int(self.env.game.big_blind)
        self._state: Mapping[str, Any] | None = None
        self._player_id: int | None = None
        self._history: list[tuple[int, int]] = []

    def reset(self) -> tuple[Mapping[str, Any], int]:
        """Start a fresh hand (RLCard resets both stacks)."""
        self._history.clear()
        self._state, self._player_id = self.env.reset()
        return self._state, self._player_id

    @property
    def state(self) -> Mapping[str, Any]:
        if self._state is None:
            raise RuntimeError('Call reset() before requesting state.')
        return self._state

    @property
    def current_player(self) -> int:
        if self._player_id is None:
            raise RuntimeError('Call reset() before requesting current_player.')
        return self._player_id

    @property
    def legal_actions(self) -> tuple[int, ...]:
        if self.is_over():
            return ()
        return tuple(int(action) for action in self.state['legal_actions'])

    def is_over(self) -> bool:
        return bool(self.env.is_over())

    def step(self, action_id: int) -> tuple[Mapping[str, Any], int]:
        """Apply a legal action; reject illegal IDs instead of RLCard fallback."""
        if self._state is None:
            raise RuntimeError('Call reset() before step().')
        if self.is_over():
            raise RuntimeError('The hand has already finished.')
        if action_id not in self.legal_actions:
            raise ValueError(
                f'Illegal action_id={action_id}; legal actions are {self.legal_actions}'
            )
        actor = self.current_player
        next_state, next_id = self.env.step(int(action_id))
        self._history.append((actor, int(action_id)))
        self._state, self._player_id = next_state, next_id
        return self._state, self._player_id

    def get_result(self) -> HandResult:
        if not self.is_over():
            raise RuntimeError('The hand is not over yet.')
        chips = tuple(float(value) for value in self.env.get_payoffs())
        if len(chips) != self.num_players:
            raise AssertionError('RLCard returned the wrong number of payoffs.')
        if not isclose(sum(chips), 0.0, rel_tol=0.0, abs_tol=1e-6):
            raise AssertionError(f'The hand is not zero-sum: {chips}')
        return HandResult(
            payoffs_chips=(chips[0], chips[1]),
            payoffs_bb=(chips[0] / self.big_blind, chips[1] / self.big_blind),
            action_history=tuple(self._history),
        )

    def play_hand(self, agents: Sequence[Agent], max_actions: int = 500) -> HandResult:
        """Play exactly one hand; the game engine and agents stay separate."""
        if len(agents) != self.num_players:
            raise ValueError('A HUNL hand needs exactly two agents.')
        if max_actions < 1:
            raise ValueError('max_actions must be positive.')
        state, player_id = self.reset()
        actions_taken = 0
        while not self.is_over():
            if actions_taken >= max_actions:
                raise RuntimeError('Hand exceeded max_actions; inspect engine/rules.')
            action_id = agents[player_id].select_action(state)
            state, player_id = self.step(action_id)
            actions_taken += 1
        return self.get_result()
