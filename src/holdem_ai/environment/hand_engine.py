"""One-hand HUNL interface. Corrected RLCard adapter is the default.

Scope: heads-up only; this is not a six-player tournament engine.
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
    action_history: tuple[tuple[int, int], ...]


class HandEngine:
    def __init__(
        self,
        starting_stack: int = 200,
        seed: int | None = 42,
        allow_step_back: bool = False,
        rules_mode: str = 'corrected',
        button: int = 0,
    ) -> None:
        if starting_stack < 2:
            raise ValueError('starting_stack must be at least 2 chips.')
        if button not in (0, 1):
            raise ValueError('button must be 0 or 1')
        if rules_mode not in ('corrected', 'upstream'):
            raise ValueError('rules_mode must be corrected or upstream')

        if rules_mode == 'corrected':
            from holdem_ai.environment.corrected_game import make_corrected_rlcard_env
            self.env = make_corrected_rlcard_env(
                starting_stack=starting_stack, seed=seed,
                button=button, allow_step_back=allow_step_back,
            )
        else:
            import rlcard
            self.env = rlcard.make('no-limit-holdem', config={
                'seed': seed,
                'game_num_players': 2,
                'chips_for_each': starting_stack,
                'dealer_id': button,
                'allow_step_back': allow_step_back,
            })
        self.rules_mode = rules_mode
        self.num_players = 2
        self.big_blind = int(self.env.game.big_blind)
        self._next_button = button
        self._state: Mapping[str, Any] | None = None
        self._player_id: int | None = None
        self._history: list[tuple[int, int]] = []

    def reset(self) -> tuple[Mapping[str, Any], int]:
        self._history.clear()
        # Rotate seats every hand; RLCard does not rotate a fixed dealer_id.
        self.env.game.dealer_id = self._next_button
        self._next_button = 1 - self._next_button
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
        return tuple(int(a) for a in self.state['legal_actions'])

    def is_over(self) -> bool:
        return bool(self.env.is_over())

    def step(self, action_id: int) -> tuple[Mapping[str, Any], int]:
        if self._state is None:
            raise RuntimeError('Call reset() before step().')
        if self.is_over():
            raise RuntimeError('The hand has already finished.')
        if action_id not in self.legal_actions:
            raise ValueError(f'Illegal action_id={action_id}; legal actions are {self.legal_actions}')
        actor = self.current_player
        state, player = self.env.step(int(action_id))
        self._history.append((actor, int(action_id)))
        self._state, self._player_id = state, player
        return self._state, self._player_id

    def get_result(self) -> HandResult:
        if not self.is_over():
            raise RuntimeError('The hand is not over yet.')
        payoffs = tuple(float(p) for p in self.env.get_payoffs())
        if len(payoffs) != 2 or not isclose(sum(payoffs), 0.0, rel_tol=0, abs_tol=1e-6):
            raise AssertionError(f'Invalid zero-sum payoff: {payoffs}')
        return HandResult(
            payoffs_chips=(payoffs[0], payoffs[1]),
            payoffs_bb=(payoffs[0] / self.big_blind, payoffs[1] / self.big_blind),
            action_history=tuple(self._history),
        )

    def play_hand(self, agents: Sequence[Agent], max_actions: int = 500) -> HandResult:
        if len(agents) != 2:
            raise ValueError('A HUNL hand needs exactly two agents.')
        if max_actions < 1:
            raise ValueError('max_actions must be positive.')
        state, player_id = self.reset()
        n = 0
        while not self.is_over():
            if n >= max_actions:
                raise RuntimeError('Hand exceeded max_actions; inspect engine/rules.')
            action_id = agents[player_id].select_action(state)
            state, player_id = self.step(action_id)
            n += 1
        return self.get_result()
