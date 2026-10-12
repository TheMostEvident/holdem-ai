"""Human-only exact bet sizing over the trained HUNL200 seven-action adapter.

The neural actor still receives the same 7-action InformationState.  A human
raise is recorded using the existing observable raise_to history grammar so
future bot observations contain the public betting action and exact amount.
This does not mutate or replace the Phase-6 HUNL200Adapter implementation.
"""
from __future__ import annotations

from holdem_ai.core.schema import Terminal, Turn
from holdem_ai.environment.bet_abstraction import EngineBetAction
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter


class InteractiveHUNLAdapter(HUNL200Adapter):
    def step_human_raise_to(self, amount: int) -> Turn | Terminal:
        engine = self.engine
        if type(amount) is not int:
            raise ValueError("raise-to amount must be an integer")
        if engine.is_over() or not hasattr(self, "_observation") or self._observation is None:
            raise RuntimeError("no active decision")
        actor, street = engine.current_player, engine.street
        if self._observation.player_id != actor:
            raise RuntimeError("observation must match acting player")
        # The Phase-3 engine validates the full raise, a permitted short all-in,
        # betting-reopen rules and stack maximum BEFORE any state is committed.
        engine.step_raise_to(amount)
        self.last_engine_action = EngineBetAction("raise_to", amount)
        self._history.append(f"{street}|seat={actor}|raise_to|raise_to={amount}")
        return self._receive()

    def reset_with_stacks(self, stacks: tuple[int, int], *, seed: int | None = None) -> Turn | Terminal:
        """GUI-only hand initialization with chips carried from previous hand.

        The Phase-6 HUNL200Adapter.reset() is deliberately NOT modified, so
        training and Phase-16 evaluations still start with two 200BB stacks.
        """
        from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine

        if (not isinstance(stacks, tuple) or len(stacks) != 2
                or any(type(value) is not int or value <= 0 for value in stacks)):
            raise ValueError("continuous hands require two strictly positive integer stacks")
        if sum(stacks) != 2 * self.stack_bb * self.big_blind:
            raise ValueError("continuous match chip conservation violated")
        self._engine = MultiPlayerHandEngine(
            stacks=stacks,
            small_blind=self.small_blind, big_blind=self.big_blind,
            button=self.button, seed=seed,
        )
        self._history = []
        self.last_engine_action = None
        self._choices = None
        self._observation = None
        self._engine.reset()
        return self._receive()
