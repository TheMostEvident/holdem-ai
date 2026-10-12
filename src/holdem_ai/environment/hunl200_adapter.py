"""HUNL200 adapter: real Phase-3 hand engine -> Phase-5 turn contract.

Public actor observations are built ONLY from engine.observe(current_player).
Privileged states must be requested explicitly and never supplied to the actor.
No RLCard internals, neural-network dependency, or changes to the hand engine.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from holdem_ai.environment.bet_abstraction import DiscreteBetActions, EngineBetAction, NoLimitBetContext, abstract_bet_actions
from holdem_ai.core.schema import InformationState, Terminal, Turn

# IDs belong to the underlying Phase 3 hand engine, not the 7-way policy head.
ENGINE_FOLD = 0
ENGINE_CHECK_CALL = 1
ENGINE_ALL_IN = 4


@dataclass(frozen=True, slots=True)
class CriticState:
    """Training-only state for centralized critics; intentionally excludes future deck.

    WARNING: Both players' hole cards appear here. NEVER feed into actor/policy.
    This does not imply that the Phase-7 token encoder is complete.
    """

    player_to_act: int
    button: int
    street: str
    board: tuple[str, ...]
    hole_cards: tuple[tuple[str, ...], ...]
    stacks: tuple[int, ...]
    contributions: tuple[int, ...]
    street_bets: tuple[int, ...]
    status: tuple[str, ...]
    pot: int
    history: tuple[str, ...]
    is_terminal: bool


class HUNL200Adapter:
    """7-head interface to an actual 2-seat Phase-3 MultiPlayerHandEngine.

    A new 200BB hand is dealt on each reset; no tournament stack carry-over.
    `button` controls heads-up SB/BTN, allowing paired-seat evaluations.
    `seed` controls the engine shuffle; passing it to reset is deterministic.
    """

    def __init__(
        self, *, big_blind: int = 100, small_blind: int = 50,
        stack_bb: int = 200, button: int = 0,
        expose_training_state: bool = False,
    ) -> None:
        if any(type(v) is not int for v in (big_blind, small_blind, stack_bb, button)):
            raise TypeError("blinds, stack_bb and button must be integers")
        if not (0 < small_blind < big_blind and stack_bb > 1 and button in (0, 1)):
            raise ValueError("require 0 < SB < BB, stack_bb > 1 and button=0/1")
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.stack_bb = stack_bb
        self.button = button
        self.expose_training_state = expose_training_state
        self._engine: Any | None = None
        self._choices: DiscreteBetActions | None = None
        self._observation: InformationState | None = None
        self._history: list[str] = []
        self.last_engine_action: EngineBetAction | None = None

    def reset(self, seed: int | None = None) -> Turn | Terminal:
        # Lazy import keeps schema-only unit tests independent of the engine.
        from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine

        self._engine = MultiPlayerHandEngine(
            num_players=2,
            starting_stack=self.stack_bb * self.big_blind,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            button=self.button,
            seed=seed,
        )
        self._history = []
        self.last_engine_action = None
        self._choices = None
        self._observation = None
        self._engine.reset()
        return self._receive()

    def _receive(self) -> Turn | Terminal:
        engine = self._require_engine()
        if engine.is_over():
            result = engine.get_result()
            if len(result.payoffs_chips) != 2:
                raise AssertionError("expected heads-up rewards")
            if sum(result.payoffs_chips) != 0:
                raise AssertionError("zero-sum chip conservation violated")
            rewards = tuple(float(x) / self.big_blind for x in result.payoffs_chips)
            if not math.isclose(sum(rewards), 0.0, abs_tol=1e-9):
                raise AssertionError("invalid BB reward conservation")
            self._choices = None
            self._observation = None
            return Terminal(rewards)

        player = engine.current_player
        raw = engine.observe(player)["raw_obs"]  # permitted player-view API only
        legal_engine = set(engine.legal_actions)
        committed = int(raw["street_bets"][player])
        to_call = int(raw["to_call"])
        stack = int(raw["stacks"][player])
        can_raise = ENGINE_ALL_IN in legal_engine
        ctx = NoLimitBetContext(
            pot=int(raw["pot"]), committed=committed, to_call=to_call,
            stack=stack,
            min_raise_to=int(raw["min_raise_to"]) if can_raise else None,
            max_raise_to=committed + stack if can_raise else None,
            can_fold=ENGINE_FOLD in legal_engine,
            can_check=ENGINE_CHECK_CALL in legal_engine and to_call == 0,
            can_call=ENGINE_CHECK_CALL in legal_engine and to_call > 0,
            can_raise=can_raise,
        )
        choices = abstract_bet_actions(ctx)
        self._choices = choices
        opp = 1 - player
        street_index = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}[raw["street"]]
        bb = float(self.big_blind)
        # Fixed shape; inputs are observable and chip-normalized.
        features = (
            float(player), float(player == self.button), float(street_index),
            int(raw["pot"]) / bb, to_call / bb,
            committed / bb, stack / bb,
            int(raw["stacks"][opp]) / bb,
            int(raw["street_bets"][opp]) / bb,
            int(raw["total_contributions"][player]) / bb,
            int(raw["total_contributions"][opp]) / bb,
            int(raw["min_raise_to"]) / bb,
        )
        fields = (
            "button", "small_blind_seat", "big_blind_seat", "stacks",
            "street_bets", "total_contributions", "status", "pot",
            "to_call", "min_raise_to",
        )
        public_state = tuple((key, json.dumps(raw[key], separators=(",", ":"))) for key in fields)
        obs = InformationState(
            player_id=player, features=features, legal_action_mask=choices.mask,
            private_cards=tuple(raw["hand"]),
            public_cards=tuple(raw["public_cards"]),
            stage=str(raw["street"]), public_state=public_state,
            history=tuple(self._history),
        )
        self._observation = obs
        return Turn(obs)

    def step(self, action_index: int) -> Turn | Terminal:
        engine = self._require_engine()
        if engine.is_over() or self._observation is None or self._choices is None:
            raise RuntimeError("reset a non-terminal hand before step")
        self._observation.assert_legal(action_index)
        action = self._choices.resolve(action_index)
        player, street = engine.current_player, engine.street
        if action.kind == "fold":
            engine.step(ENGINE_FOLD)
        elif action.kind in ("check", "call"):
            engine.step(ENGINE_CHECK_CALL)
        elif action.kind == "all_in":
            engine.step(ENGINE_ALL_IN)
        elif action.kind == "raise_to":
            assert action.raise_to is not None
            engine.step_raise_to(action.raise_to)
        else:
            raise AssertionError("unrecognized action")
        self.last_engine_action = action
        amount = str(action.raise_to) if action.raise_to is not None else "-"
        self._history.append(f"{street}|seat={player}|{action.kind}|raise_to={amount}")
        return self._receive()

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("reset must be called first")
        return self._engine

    @property
    def engine(self) -> Any:
        """INTERNAL: full mutable engine. Never give this object to an Actor."""
        return self._require_engine()

    def training_state(self) -> CriticState:
        """Privileged state for training-only data collection, opt-in required."""
        if not self.expose_training_state:
            raise PermissionError("pass expose_training_state=True only in the training collector")
        engine = self._require_engine()
        return CriticState(
            player_to_act=engine.current_player,
            button=engine.button,
            street=engine.street,
            board=tuple(engine.board),
            hole_cards=tuple(tuple(h) for h in engine.hole_cards),
            stacks=tuple(engine.stacks),
            contributions=tuple(engine.contributions),
            street_bets=tuple(engine.street_bets),
            status=tuple(engine.status),
            pot=sum(engine.contributions),
            history=tuple(self._history),
            is_terminal=engine.is_over(),
        )
