# Full Python code — Volume 8/28

Part of [integrated report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Immutable source commit: `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`.

## src/holdem_ai/environment/hunl200_adapter.py

Blob `e786c0a6a60f56db7fa605512c45b5bdb833952c` · 8695 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/hunl200_adapter.py" sha="e786c0a6a60f56db7fa605512c45b5bdb833952c" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/hunl200_adapter.py" -->

## src/holdem_ai/environment/multi_player_engine.py

Blob `70253ad3fea77a03f11b7f200905c91ea5bb9931` · 17555 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/multi_player_engine.py" sha="70253ad3fea77a03f11b7f200905c91ea5bb9931" -->

````````python
"""Standalone 2-6 player no-limit Hold'em *hand* environment.

Keeps RLCard's five integer action IDs, but does not reuse its unsupported
multiway betting engine. Uses the project's exact reference showdown evaluator
and side-pot settlement. A hand resets stacks; tournaments come in Phase 4.

All amounts are integral chips, and raise_to always means the total amount
invested *on the current street*, not the additional chips to push.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from holdem_ai.environment.hand_evaluator import RANKS, SUITS, rank_showdown
from holdem_ai.environment.multi_player_seats import multi_player_seats, next_occupied
from holdem_ai.environment.pot_manager import build_side_pots, settle_pots

FOLD = 0
CHECK_CALL = 1
RAISE_HALF_POT = 2
RAISE_POT = 3
ALL_IN = 4


@dataclass(frozen=True)
class MultiPlayerHandResult:
    payoffs_chips: tuple[int, ...]
    payoffs_bb: tuple[float, ...]
    final_stacks: tuple[int, ...]
    contributions: tuple[int, ...]
    awards: tuple[int, ...]
    board: tuple[str, ...]
    folded: tuple[bool, ...]
    button: int
    action_history: tuple[tuple[int, int], ...]
    action_raise_to: tuple[int | None, ...]


class MultiPlayerHandEngine:
    """Play one hand using fixed physical seats and a moving-button model.

    `stacks` may contain zeroes for eliminated seats, provided >=2 remain.
    `deck` is only for deterministic tests; default RNG shuffles all 52 cards.
    Does not provide step_back or tournament-level chip persistence.
    """

    def __init__(self, stacks: Sequence[int] | None = None, *,
                 num_players: int = 6, starting_stack: int = 200,
                 small_blind: int = 1, big_blind: int = 2,
                 button: int = 0, seed: int | None = 42,
                 deck: Sequence[str] | None = None):
        if stacks is None:
            if type(num_players) is not int or not 2 <= num_players <= 6:
                raise ValueError('num_players must be 2..6')
            if type(starting_stack) is not int or starting_stack <= 0:
                raise ValueError('starting_stack must be positive integer chips')
            stacks = [starting_stack] * num_players
        self.initial_stacks = tuple(stacks)
        if not (type(small_blind) is type(big_blind) is int and 0 < small_blind < big_blind):
            raise ValueError('require integer 0 < small_blind < big_blind')
        self.seats = multi_player_seats(self.initial_stacks, button)
        self.num_players = len(self.initial_stacks)
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.button = button
        self._rng = random.Random(seed)
        self._deck_override = tuple(deck) if deck is not None else None
        if self._deck_override is not None:
            canonical = {s + r for s in SUITS for r in RANKS}
            if len(self._deck_override) != 52 or set(self._deck_override) != canonical:
                raise ValueError('test deck must be a permutation of the 52 canonical cards')
        self._started = False
        self._done = False

    def _draw(self) -> str:
        card = self._deck[self._deck_idx]
        self._deck_idx += 1
        return card

    def _next_in(self, from_seat: int, choices: set[int]) -> int:
        return next_occupied(from_seat, tuple(choices), self.num_players)

    def _active(self) -> set[int]:
        return {i for i in range(self.num_players) if self.status[i] == 'active'}

    def _contenders(self) -> list[int]:
        return [i for i, st in enumerate(self.status) if st in ('active', 'allin')]

    def _put_chips(self, player: int, chips: int) -> None:
        if chips < 0 or chips > self.stacks[player]:
            raise AssertionError('invalid chip movement')
        self.stacks[player] -= chips
        self.street_bets[player] += chips
        self.contributions[player] += chips
        if self.stacks[player] == 0 and self.status[player] == 'active':
            self.status[player] = 'allin'

    def reset(self) -> tuple[Mapping[str, Any] | None, int]:
        self.stacks = list(self.initial_stacks)
        self.contributions = [0] * self.num_players
        self.street_bets = [0] * self.num_players
        self.status = ['active' if s > 0 else 'out' for s in self.stacks]
        self.board: list[str] = []
        self.burned: list[str] = []
        self.hole_cards: list[tuple[str, ...]] = [tuple() for _ in self.stacks]
        self.action_history: list[tuple[int, int]] = []
        self.action_raise_to: list[int | None] = []
        self.street = 'preflop'
        self.last_full_raise = self.big_blind
        self.last_acted_high: dict[int, int] = {}
        self.pending: set[int] = set()
        self._done = False
        self._result: MultiPlayerHandResult | None = None
        self._deck = list(self._deck_override) if self._deck_override else [s + r for s in SUITS for r in RANKS]
        if self._deck_override is None:
            self._rng.shuffle(self._deck)
        self._deck_idx = 0

        # Deal one card clockwise per round, beginning with seat left of BTN.
        occupied = set(self.seats.occupied)
        first = self._next_in(self.button, occupied)
        seat_order = [first]
        while len(seat_order) < len(occupied):
            seat_order.append(self._next_in(seat_order[-1], occupied))
        hole_temp = {i: [] for i in occupied}
        for _ in range(2):
            for i in seat_order:
                hole_temp[i].append(self._draw())
        for i in occupied:
            self.hole_cards[i] = tuple(hole_temp[i])

        # Forced blinds, including short-stack all-in blinds.
        for seat, blind in ((self.seats.small_blind, self.small_blind),
                            (self.seats.big_blind, self.big_blind)):
            self._put_chips(seat, min(blind, self.stacks[seat]))
        self._started = True
        self._begin_action(self.seats.preflop_first, preflop=True)
        self._progress_if_closed()
        return (None, -1) if self._done else (self.state, self.current_player)

    def _begin_action(self, first: int, *, preflop: bool = False) -> None:
        active = self._active()
        # A lone active player only acts if owing money to an all-in contender.
        if len(active) == 1:
            i = next(iter(active))
            self.pending = {i} if self._to_call(i) > 0 else set()
        else:
            self.pending = set(active)
        self.current_player = self._next_in((first - 1) % self.num_players, self.pending) if self.pending else -1
        self.last_acted_high = {}
        self.last_full_raise = self.big_blind

    def _current_high(self) -> int:
        high = max(self.street_bets)
        # A short all-in big blind does not reduce the nominal preflop
        # bring-in for other players who can still bet against each other.
        if self.street == 'preflop' and len(self._active()) >= 2:
            return max(high, self.big_blind)
        return high

    def _to_call(self, i: int) -> int:
        return self._current_high() - self.street_bets[i]

    def _raise_permitted(self, i: int) -> bool:
        # No betting into players who are already all-in/folded.
        if len(self._active()) < 2:
            return False
        if self.stacks[i] <= self._to_call(i):
            return False
        if i not in self.last_acted_high:
            return True
        # Multiple short all-ins cumulatively reopen action when >= full raise.
        return self._current_high() - self.last_acted_high[i] >= self.last_full_raise

    def min_raise_to(self) -> int:
        if self._done or not self.pending:
            raise RuntimeError('no active betting decision')
        high = self._current_high()
        return high + self.last_full_raise if high > 0 else self.big_blind

    def _pot_raise_target(self, i: int, factor: float) -> int:
        call = self._to_call(i)
        pot_after_call = sum(self.contributions) + call
        return self.street_bets[i] + call + int(pot_after_call * factor)

    @property
    def legal_actions(self) -> tuple[int, ...]:
        if not self._started or self._done or self.current_player < 0:
            return ()
        i = self.current_player
        if i not in self.pending:
            raise AssertionError('current_player is not pending')
        actions = [FOLD, CHECK_CALL]
        if self._raise_permitted(i):
            maximum = self.street_bets[i] + self.stacks[i]
            if maximum > self._current_high():
                actions.append(ALL_IN)
            minimum = self.min_raise_to()
            if minimum <= self._pot_raise_target(i, 0.5) <= maximum:
                actions.append(RAISE_HALF_POT)
            if minimum <= self._pot_raise_target(i, 1.0) <= maximum:
                actions.append(RAISE_POT)
        return tuple(sorted(actions))

    def observe(self, player_id: int) -> Mapping[str, Any]:
        if not self._started:
            raise RuntimeError('reset before observe')
        if player_id != self.current_player or self._done:
            raise ValueError('only the acting player may obtain an observation')
        i = player_id
        return {
            'legal_actions': {a: None for a in self.legal_actions},
            'raw_obs': {
                'hand': list(self.hole_cards[i]),
                'public_cards': list(self.board),
                'street': self.street,
                'button': self.button,
                'small_blind_seat': self.seats.small_blind,
                'big_blind_seat': self.seats.big_blind,
                'player_id': i,
                'stacks': tuple(self.stacks),
                'street_bets': tuple(self.street_bets),
                'total_contributions': tuple(self.contributions),
                'status': tuple(self.status),
                'pot': sum(self.contributions),
                'to_call': self._to_call(i),
                'min_raise_to': self.min_raise_to(),
                'action_history': tuple(self.action_history),
                'action_raise_to': tuple(self.action_raise_to),
            },
        }

    @property
    def state(self) -> Mapping[str, Any]:
        if not self._started or self._done:
            raise RuntimeError('no active decision')
        return self.observe(self.current_player)

    def step(self, action_id: int) -> tuple[Mapping[str, Any] | None, int]:
        if action_id not in self.legal_actions:
            raise ValueError(f'illegal action_id={action_id}; legal={self.legal_actions}')
        if action_id == FOLD:
            self._apply_action(action_id, None)
        elif action_id == CHECK_CALL:
            self._apply_action(action_id, None)
        elif action_id == ALL_IN:
            i = self.current_player
            self._apply_action(action_id, self.street_bets[i] + self.stacks[i])
        else:
            factor = 0.5 if action_id == RAISE_HALF_POT else 1.0
            self._apply_action(action_id, self._pot_raise_target(self.current_player, factor))
        return (None, -1) if self._done else (self.state, self.current_player)

    def step_raise_to(self, amount: int) -> tuple[Mapping[str, Any] | None, int]:
        """Exact raise-to action for future UI/full bet sizing. History ID = 5.

        Unlike the five abstract action IDs, this accepts *any* legal full raise
        or a short all-in. It never silently clamps an illegal bet.
        """
        if type(amount) is not int:
            raise ValueError('raise-to amount must be integer chips')
        self._apply_action(5, amount)
        return (None, -1) if self._done else (self.state, self.current_player)

    def _apply_action(self, action_id: int, raise_to: int | None) -> None:
        if not self._started or self._done or self.current_player < 0:
            raise RuntimeError('no action can be taken')
        i = self.current_player
        high = self._current_high()
        if action_id == 5:
            if not self._raise_permitted(i) or raise_to is None:
                raise ValueError('raise is not permitted')
            maximum = self.street_bets[i] + self.stacks[i]
            if not high < raise_to <= maximum:
                raise ValueError('raise-to target outside valid range')
            if raise_to < self.min_raise_to() and raise_to != maximum:
                raise ValueError('short raise is legal only if all-in')
        self.action_history.append((i, action_id))
        self.action_raise_to.append(raise_to)
        self.pending.discard(i)
        if action_id == FOLD:
            self.status[i] = 'folded'
        elif action_id == CHECK_CALL:
            self._put_chips(i, min(self._to_call(i), self.stacks[i]))
            self.last_acted_high[i] = self._current_high()
        else:
            assert raise_to is not None
            difference = raise_to - high
            self._put_chips(i, raise_to - self.street_bets[i])
            self.last_acted_high[i] = self._current_high()
            if difference >= self.last_full_raise:
                self.last_full_raise = difference
                self.pending = self._active() - {i}
            else:
                # Partial raise leaves previously acted players' rights unchanged.
                self.pending.update(j for j in self._active() if j != i and self.street_bets[j] < raise_to)

        if len(self._active()) == 1:
            last = next(iter(self._active()))
            if self._to_call(last) == 0:
                self.pending.clear()
            else:
                self.pending = {last} if last != i else set()
        else:
            self.pending &= self._active()
        self._progress_if_closed(actor=i)

    def _deal_next_street(self) -> None:
        if self.street == 'preflop':
            self._burn()
            self.board.extend(self._draw() for _ in range(3))
            self.street = 'flop'
        elif self.street == 'flop':
            self._burn()
            self.board.append(self._draw())
            self.street = 'turn'
        elif self.street == 'turn':
            self._burn()
            self.board.append(self._draw())
            self.street = 'river'
        else:
            raise AssertionError('cannot advance from river')
        self.street_bets = [0] * self.num_players
        self._begin_action(self.seats.postflop_first)

    def _burn(self) -> None:
        self.burned.append(self._draw())

    def _progress_if_closed(self, actor: int | None = None) -> None:
        if len(self._contenders()) == 1:
            self._settle()
            return
        if self.pending:
            if actor is not None:
                self.current_player = self._next_in(actor, self.pending)
            return
        # Betting round is closed. When all players are all-in, automatically
        # run out the remaining board, with burns, without spurious decisions.
        while not self.pending and not self._done:
            if self.street == 'river':
                self._settle()
                return
            self._deal_next_street()

    def _settle(self) -> None:
        folded = tuple(s in ('folded', 'out') for s in self.status)
        breakdown = build_side_pots(self.contributions, folded)
        contenders = self._contenders()
        strengths: list[tuple[int, ...] | None] = [None] * self.num_players
        if len(contenders) == 1:
            strengths[contenders[0]] = (0,)
        else:
            if len(self.board) != 5:
                raise AssertionError('showdown before board complete')
            for i in contenders:
                strengths[i] = rank_showdown(self.hole_cards[i], self.board)
        awards = settle_pots(breakdown, strengths, button=self.button)
        payoffs = tuple(awards[i] - self.contributions[i] for i in range(self.num_players))
        final_stacks = tuple(self.stacks[i] + awards[i] for i in range(self.num_players))
        if sum(payoffs) != 0 or sum(final_stacks) != sum(self.initial_stacks):
            raise AssertionError('chip conservation violated')
        self._result = MultiPlayerHandResult(
            payoffs_chips=payoffs,
            payoffs_bb=tuple(p / self.big_blind for p in payoffs),
            final_stacks=final_stacks,
            contributions=tuple(self.contributions), awards=awards,
            board=tuple(self.board), folded=folded,
            button=self.button, action_history=tuple(self.action_history),
            action_raise_to=tuple(self.action_raise_to),
        )
        self._done = True
        self.current_player = -1
        self.pending.clear()

    def is_over(self) -> bool:
        return self._done

    def get_result(self) -> MultiPlayerHandResult:
        if not self._done or self._result is None:
            raise RuntimeError('hand must end before result')
        return self._result

    def play_hand(self, agents: Sequence[Any], max_actions: int = 1000) -> MultiPlayerHandResult:
        if len(agents) != self.num_players:
            raise ValueError('one agent per physical seat is required')
        if max_actions <= 0:
            raise ValueError('max_actions must be positive')
        state, actor = self.reset()
        steps = 0
        while not self.is_over():
            if steps >= max_actions:
                raise RuntimeError('max_actions exceeded; investigate betting loop')
            selected = agents[actor].select_action(state)
            state, actor = self.step(int(selected))
            steps += 1
        return self.get_result()

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/multi_player_engine.py" -->

## src/holdem_ai/environment/multi_player_seats.py

Blob `867ef2000e2b1cc41d00fcefe048cc3cf40ad82e` · 2113 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/multi_player_seats.py" sha="867ef2000e2b1cc41d00fcefe048cc3cf40ad82e" -->

````````python
"""Moving-button seat and action order for 2-6 seat Texas Hold'em hands.

Seat indexes are fixed across hands; stack == 0 denotes an empty/busted seat.
Heads-up is a special case: button posts SB and acts first preflop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MultiPlayerSeats:
    button: int
    small_blind: int
    big_blind: int
    preflop_first: int
    postflop_first: int
    occupied: tuple[int, ...]


def next_occupied(seat: int, occupied: Sequence[int], num_seats: int) -> int:
    occupied_set = set(occupied)
    for offset in range(1, num_seats + 1):
        candidate = (seat + offset) % num_seats
        if candidate in occupied_set:
            return candidate
    raise ValueError('no occupied seat')


def multi_player_seats(stacks: Sequence[int], button: int) -> MultiPlayerSeats:
    n = len(stacks)
    if n < 2 or n > 6 or any(type(x) is not int or x < 0 for x in stacks):
        raise ValueError('require 2-6 nonnegative integer stacks')
    occupied = tuple(i for i, x in enumerate(stacks) if x > 0)
    if len(occupied) < 2:
        raise ValueError('a hand requires at least two occupied seats')
    if type(button) is not int or button not in occupied:
        raise ValueError('button must be an occupied seat')
    if len(occupied) == 2:
        sb = button
        bb = next_occupied(button, occupied, n)
        pre = sb
        post = bb
    else:
        sb = next_occupied(button, occupied, n)
        bb = next_occupied(sb, occupied, n)
        pre = next_occupied(bb, occupied, n)
        post = sb
    return MultiPlayerSeats(button, sb, bb, pre, post, occupied)


def rotate_button(stacks: Sequence[int], button: int) -> int:
    """Moving-button model. Next occupied seat clockwise from prior button.

    Explicitly not a tournament dead-button/blind-placement policy.
    """
    occupied = tuple(i for i, chips in enumerate(stacks) if chips > 0)
    if not occupied:
        raise ValueError('no occupied players')
    return next_occupied(button, occupied, len(stacks))

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/multi_player_seats.py" -->

## src/holdem_ai/environment/observation.py

Blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` · 0 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/observation.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/observation.py" -->

## src/holdem_ai/environment/player_manager.py

Blob `50d9a973ae3bd97958f4f04c745ec4cfd6125965` · 2666 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/player_manager.py" sha="50d9a973ae3bd97958f4f04c745ec4cfd6125965" -->

````````python
"""Persistent chip stacks and elimination ranks across tournament hands.

Simultaneous busts are ordered by stacks immediately before that hand.
Equal pre-hand stacks share a placement (competition ranking).
"""
from __future__ import annotations

from typing import Sequence


class PlayerManager:
    def __init__(self, initial_stacks: Sequence[int]):
        initial = tuple(initial_stacks)
        if not 2 <= len(initial) <= 6 or any(type(x) is not int or x <= 0 for x in initial):
            raise ValueError('require 2-6 positive integer starting stacks')
        self.initial_stacks = initial
        self.total_chips = sum(initial)
        self.stacks = initial
        self.ranks: list[int | None] = [None] * len(initial)
        self.eliminated_at: list[int | None] = [None] * len(initial)

    @property
    def alive(self) -> tuple[int, ...]:
        return tuple(i for i, chips in enumerate(self.stacks) if chips > 0)

    @property
    def winner(self) -> int | None:
        alive = self.alive
        return alive[0] if len(alive) == 1 else None

    def update(self, final_stacks: Sequence[int], hand_number: int) -> tuple[int, ...]:
        """Commit a completed hand; return busted seats (ascending IDs)."""
        after = tuple(final_stacks)
        if type(hand_number) is not int or hand_number <= 0:
            raise ValueError('hand number must be positive')
        if len(after) != len(self.stacks) or any(type(x) is not int or x < 0 for x in after):
            raise ValueError('invalid final stacks')
        if sum(after) != self.total_chips:
            raise ValueError('chips cannot enter or leave the tournament')
        if any(old == 0 and new != 0 for old, new in zip(self.stacks, after)):
            raise ValueError('busted players cannot be revived')
        if self.winner is not None:
            raise ValueError('tournament already finished')

        busts = [i for i, (old, new) in enumerate(zip(self.stacks, after)) if old > 0 and new == 0]
        # The lowest pre-hand stack finishes worse in a same-hand knockout.
        # Players with equal pre-hand stacks share the occupied prize positions.
        cursor = len(self.alive)
        for start_stack in sorted({self.stacks[i] for i in busts}):
            tied = [i for i in busts if self.stacks[i] == start_stack]
            shared_rank = cursor - len(tied) + 1
            for i in tied:
                self.ranks[i] = shared_rank
                self.eliminated_at[i] = hand_number
            cursor -= len(tied)
        self.stacks = after
        if self.winner is not None:
            self.ranks[self.winner] = 1
        return tuple(sorted(busts))

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/player_manager.py" -->

## src/holdem_ai/environment/pot_manager.py

Blob `06e907b563bedb422d7603d4d24a8d77513a3eaa` · 3297 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/pot_manager.py" sha="06e907b563bedb422d7603d4d24a8d77513a3eaa" -->

````````python
"""Reference side-pot construction and settlement with chip conservation.

Works with 2+ players; prepared for 6-max tests. Winner ranks must be supplied by
an independently verified hand evaluator (larger rank means stronger hand).
No rake. Integer-chip odd chips go clockwise from the seat left of the button.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Pot:
    amount: int
    eligible: tuple[int, ...]


@dataclass(frozen=True)
class PotBreakdown:
    pots: tuple[Pot, ...]
    uncalled_returns: tuple[int, ...]


def build_side_pots(contributions: Sequence[int], folded: Sequence[bool]) -> PotBreakdown:
    if len(contributions) < 2 or len(folded) != len(contributions):
        raise ValueError('need two or more players and matching fold flags')
    if any(not isinstance(v, int) or v < 0 for v in contributions):
        raise ValueError('contributions must be nonnegative integer chips')
    if any(not isinstance(v, bool) for v in folded):
        raise ValueError('folded flags must be bool')
    if all(folded):
        raise ValueError('at least one player must remain eligible')

    levels = sorted(set(v for v in contributions if v > 0))
    previous = 0
    pots: list[Pot] = []
    returned = [0] * len(contributions)
    for upper in levels:
        participants = [i for i, chips in enumerate(contributions) if chips >= upper]
        layer = (upper - previous) * len(participants)
        eligible = tuple(i for i in participants if not folded[i])
        if len(participants) == 1:
            # An unmatched wager belongs to its contributor, even if they folded.
            returned[participants[0]] += layer
        elif not eligible:
            raise ValueError('pot has no eligible winner: invalid hand state')
        else:
            pots.append(Pot(layer, eligible))
        previous = upper

    assert sum(p.amount for p in pots) + sum(returned) == sum(contributions)
    return PotBreakdown(tuple(pots), tuple(returned))


def settle_pots(
    breakdown: PotBreakdown,
    strengths: Sequence[object | None],
    button: int,
) -> tuple[int, ...]:
    """Distribute pots by comparable showdown strengths; equal strengths split.

    Folded players should have strength None. Strength objects must be ordered
    so higher values are better (e.g. tuples of ranking components).
    """
    n = len(breakdown.uncalled_returns)
    if len(strengths) != n or not 0 <= button < n:
        raise ValueError('player count / button mismatch')
    awards = list(breakdown.uncalled_returns)
    for pot in breakdown.pots:
        eligible = [i for i in pot.eligible if strengths[i] is not None]
        if not eligible:
            raise ValueError('eligible pot player must have showdown strength')
        best = max(strengths[i] for i in eligible)
        winners = [i for i in eligible if strengths[i] == best]
        share, odd = divmod(pot.amount, len(winners))
        for winner in winners:
            awards[winner] += share
        # First odd chip goes to winner seated immediately left of button.
        clockwise = sorted(winners, key=lambda i: (i - button - 1) % n)
        for winner in clockwise[:odd]:
            awards[winner] += 1
    return tuple(awards)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/pot_manager.py" -->

## src/holdem_ai/environment/rewards.py

Blob `619a7aa20e520d22366c7230b561d02571123d8a` · 1789 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/rewards.py" sha="619a7aa20e520d22366c7230b561d02571123d8a" -->

````````python
"""Hand-level chip rewards and terminal tournament placement rewards."""
from __future__ import annotations

from typing import Sequence


def chip_rewards(payoffs_chips: Sequence[int], big_blind: int) -> tuple[float, ...]:
    if type(big_blind) is not int or big_blind <= 0:
        raise ValueError('BB must be positive integer')
    return tuple(x / big_blind for x in payoffs_chips)


def placement_rewards(
    ranks: Sequence[int | None], payouts: Sequence[float] | None = None
) -> tuple[float, ...]:
    """Terminal reward with ties splitting prizes for occupied rank slots.

    Example: six entrants, two tied at rank 5 share prizes for 5th and 6th.
    Default payout scheme awards 1 to the champion, 0 to everybody else.
    """
    n = len(ranks)
    if n < 2 or any(type(r) is not int or not 1 <= r <= n for r in ranks):
        raise ValueError('all ranks must be final integers within 1..N')
    if payouts is None:
        payouts = (1.0,) + (0.0,) * (n - 1)
    if len(payouts) != n or any(not isinstance(v, (float, int)) or isinstance(v, bool) or v < 0 for v in payouts):
        raise ValueError('payouts must contain N nonnegative numbers')
    # A valid tied rank must skip as many following places as seats in the tie.
    groups: dict[int, list[int]] = {}
    for i, rank in enumerate(ranks):
        groups.setdefault(rank, []).append(i)
    next_rank = 1
    for rank, seats in sorted(groups.items()):
        if rank != next_rank:
            raise ValueError('rank sequence has invalid tie/placement gaps')
        next_rank += len(seats)
    out = [0.0] * n
    for rank, seats in groups.items():
        shared = sum(payouts[rank - 1:rank - 1 + len(seats)]) / len(seats)
        for seat in seats:
            out[seat] = float(shared)
    return tuple(out)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/rewards.py" -->

## src/holdem_ai/environment/rlcard_adapter.py

Blob `cbec3206783ce0cf4e68b3da8decc00f6a6a337a` · 7021 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/rlcard_adapter.py" sha="cbec3206783ce0cf4e68b3da8decc00f6a6a337a" -->

````````python
"""Conservative RLCard Env adapter. No import-time dependency on rlcard/numpy."""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from holdem_ai.core.schema import InformationState, Terminal, Turn

# These are public fields only. NEVER pass raw_obs or engine state to a policy.
_PUBLIC_FIELDS = ("all_chips", "stakes", "pot", "total_chips", "current_player")


def _float_features(obj: Any) -> tuple[float, ...]:
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        value = float(obj)
        if not math.isfinite(value):
            raise ValueError("observation must be finite")
        return (value,)
    if isinstance(obj, (tuple, list)):
        return tuple(value for x in obj for value in _float_features(x))
    raise TypeError("RLCard obs must contain numeric feature data")


def _json_safe(value: Any) -> str:
    """Encode public JSON fields, including nested NumPy arrays/scalars.

    RLCard can return np.int64 values *inside* lists and dictionaries.  A
    top-level .tolist() call does not reach those nested values.  The encoder's
    ``default`` callback handles them recursively without importing NumPy.
    Unknown objects remain rejected rather than being stringified.
    """

    def convert_numpy(obj: Any) -> Any:
        tolist = getattr(obj, "tolist", None)
        if callable(tolist):
            return tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=convert_numpy,
    )


def observe_rlcard_state(
    state: Mapping[str, Any],
    player_id: int,
    num_actions: int,
    *,
    history: Sequence[str] = (),
) -> InformationState:
    """Build player POV only from RLCard's get_state(player_id) projection.

    Required: ``obs`` and legal_actions (mapping of action_id => metadata).
    Public and own-card fields are separately whitelisted from raw_obs.
    Missing history means this is NOT enough for perfect-recall CFR.
    """
    if not isinstance(state, Mapping):
        raise TypeError("RLCard state must be a mapping")
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    legal = state.get("legal_actions")
    if not isinstance(legal, Mapping):
        raise TypeError("expected legal_actions to be a mapping of integer action IDs")
    mask = [False] * num_actions
    for action_id in legal.keys():
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise TypeError("RLCard legal action keys must be integer action IDs")
        if not 0 <= action_id < num_actions:
            raise ValueError(f"action_id {action_id} outside action space")
        mask[action_id] = True
    raw = state.get("raw_obs", {})
    if not isinstance(raw, Mapping):
        raw = {}
    hand = raw.get("hand", ())
    board = raw.get("public_cards", ())
    if not isinstance(hand, (tuple, list)) or not isinstance(board, (tuple, list)):
        raise TypeError("own hand / community cards must be lists of cards")
    public_values = tuple(
        (name, _json_safe(raw[name])) for name in _PUBLIC_FIELDS if name in raw
    )
    return InformationState(
        player_id=player_id,
        features=_float_features(state["obs"]),
        legal_action_mask=tuple(mask),
        private_cards=tuple(str(card) for card in hand),
        public_cards=tuple(str(card) for card in board),
        stage=str(raw.get("stage", "unknown")),
        public_state=public_values,
        history=tuple(history),
    )


class RLCardTurnDriver:
    """Adapter for RLCard .reset()/.step() -> (state, player_id).

    Provide explicit canonical action_labels for the ENVIRONMENT version; do
    not assume one no-limit action vocabulary works for arbitrary bet sizing.
    A custom 2-6 player/tournament engine should implement TurnBasedEnvironment
    directly, reusing observe_rlcard_state if it exposes the RLCard state API.
    """

    def __init__(
        self,
        env: Any,
        action_labels: Sequence[str],
        *,
        history_fn: Callable[[Any, int, Mapping[str, Any]], Sequence[str]] | None = None,
        terminal_rewards_fn: Callable[[Any], Sequence[float]] | None = None,
    ) -> None:
        if not action_labels or len(set(action_labels)) != len(action_labels):
            raise ValueError("action_labels must be nonempty and unique")
        self.env = env
        self.action_labels = tuple(action_labels)
        self.history_fn = history_fn
        self.terminal_rewards_fn = terminal_rewards_fn
        self._current: Turn | Terminal | None = None

    def reset(self, seed: int | None = None) -> Turn | Terminal:
        if seed is not None:
            seeder = getattr(self.env, "seed", None)
            if not callable(seeder):
                raise TypeError("environment cannot be seeded; do not silently claim reproducibility")
            seeder(seed)
        result = self.env.reset()
        return self._receive(result)

    def step(self, action_index: int) -> Turn | Terminal:
        if not isinstance(self._current, Turn):
            raise RuntimeError("reset before step and never step after terminal")
        self._current.observation.assert_legal(action_index)
        result = self.env.step(action_index)
        return self._receive(result)

    def _is_over(self) -> bool:
        checker = getattr(self.env, "is_over", None)
        if callable(checker):
            return bool(checker())
        game = getattr(self.env, "game", None)
        checker = getattr(game, "is_over", None)
        if callable(checker):
            return bool(checker())
        raise TypeError("RLCard env needs is_over() or game.is_over()")

    def _receive(self, result: Any) -> Turn | Terminal:
        if self._is_over():
            if self.terminal_rewards_fn is not None:
                payoffs = self.terminal_rewards_fn(self.env)
            else:
                getter = getattr(self.env, "get_payoffs", None)
                if not callable(getter):
                    raise TypeError("terminal payoff getter is required")
                payoffs = getter()
            self._current = Terminal(tuple(float(x) for x in payoffs))
            return self._current
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("RLCard .reset/.step must return (player_state, player_id)")
        state, seat = result
        if isinstance(seat, bool) or not isinstance(seat, int) or seat < 0:
            raise TypeError("player id must be a nonnegative integer")
        history = self.history_fn(self.env, seat, state) if self.history_fn else ()
        self._current = Turn(observe_rlcard_state(state, seat, len(self.action_labels), history=history))
        return self._current

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/rlcard_adapter.py" -->

## src/holdem_ai/environment/rules_audit.py

Blob `e500dfab80e69f6899225e4b9914507987aabd36` · 1782 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/rules_audit.py" sha="e500dfab80e69f6899225e4b9914507987aabd36" -->

````````python
"""Observational, non-mutating audit of installed RLCard HUNL baseline.

Failures indicate the UPSTREAM engine differs from HUNL requirements, not that
our reference rules are wrong. No RLCard monkey-patching is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from holdem_ai.environment.seat_manager import heads_up_seats


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    expected: str
    observed: str


def audit_preflop(game: Any) -> tuple[AuditCheck, ...]:
    """Call immediately after RLCard's `env.reset()`.

    RLCard has a public `game.dealer_id`, `game.game_pointer`, player.in_chips.
    Checks against standard heads-up: BTN=SB, BB=other, BTN acts first.
    """
    seats = heads_up_seats(int(game.dealer_id))
    sb_paid = int(game.players[seats.small_blind].in_chips)
    bb_paid = int(game.players[seats.big_blind].in_chips)
    first = int(game.game_pointer)
    expected_sb = min(int(game.small_blind), int(game.init_chips[seats.small_blind]))
    expected_bb = min(int(game.big_blind), int(game.init_chips[seats.big_blind]))
    return (
        AuditCheck('button_posts_small_blind', sb_paid == expected_sb, str(expected_sb), str(sb_paid)),
        AuditCheck('opponent_posts_big_blind', bb_paid == expected_bb, str(expected_bb), str(bb_paid)),
        AuditCheck('button_acts_first_preflop', first == seats.preflop_first, str(seats.preflop_first), str(first)),
    )


def summarize(checks: tuple[AuditCheck, ...] | list[AuditCheck]) -> str:
    lines = []
    for check in checks:
        prefix = 'PASS' if check.passed else 'FAIL'
        lines.append(f'{prefix} {check.name}: expected={check.expected}, observed={check.observed}')
    return '\n'.join(lines)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/rules_audit.py" -->

## src/holdem_ai/environment/seat_manager.py

Blob `5948268e0ddfbaa07f35f1c6aa1732321cd97ef6` · 1643 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/seat_manager.py" sha="5948268e0ddfbaa07f35f1c6aa1732321cd97ef6" -->

````````python
"""Rule reference for heads-up seating and betting order.

This module intentionally does not mutate RLCard internals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadsUpSeats:
    """Button is the small blind in a two-player hand."""

    button: int
    small_blind: int
    big_blind: int
    preflop_first: int
    postflop_first: int


def heads_up_seats(button: int) -> HeadsUpSeats:
    if button not in (0, 1):
        raise ValueError('A two-player button must be player 0 or 1.')
    bb = 1 - button
    return HeadsUpSeats(
        button=button,
        small_blind=button,
        big_blind=bb,
        preflop_first=button,
        postflop_first=bb,
    )


def rotate_heads_up_button(button: int) -> int:
    """Switch the button to the other remaining seat between hands."""
    return heads_up_seats(button).big_blind


def post_blinds(stacks: tuple[int, int], button: int, sb: int, bb: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (posted_chips, remaining_stacks), including short-stack blind all-in."""
    if len(stacks) != 2 or any(not isinstance(s, int) or s < 0 for s in stacks):
        raise ValueError('stacks must contain two nonnegative integer balances')
    if not (0 < sb < bb):
        raise ValueError('require 0 < small blind < big blind')
    seats = heads_up_seats(button)
    posts = [0, 0]
    posts[seats.small_blind] = min(sb, stacks[seats.small_blind])
    posts[seats.big_blind] = min(bb, stacks[seats.big_blind])
    remaining = tuple(stacks[i] - posts[i] for i in range(2))
    return (posts[0], posts[1]), remaining

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/seat_manager.py" -->

## src/holdem_ai/environment/tournament_env.py

Blob `7c0c70a541c65da07a04c999793937fba1fe74ad` · 10191 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/tournament_env.py" sha="7c0c70a541c65da07a04c999793937fba1fe74ad" -->

````````python
"""Single-table 2-6 seat knockout tournament over the Phase 3 hand engine.

Uses physical seat IDs throughout: busted seats remain in the tuple with
stack 0. Each new hand reuses the *actual* final stacks from the preceding
hand. A hand may eliminate multiple opponents. No rebuy, antes, or ICM.

This is a development/testing environment, not a full tournament certification.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from holdem_ai.environment.blind_manager import BlindLevel, BlindManager
from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine, MultiPlayerHandResult
from holdem_ai.environment.multi_player_seats import rotate_button
from holdem_ai.environment.player_manager import PlayerManager
from holdem_ai.environment.rewards import placement_rewards


@dataclass(frozen=True)
class TournamentHandRecord:
    hand_number: int
    button: int
    level_index: int
    blinds: BlindLevel
    stacks_before: tuple[int, ...]
    stacks_after: tuple[int, ...]
    eliminated: tuple[int, ...]
    hand_result: MultiPlayerHandResult
    # Cards are available only after a multiway showdown; never include folds.
    showdown_hole_cards: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class TournamentResult:
    winner: int
    ranks: tuple[int, ...]
    final_stacks: tuple[int, ...]
    eliminated_at: tuple[int | None, ...]
    hands_played: int
    rewards: tuple[float, ...]
    history: tuple[TournamentHandRecord, ...]


class TournamentEnv:
    """Tournament API: reset -> observe/current_player -> step -> result.

    `max_hands` is a safety guard: raising rather than silently returning a
    spurious champion. Button follows a moving-button, NOT a dead-button rule.
    """

    def __init__(
        self, *, num_players: int = 6, starting_stack: int = 1000,
        starting_stacks: Sequence[int] | None = None,
        button: int = 0,
        blind_levels: Sequence[BlindLevel | tuple[int, int]] | None = None,
        hands_per_level: int = 10,
        seed: int | None = 42,
        max_hands: int = 10_000,
        payouts: Sequence[float] | None = None,
    ) -> None:
        if type(num_players) is not int or not 2 <= num_players <= 6:
            raise ValueError('num_players must be 2..6')
        if starting_stacks is None:
            if type(starting_stack) is not int or starting_stack <= 0:
                raise ValueError('starting_stack must be a positive integer')
            starting_stacks = (starting_stack,) * num_players
        self.starting_stacks = tuple(starting_stacks)
        if len(self.starting_stacks) != num_players:
            raise ValueError('starting_stacks length does not match num_players')
        # PlayerManager validates positive integers.
        PlayerManager(self.starting_stacks)
        if type(button) is not int or not 0 <= button < num_players:
            raise ValueError('invalid initial button')
        if type(max_hands) is not int or max_hands <= 0:
            raise ValueError('max_hands must be positive integer')
        if payouts is not None:
            # Validate immediately using a synthetic full finishing order.
            placement_rewards(tuple(range(1, num_players + 1)), payouts)
        self.blinds = BlindManager(blind_levels, hands_per_level)
        self.num_players = num_players
        self.initial_button = button
        self.seed = seed
        self.max_hands = max_hands
        self.payouts = tuple(payouts) if payouts is not None else None
        self._started = False
        self._done = False
        self._hand: MultiPlayerHandEngine | None = None
        self.hand_history: list[TournamentHandRecord] = []

    def reset(self, seed: int | None = None) -> tuple[Mapping[str, Any] | None, int]:
        """Restart the entire tournament. Optional seed overrides the initial seed."""
        self.players = PlayerManager(self.starting_stacks)
        self.hand_history = []
        self.button = self.initial_button
        self._rng = random.Random(self.seed if seed is None else seed)
        self._started = True
        self._done = False
        self._hand = None
        self._next_hand()
        return (None, -1) if self._done else (self.state, self.current_player)

    def _next_hand(self) -> None:
        """Open a hand, collecting any immediate no-decision showdown hands."""
        while not self._done:
            if len(self.players.alive) == 1:
                self._done = True
                self._hand = None
                return
            if len(self.hand_history) >= self.max_hands:
                raise RuntimeError('max_hands reached before champion; increase limit or blinds')
            hand_number = len(self.hand_history) + 1
            level_index, level = self.blinds.at_hand(hand_number)
            before = self.players.stacks
            hand = MultiPlayerHandEngine(
                stacks=before, button=self.button,
                small_blind=level.small_blind, big_blind=level.big_blind,
                seed=self._rng.getrandbits(64),
            )
            self._hand = hand
            self._hand_number = hand_number
            self._level_index = level_index
            self._level = level
            hand.reset()
            if not hand.is_over():
                return
            self._finish_hand()

    def _finish_hand(self) -> None:
        assert self._hand is not None and self._hand.is_over()
        result = self._hand.get_result()
        before = self.players.stacks
        # Capture at settlement before the next hand overwrites the live engine.
        # 0 or 1 non-folded contenders = uncontested pot (no showdown reveal).
        contenders = [i for i, folded in enumerate(result.folded) if not folded]
        showdown_hole_cards = (
            tuple(tuple(self._hand.hole_cards[i]) if i in contenders else ()
                  for i in range(self.num_players))
            if len(contenders) >= 2 and len(result.board) == 5 else ()
        )
        busts = self.players.update(result.final_stacks, self._hand_number)
        self.hand_history.append(TournamentHandRecord(
            hand_number=self._hand_number,
            button=self.button,
            level_index=self._level_index,
            blinds=self._level,
            stacks_before=before,
            stacks_after=self.players.stacks,
            eliminated=busts,
            hand_result=result,
            showdown_hole_cards=showdown_hole_cards,
        ))
        if self.players.winner is not None:
            self._done = True
            self._hand = None
        else:
            self.button = rotate_button(self.players.stacks, self.button)
            self._hand = None

    @property
    def current_player(self) -> int:
        if not self._started or self._done or self._hand is None:
            return -1
        return self._hand.current_player

    @property
    def state(self) -> Mapping[str, Any]:
        if not self._started or self._done:
            raise RuntimeError('no active tournament decision')
        return self.observe(self.current_player)

    def observe(self, player_id: int) -> Mapping[str, Any]:
        if not self._started or self._done or self._hand is None:
            raise RuntimeError('tournament is not at an active decision')
        observation = dict(self._hand.observe(player_id))
        observation['tournament'] = {
            'hand_number': self._hand_number,
            'level_index': self._level_index,
            'small_blind': self._level.small_blind,
            'big_blind': self._level.big_blind,
            'players_alive': self.players.alive,
            'initial_stacks_this_hand': self.players.stacks,
            'remaining_players': len(self.players.alive),
        }
        return observation

    def _step(self, action_id: int | None, raise_to: int | None = None) -> tuple[Mapping[str, Any] | None, int]:
        if not self._started or self._done or self._hand is None:
            raise RuntimeError('reset tournament before taking actions')
        if raise_to is None:
            assert action_id is not None
            self._hand.step(action_id)
        else:
            self._hand.step_raise_to(raise_to)
        if self._hand.is_over():
            self._finish_hand()
            if not self._done:
                self._next_hand()
        return (None, -1) if self._done else (self.state, self.current_player)

    def step(self, action_id: int) -> tuple[Mapping[str, Any] | None, int]:
        return self._step(action_id)

    def step_raise_to(self, amount: int) -> tuple[Mapping[str, Any] | None, int]:
        return self._step(None, raise_to=amount)

    def is_over(self) -> bool:
        return self._done

    def get_result(self) -> TournamentResult:
        if not self._started or not self._done:
            raise RuntimeError('tournament must end before requesting result')
        winner = self.players.winner
        assert winner is not None and all(r is not None for r in self.players.ranks)
        ranks = tuple(self.players.ranks)
        return TournamentResult(
            winner=winner, ranks=ranks,
            final_stacks=self.players.stacks,
            eliminated_at=tuple(self.players.eliminated_at),
            hands_played=len(self.hand_history),
            rewards=placement_rewards(ranks, self.payouts),
            history=tuple(self.hand_history),
        )

    def play_tournament(self, agents: Sequence[Any], max_actions: int = 100_000) -> TournamentResult:
        if len(agents) != self.num_players:
            raise ValueError('one agent per physical seat is required')
        if type(max_actions) is not int or max_actions <= 0:
            raise ValueError('max_actions must be positive')
        state, actor = self.reset()
        actions = 0
        while not self.is_over():
            if actions >= max_actions:
                raise RuntimeError('max_actions reached before champion')
            selected = agents[actor].select_action(state)
            state, actor = self.step(int(selected))
            actions += 1
        return self.get_result()

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/tournament_env.py" -->

## src/holdem_ai/evaluation/__init__.py

Blob `429b0243634014c4f23357990ca6962a3d7cb2df` · 38 bytes.

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/evaluation/__init__.py" sha="429b0243634014c4f23357990ca6962a3d7cb2df" -->

````````python
"""Game and tournament benchmarks."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/evaluation/__init__.py" -->
