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
