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
