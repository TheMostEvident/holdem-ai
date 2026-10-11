"""Two-player no-limit betting round for the RLCard 1.2.0 action abstraction.

This module intentionally keeps the upstream integer action IDs (0..4).
Bet sizing uses raise-to targets derived from the pot *after* calling.
A short all-in does not reopen betting for an opponent who already acted.
"""
from __future__ import annotations

from enum import IntEnum


class ActionId(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF_POT = 2
    RAISE_POT = 3
    ALL_IN = 4


class HeadsUpBettingRound:
    """Minimal Round-compatible engine; accepts actual RLCard player objects."""

    def __init__(self, players, big_blind: int, dealer, action_enum, alive, allin, folded):
        if len(players) != 2:
            raise ValueError('HeadsUpBettingRound requires exactly two players')
        self.players = players
        self.num_players = 2
        self.init_raise_amount = big_blind
        self.big_blind = big_blind
        self.dealer = dealer
        self.action_enum = action_enum
        self.alive = alive
        self.allin = allin
        self.folded = folded
        self.raised = [0, 0]
        self.last_full_raise = big_blind
        self.game_pointer = 0
        self.pending: set[int] = set()
        self.acted_since_full_raise: set[int] = set()

    def start_new_round(self, game_pointer, raised=None):
        self.raised = list(raised) if raised is not None else [0, 0]
        self.last_full_raise = self.big_blind
        self.acted_since_full_raise = set()
        live = {i for i, p in enumerate(self.players) if p.status == self.alive and p.remained_chips > 0}
        # With one player all-in, no postflop betting can occur; preflop
        # still needs to offer the remaining player a call/fold decision.
        self.pending = live if raised is not None or len(live) == 2 else set()
        self.game_pointer = game_pointer

    def _to_call(self, actor):
        return max(self.raised) - self.raised[actor]

    def _raise_to(self, actor, action):
        current = self.raised[actor]
        to_call = self._to_call(actor)
        pot_after_call = sum(p.in_chips for p in self.players) + to_call
        factor = 0.5 if action == self.action_enum.RAISE_HALF_POT else 1.0
        raise_increment = int(pot_after_call * factor)
        return current + to_call + raise_increment

    def _can_raise(self, actor):
        opponent = self.players[1 - actor]
        if opponent.status != self.alive or opponent.remained_chips == 0:
            return False  # No legal raise into a player who cannot respond.
        if actor in self.acted_since_full_raise and self._to_call(actor) > 0:
            return False
        p = self.players[actor]
        return p.remained_chips > self._to_call(actor)

    def get_nolimit_legal_actions(self, players=None):
        actor = self.game_pointer
        p = self.players[actor]
        if actor not in self.pending or p.status != self.alive or p.remained_chips <= 0:
            return []
        legal = [self.action_enum.FOLD, self.action_enum.CHECK_CALL]
        if self._can_raise(actor):
            old_high = max(self.raised)
            min_raise_to = (old_high + self.last_full_raise) if old_high else self.big_blind
            max_to = self.raised[actor] + p.remained_chips
            if max_to > old_high:
                legal.append(self.action_enum.ALL_IN)
            for action in (self.action_enum.RAISE_HALF_POT, self.action_enum.RAISE_POT):
                target = self._raise_to(actor, action)
                if min_raise_to <= target <= max_to and target > old_high:
                    legal.append(action)
        return sorted(legal, key=lambda a: a.value)

    def proceed_round(self, players, action):
        actor = self.game_pointer
        if action not in self.get_nolimit_legal_actions(players):
            raise ValueError(f'illegal action {action} for actor={actor}')
        p = self.players[actor]
        old_high = max(self.raised)
        before = self.raised[actor]
        if action == self.action_enum.FOLD:
            p.status = self.folded
            self.pending.clear()
        else:
            if action == self.action_enum.CHECK_CALL:
                chips = min(self._to_call(actor), p.remained_chips)
            elif action == self.action_enum.ALL_IN:
                chips = p.remained_chips
            else:
                chips = self._raise_to(actor, action) - before
            p.bet(chips=chips)
            self.raised[actor] = before + chips
            if p.remained_chips == 0:
                p.status = self.allin
            self.pending.discard(actor)
            self.acted_since_full_raise.add(actor)
            delta = self.raised[actor] - old_high
            if delta > 0:
                if delta >= self.last_full_raise:
                    self.last_full_raise = delta
                    self.acted_since_full_raise = {actor}
                # A short all-in still requires any opponent who owes chips
                # to respond, but it does not restore their raise rights.
                other = 1 - actor
                if self.players[other].status == self.alive and self.players[other].remained_chips > 0:
                    self.pending.add(other)
        # Filter out opponents who can no longer act.
        self.pending = {i for i in self.pending if self.players[i].status == self.alive}
        self.game_pointer = (1 - actor) if (1 - actor) in self.pending else actor
        return self.game_pointer

    def is_over(self):
        return not self.pending or any(p.status == self.folded for p in self.players)
