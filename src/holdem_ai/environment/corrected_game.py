"""RLCard 1.2.0 game subclass fixing heads-up seats and betting round.

Upstream RLCard source is never modified. Integration is version-guarded.
"""
from __future__ import annotations

from importlib.metadata import version

from holdem_ai.environment.corrected_round import HeadsUpBettingRound
from holdem_ai.environment.hand_evaluator import rank_showdown
from holdem_ai.environment.pot_manager import build_side_pots, settle_pots


def make_corrected_rlcard_env(*, starting_stack=200, seed=42, button=0, allow_step_back=False):
    if version('rlcard') != '1.2.0':
        raise RuntimeError('Corrected HUNL adapter is verified only for rlcard==1.2.0')
    if not isinstance(starting_stack, int) or starting_stack < 2:
        raise ValueError('starting_stack must be >= 2')
    if button not in (0, 1):
        raise ValueError('button must be player 0 or 1')

    import rlcard
    from copy import deepcopy
    from rlcard.games.nolimitholdem import Game as UpstreamGame
    from rlcard.games.nolimitholdem.round import Action
    from rlcard.games.limitholdem import PlayerStatus

    class CorrectedGame(UpstreamGame):
        """Locally substituted game, preserving the RLCard Env API."""

        def init_game(self):
            # Use RLCard's seeded deck, card distribution and player creation.
            super().init_game()
            if self.num_players != 2:
                raise ValueError('CorrectedGame supports only two players')
            # RLCard initially posts blinds using its multiplayer formula,
            # which is wrong for heads-up. Restore the chips before reposting.
            for player in self.players:
                player.remained_chips += player.in_chips
                player.in_chips = 0
                player.status = PlayerStatus.ALIVE
            btn = int(self.dealer_id)
            bb = 1 - btn
            self.players[btn].bet(chips=self.small_blind)
            self.players[bb].bet(chips=self.big_blind)
            for player in self.players:
                if player.remained_chips == 0:
                    player.status = PlayerStatus.ALLIN
            self.game_pointer = btn
            self.round = HeadsUpBettingRound(
                self.players, self.big_blind, self.dealer, Action,
                PlayerStatus.ALIVE, PlayerStatus.ALLIN, PlayerStatus.FOLDED,
            )
            self.round.start_new_round(self.game_pointer, [p.in_chips for p in self.players])
            # Dealer pot must match the new blind contributions, otherwise
            # the first pot-sized action would be computed incorrectly.
            self.dealer.pot = sum(p.in_chips for p in self.players)
            return self.get_state(self.game_pointer), self.game_pointer

        def step(self, action):
            if action == Action.FOLD:
                if action not in self.get_legal_actions():
                    raise ValueError('Illegal fold')
                if self.allow_step_back:
                    self.history.append((
                        deepcopy(self.round), self.game_pointer, self.round_counter,
                        deepcopy(self.dealer), deepcopy(self.public_cards),
                        deepcopy(self.players),
                    ))
                actor = self.game_pointer
                self.round.proceed_round(self.players, action)
                self.game_pointer = 1 - actor
                self.round.game_pointer = self.game_pointer
                # The hand terminates immediately, retaining the board as-is.
                # Unlike the upstream game, no extra cards are dealt on a fold.
                return self.get_state(self.game_pointer), self.game_pointer
            return super().step(action)

        def step_back(self):
            result = super().step_back()
            if result:
                self.round.players = self.players
                self.round.dealer = self.dealer
            return result

        def get_payoffs(self):
            if not self.is_over():
                raise RuntimeError('Cannot calculate payoffs before the hand ends')
            folded = [p.status == PlayerStatus.FOLDED for p in self.players]
            contributions = [int(p.in_chips) for p in self.players]
            breakdown = build_side_pots(contributions, folded)
            if sum(not f for f in folded) == 1:
                # Fold winner needs no completed five-card board.
                strengths = [None if f else (0,) for f in folded]
            else:
                if len(self.public_cards) != 5:
                    raise AssertionError('Showdown requires five community cards')
                board = [c.get_index() for c in self.public_cards]
                strengths = [rank_showdown([c.get_index() for c in p.hand], board) for p in self.players]
            awards = settle_pots(breakdown, strengths, button=int(self.dealer_id))
            payoffs = [awards[i] - contributions[i] for i in range(2)]
            if sum(payoffs) != 0:
                raise AssertionError('Chip conservation violated')
            return payoffs

    env = rlcard.make('no-limit-holdem', config={
        'seed': seed,
        'game_num_players': 2,
        'chips_for_each': starting_stack,
        'dealer_id': button,
        'allow_step_back': allow_step_back,
    })
    upstream = env.game
    corrected = CorrectedGame(allow_step_back=allow_step_back, num_players=2)
    corrected.configure({
        'game_num_players': 2,
        'chips_for_each': starting_stack,
        'dealer_id': button,
    })
    # Retain RLCard Env's configured RNG, used by all dealt cards.
    corrected.np_random = upstream.np_random
    env.game = corrected
    return env
