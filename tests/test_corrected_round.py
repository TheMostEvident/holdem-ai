"""Independent betting scenarios requiring no RLCard installation."""
from enum import IntEnum
from types import SimpleNamespace

import pytest

from holdem_ai.environment.corrected_round import ActionId, HeadsUpBettingRound


class Status(IntEnum):
    ALIVE = 0
    ALLIN = 1
    FOLDED = 2


class Player:
    def __init__(self, stack, posted=0):
        self.in_chips = posted
        self.remained_chips = stack - posted
        self.status = Status.ALIVE

    def bet(self, chips):
        amount = min(chips, self.remained_chips)
        self.in_chips += amount
        self.remained_chips -= amount


def make_round(stacks=(200, 200)):
    players = [Player(stacks[0], 1), Player(stacks[1], 2)]
    r = HeadsUpBettingRound(players, 2, SimpleNamespace(pot=3), ActionId,
                            Status.ALIVE, Status.ALLIN, Status.FOLDED)
    r.start_new_round(0, raised=[1, 2])
    return r, players


def test_heads_up_preflop_blinds_call_then_big_blind_option():
    r, p = make_round()
    assert r.game_pointer == 0
    assert 1 in r.get_nolimit_legal_actions()
    r.proceed_round(p, ActionId.CHECK_CALL)
    assert p[0].in_chips == 2
    assert not r.is_over()
    assert r.game_pointer == 1
    assert ActionId.RAISE_POT in r.get_nolimit_legal_actions()
    r.proceed_round(p, ActionId.CHECK_CALL)
    assert r.is_over()


def test_min_raise_preflop_four_chips():
    r, players = make_round()
    assert r._raise_to(0, ActionId.RAISE_HALF_POT) == 4
    assert r._raise_to(0, ActionId.RAISE_POT) == 6
    r.proceed_round(players, ActionId.RAISE_HALF_POT)
    assert players[0].in_chips == 4
    assert r.last_full_raise == 2
    assert r.game_pointer == 1
    assert r._to_call(1) == 2


def test_short_all_in_does_not_reopen_betting():
    r, players = make_round(stacks=(200, 5))
    # BTN first full-raises to 4; BB shoves to 5, a short raise of 1.
    r.proceed_round(players, ActionId.RAISE_HALF_POT)
    assert ActionId.ALL_IN in r.get_nolimit_legal_actions()
    r.proceed_round(players, ActionId.ALL_IN)
    assert r.game_pointer == 0
    assert r._to_call(0) == 1
    assert r.get_nolimit_legal_actions() == [ActionId.FOLD, ActionId.CHECK_CALL]
    r.proceed_round(players, ActionId.CHECK_CALL)
    assert r.is_over()


def test_short_all_in_call_by_smaller_stack():
    r, players = make_round(stacks=(2, 200))
    assert ActionId.ALL_IN not in r.get_nolimit_legal_actions()
    r.proceed_round(players, ActionId.CHECK_CALL)
    assert players[0].in_chips == 2
    assert players[0].status == Status.ALLIN
    assert not r.is_over()  # BB still has the option to check or raise.
    assert r.game_pointer == 1
    r.proceed_round(players, ActionId.CHECK_CALL)
    assert r.is_over()


def test_fold_finishes_round():
    r, players = make_round()
    r.proceed_round(players, ActionId.FOLD)
    assert players[0].status == Status.FOLDED
    assert r.is_over()


def test_illegal_action_rejected():
    r, players = make_round(stacks=(2, 200))
    with pytest.raises(ValueError):
        r.proceed_round(players, ActionId.RAISE_POT)


def test_cannot_raise_once_opponent_is_all_in():
    r, players = make_round(stacks=(200, 200))
    r.proceed_round(players, ActionId.ALL_IN)
    assert players[0].status == Status.ALLIN
    assert r.game_pointer == 1
    assert r.get_nolimit_legal_actions() == [ActionId.FOLD, ActionId.CHECK_CALL]
