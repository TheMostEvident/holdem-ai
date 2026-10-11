import pytest

from holdem_ai.environment.multi_player_seats import multi_player_seats, rotate_button


def test_heads_up_seat_rule_and_sparse_empty_seats():
    p = multi_player_seats([200, 200], button=0)
    assert (p.small_blind, p.big_blind, p.preflop_first, p.postflop_first) == (0, 1, 0, 1)
    q = multi_player_seats([0, 200, 0, 200, 0, 0], button=3)
    assert (q.small_blind, q.big_blind, q.preflop_first, q.postflop_first) == (3, 1, 3, 1)


def test_three_and_six_players():
    p = multi_player_seats([200] * 3, button=2)
    assert (p.small_blind, p.big_blind, p.preflop_first, p.postflop_first) == (0, 1, 2, 0)
    p = multi_player_seats([200] * 6, button=1)
    assert (p.small_blind, p.big_blind, p.preflop_first, p.postflop_first) == (2, 3, 4, 2)


def test_rotate_button_skips_empty_seats():
    assert rotate_button([100, 0, 100, 0, 100, 0], 0) == 2
    assert rotate_button([100, 0, 100, 0, 100, 0], 4) == 0
    assert rotate_button([0, 100, 0, 50, 0, 0], 1) == 3


@pytest.mark.parametrize('stacks,button', [([1], 0), ([0, 0, 1], 2), ([1, -1], 0), ([1, 1], 3)])
def test_invalid_seats(stacks, button):
    with pytest.raises(ValueError):
        multi_player_seats(stacks, button)
