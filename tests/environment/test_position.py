import pytest

from holdem_ai.environment.seat_manager import heads_up_seats, post_blinds, rotate_heads_up_button


@pytest.mark.parametrize('button', [0, 1])
def test_heads_up_button_is_sb_and_first_preflop(button):
    seats = heads_up_seats(button)
    assert seats.small_blind == button
    assert seats.preflop_first == button
    assert seats.big_blind == 1 - button
    assert seats.postflop_first == 1 - button
    assert rotate_heads_up_button(button) == 1 - button


def test_blinds_are_charged_to_correct_seats():
    posted, remaining = post_blinds((100, 100), button=0, sb=1, bb=2)
    assert posted == (1, 2)
    assert remaining == (99, 98)
    posted, remaining = post_blinds((100, 1), button=0, sb=1, bb=2)
    assert posted == (1, 1)
    assert remaining == (99, 0)


@pytest.mark.parametrize('button', [-1, 2])
def test_invalid_button_is_rejected(button):
    with pytest.raises(ValueError):
        heads_up_seats(button)
