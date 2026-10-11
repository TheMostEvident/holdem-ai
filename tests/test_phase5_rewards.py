import math

import pytest

from holdem_agent_phase5 import ChipDeltaReward, WinnerTakeAllReward


def test_cash_zero_sum_in_bb_units():
    r = ChipDeltaReward(scale=50).calculate((200, 200), (300, 100))
    assert r == pytest.approx((2, -2))
    assert sum(r) == pytest.approx(0)


def test_chip_deltas_six_seat_and_bust():
    r = ChipDeltaReward().calculate((100,) * 6, (600, 0, 0, 0, 0, 0))
    assert r == pytest.approx((500, -100, -100, -100, -100, -100))


def test_chip_delta_chips_not_conserved():
    with pytest.raises(ValueError, match="conservation"):
        ChipDeltaReward().calculate((100, 100), (105, 100))
    assert ChipDeltaReward(enforce_conservation=False).calculate((100, 100), (105, 100)) == (5, 0)


def test_bad_scale_and_shape():
    with pytest.raises(ValueError):
        ChipDeltaReward(scale=0)
    with pytest.raises(ValueError):
        ChipDeltaReward().calculate((100, 100), (200,))


def test_winner_take_all_centered_zero_sum():
    r = WinnerTakeAllReward().calculate(winner_id=4, num_players=6)
    assert sum(r) == pytest.approx(0)
    assert r[4] == pytest.approx(5 / 6)
    assert all(r[i] == pytest.approx(-1 / 6) for i in (0, 1, 2, 3, 5))


@pytest.mark.parametrize("winner, players", [(-1, 6), (6, 6), (0, 1)])
def test_invalid_winner(winner, players):
    with pytest.raises(ValueError):
        WinnerTakeAllReward().calculate(winner, players)
