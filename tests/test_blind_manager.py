import pytest

from holdem_ai.environment.blind_manager import BlindLevel, BlindManager


def test_blind_levels_repeat_last_level():
    b = BlindManager([(5, 10), (10, 20), (20, 40)], hands_per_level=3)
    assert [b.at_hand(i) for i in range(1, 12)] == [
        (0, BlindLevel(5, 10)), (0, BlindLevel(5, 10)), (0, BlindLevel(5, 10)),
        (1, BlindLevel(10, 20)), (1, BlindLevel(10, 20)), (1, BlindLevel(10, 20)),
        (2, BlindLevel(20, 40)), (2, BlindLevel(20, 40)), (2, BlindLevel(20, 40)),
        (2, BlindLevel(20, 40)), (2, BlindLevel(20, 40)),
    ]


@pytest.mark.parametrize('levels,hands', [([], 10), ([(2, 2)], 1), ([(0, 10)], 1),
                                          ([(5, 10), (4, 20)], 2), ([(5, 10)], 0),
                                          ([(5, 10)], True)])
def test_invalid_blinds(levels, hands):
    with pytest.raises(ValueError):
        BlindManager(levels, hands)


def test_blind_hand_number_validation():
    b = BlindManager([(1, 2)])
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            b.at_hand(value)
