import pytest

from holdem_ai.environment.rewards import chip_rewards, placement_rewards


def test_chip_rewards_and_winner_take_all():
    assert chip_rewards((10, -4, -6), 2) == (5.0, -2.0, -3.0)
    assert placement_rewards((3, 1, 2)) == (0.0, 1.0, 0.0)


def test_joint_placing_split_prize_slots():
    rewards = placement_rewards((1, 2, 3, 4, 5, 5), (60, 20, 10, 5, 3, 2))
    assert rewards == (60, 20, 10, 5, 2.5, 2.5)
    assert sum(rewards) == 100


@pytest.mark.parametrize('ranks', [(1, 1, 2, 4), (1, 3, 3, 4), (1, 2, None, 4)])
def test_invalid_rank_sequences(ranks):
    with pytest.raises(ValueError):
        placement_rewards(ranks)


def test_invalid_payouts():
    with pytest.raises(ValueError):
        placement_rewards((1, 2), (10,))
    with pytest.raises(ValueError):
        placement_rewards((1, 2), (10, -1))
