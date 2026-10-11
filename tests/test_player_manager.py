import pytest

from holdem_ai.environment.player_manager import PlayerManager


def test_elimination_and_pre_hand_stack_tiebreak():
    p = PlayerManager([200, 200, 100, 100, 200, 200])
    eliminated = p.update([350, 350, 0, 0, 100, 200], hand_number=1)
    assert eliminated == (2, 3)
    assert p.ranks == [None, None, 5, 5, None, None]
    assert p.alive == (0, 1, 4, 5)
    assert p.eliminated_at == [None, None, 1, 1, None, None]
    # Next same-hand busts: 100 chips ends worse than 200 chips.
    assert p.update([450, 450, 0, 0, 0, 100], hand_number=2) == (4,)
    assert p.ranks[4] == 4
    assert p.update([0, 800, 0, 0, 0, 200], hand_number=3) == (0,)
    assert p.ranks[0] == 3
    assert p.update([0, 1000, 0, 0, 0, 0], hand_number=4) == (5,)
    assert p.winner == 1
    assert p.ranks == [3, 1, 5, 5, 4, 2]


def test_different_prehand_stacks_same_hand_bust():
    p = PlayerManager([5, 10, 20, 30])
    p.update([0, 0, 0, 65], 1)
    assert p.ranks == [4, 3, 2, 1]
    assert p.winner == 3


def test_equal_start_busts_joint_rank():
    p = PlayerManager([10, 10, 10, 10])
    p.update([40, 0, 0, 0], 1)
    assert p.ranks == [1, 2, 2, 2]


def test_chip_conservation_and_no_resurrection():
    p = PlayerManager([10, 10, 10])
    with pytest.raises(ValueError):
        p.update([9, 10, 10], 1)
    assert p.stacks == (10, 10, 10)
    p.update([0, 15, 15], 1)
    with pytest.raises(ValueError):
        p.update([1, 14, 15], 2)
    assert p.stacks == (0, 15, 15)


@pytest.mark.parametrize('stacks', [[0, 10], [-1, 10], [10], [1] * 7, [True, 10]])
def test_invalid_starting_stacks(stacks):
    with pytest.raises(ValueError):
        PlayerManager(stacks)
