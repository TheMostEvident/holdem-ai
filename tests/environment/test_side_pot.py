import pytest

from holdem_ai.environment.pot_manager import build_side_pots, settle_pots


def test_three_way_all_in_has_correct_main_and_side_pots():
    p = build_side_pots((100, 200, 500), (False, False, False))
    assert [(x.amount, x.eligible) for x in p.pots] == [
        (300, (0, 1, 2)), (200, (1, 2)),
    ]
    assert p.uncalled_returns == (0, 0, 300)
    # Player 0 wins main; player 1 wins side; 300 uncalled returned to player 2.
    assert settle_pots(p, ((9,), (8,), (7,)), button=0) == (300, 200, 300)


def test_folded_player_contribution_stays_in_pot():
    p = build_side_pots((100, 100, 100), (False, True, False))
    assert p.pots[0].amount == 300
    assert p.pots[0].eligible == (0, 2)
    assert settle_pots(p, ((10,), None, (7,)), button=0) == (300, 0, 0)


def test_unmatched_wager_returns_to_contributor():
    p = build_side_pots((100, 20), (False, True))
    assert [(pot.amount, pot.eligible) for pot in p.pots] == [(40, (0,))]
    assert p.uncalled_returns == (80, 0)
    assert settle_pots(p, ((8,), None), button=0) == (120, 0)


def test_odd_chip_goes_to_first_winner_left_of_button():
    # Three-way pot 3 chips + a second player only ties once but wins 1 odd.
    p = build_side_pots((1, 1, 1), (False, False, False))
    # Tied players are seats 0 and 2; button 0 -> seat 2 is first left of BTN.
    assert settle_pots(p, ((8,), (7,), (8,)), button=0) == (1, 0, 2)


def test_conservation_and_reject_invalid_states():
    p = build_side_pots((51, 68, 72, 12), (False, False, True, False))
    result = settle_pots(p, ((2,), (4,), None, (9,)), button=1)
    assert sum(result) == 51 + 68 + 72 + 12
    with pytest.raises(ValueError, match='matching fold flags'):
        build_side_pots((1, 2), (False,))
    with pytest.raises(ValueError, match='at least one'):
        build_side_pots((1, 1), (True, True))
