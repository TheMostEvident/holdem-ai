import pytest

from holdem_ai.environment.hand_evaluator import rank_five, rank_seven, rank_showdown
from holdem_ai.environment.pot_manager import build_side_pots, settle_pots


def test_hierarchy_of_poker_hands():
    candidates = [
        ['sA', 'hK', 'd9', 'c6', 's3'], # high card
        ['sA', 'hA', 'd9', 'c6', 's3'], # pair
        ['sA', 'hA', 'd9', 'c9', 's3'], # two pair
        ['sA', 'hA', 'dA', 'c6', 's3'], # trips
        ['s2', 'h3', 'd4', 'c5', 's6'], # straight
        ['sA', 'sJ', 's8', 's6', 's3'], # flush
        ['sA', 'hA', 'dA', 'c6', 's6'], # full house
        ['sA', 'hA', 'dA', 'cA', 's3'], # four of a kind
        ['s9', 'sT', 'sJ', 'sQ', 'sK'], # straight flush
    ]
    scores = [rank_five(hand) for hand in candidates]
    assert scores == sorted(scores)
    assert len(set(scores)) == 9


def test_wheel_straight_loses_to_six_high():
    assert rank_five(['sA', 'h2', 'd3', 'c4', 's5']) == (4, 5)
    assert rank_five(['s2', 'h3', 'd4', 'c5', 's6']) > (4, 5)


def test_two_triplets_form_highest_full_house():
    assert rank_seven(['sA', 'hA', 'dA', 'sK', 'hK', 'dK', 'c2']) == (6, 14, 13)


def test_board_plays_and_split_at_showdown():
    board = ['sT', 'sJ', 'sQ', 'sK', 'sA']
    a = rank_showdown(['h2', 'c3'], board)
    b = rank_showdown(['h4', 'd5'], board)
    assert a == b == (8, 14)
    breakdown = build_side_pots((20, 20), (False, False))
    assert settle_pots(breakdown, (a, b), button=0) == (20, 20)


def test_invalid_duplicate_or_bad_card():
    with pytest.raises(ValueError, match='duplicate'):
        rank_seven(['sA', 'sA', 'd5', 'c3', 'h8', 'c9', 'cT'])
    with pytest.raises(ValueError):
        rank_five(['sA', 'sK', 'sT', 'sJ', 'bad'])
