from __future__ import annotations

import pytest

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_evaluator import RANKS, SUITS
from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine


def test_preflop_and_postflop_order_three_players():
    e = MultiPlayerHandEngine(num_players=3, seed=9)
    s, actor = e.reset()
    assert actor == 0  # BTN acts first preflop in three players (UTG=BTN)
    assert (e.contributions, e.stacks) == ([0, 1, 2], [200, 199, 198])
    assert s['raw_obs']['to_call'] == 2
    s, actor = e.step(1)  # btn call
    assert actor == 1
    s, actor = e.step(1)  # SB call
    assert actor == 2
    s, actor = e.step(1)  # BB option check
    assert actor == 1 and e.street == 'flop'
    assert len(e.board) == 3 and len(e.burned) == 1
    s, actor = e.step(1)
    assert actor == 2
    s, actor = e.step(1)
    assert actor == 0
    s, actor = e.step(1)
    assert actor == 1 and e.street == 'turn'
    assert len(e.board) == 4 and len(e.burned) == 2


def test_six_players_betting_order():
    e = MultiPlayerHandEngine(num_players=6, seed=1, button=0)
    s, actor = e.reset()
    assert (e.seats.small_blind, e.seats.big_blind) == (1, 2)
    actors = []
    for _ in range(6):
        actors.append(actor)
        s, actor = e.step(1)
    assert actors == [3, 4, 5, 0, 1, 2]
    assert e.street == 'flop' and actor == 1


def test_headsup_compatible_and_private_hole_cards():
    e = MultiPlayerHandEngine(num_players=2, seed=99, button=1)
    s, actor = e.reset()
    assert actor == 1
    assert e.seats.small_blind == 1 and e.seats.big_blind == 0
    assert s['raw_obs']['hand'] == list(e.hole_cards[1])
    assert e.hole_cards[0] != e.hole_cards[1]
    assert str(e.hole_cards[0]) not in str(s)
    with pytest.raises(ValueError):
        e.observe(0)


def test_multiplayer_fold_immediate_no_extra_board():
    e = MultiPlayerHandEngine(num_players=4, seed=7)
    e.reset()
    # BTN=0, SB=1, BB=2, UTG=3. UTG/BTN/SB fold => BB wins immediately.
    for player in [3, 0, 1]:
        assert e.current_player == player
        e.step(0)
    assert e.is_over() and e.board == [] and e.burned == []
    result = e.get_result()
    assert result.payoffs_chips == (0, -1, 1, 0)
    assert sum(result.payoffs_chips) == 0


def test_illegal_actions_have_no_effect():
    e = MultiPlayerHandEngine(num_players=3, seed=2)
    e.reset()
    original = tuple(e.stacks), tuple(e.contributions), tuple(e.action_history)
    with pytest.raises(ValueError):
        e.step(123)
    with pytest.raises(ValueError):
        e.step_raise_to(3)  # first open must raise to >=4
    assert original == (tuple(e.stacks), tuple(e.contributions), tuple(e.action_history))
    with pytest.raises(RuntimeError):
        e.get_result()


def test_full_raise_and_minimum_re_raise():
    e = MultiPlayerHandEngine(num_players=3, seed=4)
    e.reset()
    # Button preflop raise-to 6 -> min re-raise to 10.
    e.step_raise_to(6)
    assert e.current_player == 1 and e.min_raise_to() == 10
    with pytest.raises(ValueError):
        e.step_raise_to(9)
    e.step_raise_to(10)
    assert e.current_player == 2 and e.min_raise_to() == 14


def test_single_active_vs_allins_cannot_raise():
    e = MultiPlayerHandEngine(stacks=[10, 15, 50], button=0)
    e.reset()
    e.step(4)  # P0 allin10
    e.step(4)  # P1 allin15
    assert e.current_player == 2
    assert e.legal_actions == (0, 1)
    e.step(1)  # P2 call 15 (not forced all-in for 50)
    assert e.is_over() and len(e.board) == 5
    r = e.get_result()
    assert sorted(r.contributions) == [10, 15, 15]
    assert sum(r.payoffs_chips) == 0
    assert sum(r.final_stacks) == 75


def test_multiway_short_allin_does_not_reopen_prior_actor():
    # BTN P0=200, SB P1=200, BB P2=15, UTG P3=200.
    # UTG calls 2, BTN raises to 10, SB calls, BB allin to 15,
    # UTG calls extra 13. Action returns to BTN, who faces +5 < 8 full raise.
    e = MultiPlayerHandEngine(stacks=[200, 200, 15, 200], button=0)
    e.reset()
    assert e.current_player == 3
    e.step(1)  # UTG 2
    e.step_raise_to(10)  # BTN
    e.step(1)  # SB call
    e.step(4)  # BB all-in 15 short raise
    assert e.current_player == 3
    e.step(1)
    assert e.current_player == 0
    assert e.legal_actions == (0, 1)
    with pytest.raises(ValueError):
        e.step_raise_to(25)


def test_cumulative_short_allins_reopen_earlier_action():
    # BTN (0) raises to 10 (last full raise 8), SB(1) goes allin 14,
    # BB(2) goes allin 18; BTN faces +8 total -> reopening permitted.
    e = MultiPlayerHandEngine(stacks=[200, 14, 18], button=0)
    e.reset()
    e.step_raise_to(10)
    e.step(4)
    e.step(4)
    assert e.current_player == 0
    assert e.min_raise_to() == 26
    # No one with chips remains to call a raise: may only call or fold.
    assert e.legal_actions == (0, 1)


def test_allin_with_uncalled_chips_and_folds():
    # An initial raise to 20 that nobody calls returns uncalled 18.
    e = MultiPlayerHandEngine(num_players=3, seed=12)
    e.reset()
    e.step_raise_to(20)
    e.step(0)
    e.step(0)
    r = e.get_result()
    assert r.contributions == (20, 1, 2)
    assert r.payoffs_chips == (3, -1, -2)
    assert r.final_stacks == (203, 199, 198)


def test_no_duplicate_dealt_cards_six_players():
    e = MultiPlayerHandEngine(num_players=6, seed=6)
    e.play_hand([RandomAgent(i) for i in range(6)])
    all_cards = [c for hole in e.hole_cards for c in hole] + e.board + e.burned
    assert len(all_cards) == len(set(all_cards))


def test_explicit_tie_odd_chip_seat_left_of_button():
    # 3 player: deal clockwise 1 2 0, twice, with royal-flush board.
    used = ['s2', 'd3', 'c4', 'h5', 's6', 'd7',
            'c8', 'hT', 'hJ', 'hQ', 'd8', 'hK', 'c9', 'hA']
    deck = used + [s + r for s in SUITS for r in RANKS if s + r not in used]
    e = MultiPlayerHandEngine(num_players=3, button=0, deck=deck)
    e.reset()
    e.step(1)  # button calls 2
    e.step(0)  # SB folds (1 chip dead)
    e.step(1)  # BB checks; both players check down
    while not e.is_over():
        e.step(1)
    r = e.get_result()
    assert r.board == ('hT', 'hJ', 'hQ', 'hK', 'hA')
    assert r.payoffs_chips == (0, -1, 1)
    assert r.awards == (2, 0, 3)


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6])
def test_random_stress_chip_conservation(n):
    for seed in range(75):
        e = MultiPlayerHandEngine(num_players=n, seed=seed, button=seed % n)
        agents = [RandomAgent(seed * 11 + i) for i in range(n)]
        result = e.play_hand(agents, max_actions=200)
        assert sum(result.payoffs_chips) == 0
        assert sum(result.final_stacks) == n * 200
        assert min(result.final_stacks) >= 0
        assert sum(result.contributions) == sum(result.awards)
        assert len(result.board) in (0, 3, 4, 5)


def test_non_contiguous_active_seats():
    e = MultiPlayerHandEngine(stacks=[200, 0, 200, 0, 200, 0], button=4, seed=5)
    state, player = e.reset()
    assert (e.seats.small_blind, e.seats.big_blind, e.seats.preflop_first) == (0, 2, 4)
    assert player == 4
    res = e.play_hand([RandomAgent(i) for i in range(6)])
    assert res.final_stacks[1] == res.final_stacks[3] == res.final_stacks[5] == 0
    assert sum(res.final_stacks) == 600


def test_short_big_blind_still_requires_full_bring_in_with_two_active_opponents():
    # BTN and SB have chips; BB can post only one chip of the nominal 2.
    e = MultiPlayerHandEngine(stacks=[200, 200, 1], button=0)
    s, actor = e.reset()
    assert actor == 0 and e.contributions == [0, 1, 1]
    assert s['raw_obs']['to_call'] == 2
    e.step(1)  # BTN calls 2, not the short BB's posted 1
    assert e.contributions == [2, 1, 1]
    assert e.state['raw_obs']['to_call'] == 1
    e.step(1)  # SB completes nominal BB of 2
    assert e.contributions == [2, 2, 1]
    # Flop has two players with chips, and BB is all-in.
    assert e.street == 'flop'
    assert e.current_player == 1


def test_multiway_side_pot_different_main_and_side_winners():
    # P0 AA > P1 KK > P2 QQ; P0 wins main pot, P1 wins side pot.
    # Deal order is seats P1,P2,P0, twice, then burn/flop/burn/turn/burn/river.
    cards = ['sK', 'sQ', 'sA', 'dK', 'dQ', 'dA',
             'h5', 'h2', 'c3', 's4', 'h6', 'd8', 'h7', 'h9']
    deck = cards + [s + r for s in SUITS for r in RANKS if s + r not in cards]
    e = MultiPlayerHandEngine(stacks=[10, 20, 40], deck=deck)
    e.reset()
    e.step(4)  # P0 all-in for 10
    e.step(4)  # P1 all-in for 20
    e.step(1)  # P2 can only call to 20 against the two all-ins
    assert e.is_over()
    r = e.get_result()
    assert r.board == ('h2', 'c3', 's4', 'd8', 'h9')
    assert r.contributions == (10, 20, 20)
    assert r.awards == (30, 20, 0)
    assert r.payoffs_chips == (20, 0, -20)
    assert r.final_stacks == (30, 20, 20)


def test_multiple_short_allins_cumulatively_reopen_action_when_opponents_can_call():
    # 5 seats BTN0/SB1/BB2/UTG3/HJ4. BTN raises to 10;
    # SB all-in14 and BB all-in18 are individually short raises of 4.
    # Facing +8 cumulatively reopens betting for BTN when UTG/HJ have chips.
    e = MultiPlayerHandEngine(stacks=[200, 14, 18, 200, 200], button=0)
    e.reset()
    e.step(1)  # UTG calls 2
    e.step(1)  # HJ calls 2
    e.step_raise_to(10)  # BTN full raise +8
    e.step(4)  # SB all-in14
    e.step(4)  # BB all-in18
    e.step(1)  # UTG calls18
    e.step(1)  # HJ calls18
    assert e.current_player == 0
    assert e.min_raise_to() == 26
    assert 4 in e.legal_actions
    e.step_raise_to(26)
    assert e.current_player == 3


def test_reset_can_finish_without_any_decisions_when_blinds_allin():
    e = MultiPlayerHandEngine(stacks=[1, 1], button=0, seed=1)
    state, actor = e.reset()
    assert state is None and actor == -1
    assert e.is_over() and len(e.board) == 5
    assert sum(e.get_result().final_stacks) == 2


def test_action_raise_to_history_is_recorded_for_replay():
    e = MultiPlayerHandEngine(num_players=3)
    e.reset()
    e.step_raise_to(7)
    assert e.action_history[-1] == (0, 5)
    assert e.action_raise_to[-1] == 7
    assert e.state['raw_obs']['action_raise_to'] == (7,)


def test_small_stacks_stress_and_total_chips_at_every_action():
    import random
    rng = random.Random(2026)
    for hand in range(500):
        n = rng.randint(2, 6)
        stacks = [rng.choice([0, 1, 2, 3, 5, 7, 10, 50]) for _ in range(n)]
        occupied = [i for i, v in enumerate(stacks) if v > 0]
        if len(occupied) < 2:
            continue
        e = MultiPlayerHandEngine(stacks=stacks, button=rng.choice(occupied), seed=hand)
        s, actor = e.reset()
        for _ in range(200):
            assert sum(e.stacks) + sum(e.contributions) == sum(stacks)
            if e.is_over():
                break
            s, actor = e.step(rng.choice(e.legal_actions))
        else:
            raise AssertionError('more than 200 actions in one hand')
        r = e.get_result()
        assert sum(r.final_stacks) == sum(stacks)
        assert sum(r.payoffs_chips) == 0
