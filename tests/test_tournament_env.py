import pytest

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.tournament_env import TournamentEnv


class PassiveAgent:
    def select_action(self, state):
        return 1  # Check or call, including all-in call when stack is short.


class FoldAgent:
    def select_action(self, state):
        return 0  # May fold even when checking is possible.


def test_reproducible_two_player_complete_tournament():
    options = dict(num_players=2, starting_stack=60, seed=91, hands_per_level=2)
    a = TournamentEnv(**options).play_tournament([RandomAgent(1), RandomAgent(2)])
    b = TournamentEnv(**options).play_tournament([RandomAgent(1), RandomAgent(2)])
    assert a == b
    assert len(a.history) >= 1
    assert sum(a.final_stacks) == 120
    assert sorted(a.ranks) == [1, 2]
    assert sum(a.rewards) == 1.0
    assert a.final_stacks[a.winner] == 120


@pytest.mark.parametrize('num_players', [2, 3, 4, 5, 6])
def test_tournament_random_end_to_end(num_players):
    for seed in range(10):
        env = TournamentEnv(
            num_players=num_players,
            starting_stack=80,
            blind_levels=[(1, 2), (2, 4), (5, 10), (10, 20), (20, 40),
                          (40, 80), (80, 160), (160, 320), (320, 640)],
            hands_per_level=3,
            seed=seed,
            max_hands=1000,
        )
        outcome = env.play_tournament([RandomAgent(seed * 100 + i) for i in range(num_players)])
        assert env.is_over() and env.current_player == -1
        assert sum(outcome.final_stacks) == num_players * 80
        assert outcome.final_stacks[outcome.winner] == num_players * 80
        assert sum(c > 0 for c in outcome.final_stacks) == 1
        assert sum(outcome.rewards) == 1
        assert outcome.rewards[outcome.winner] == 1
        assert len(outcome.history) == outcome.hands_played
        assert sorted({r for r in outcome.ranks})[0] == 1
        for h, record in enumerate(outcome.history):
            assert record.hand_number == h + 1
            assert sum(record.stacks_before) == num_players * 80
            assert sum(record.stacks_after) == num_players * 80
            assert record.stacks_after == record.hand_result.final_stacks
            if h:
                assert record.stacks_before == outcome.history[h - 1].stacks_after
                assert record.button != outcome.history[h - 1].button
                assert record.button in [i for i, c in enumerate(record.stacks_before) if c > 0]
            for bust in record.eliminated:
                assert record.stacks_before[bust] > 0
                assert record.stacks_after[bust] == 0


def test_incremental_step_observation_no_private_cards():
    env = TournamentEnv(num_players=3, starting_stack=100, button=0,
                        blind_levels=[(1, 2)], seed=11)
    state, actor = env.reset()
    assert actor == 0
    assert state['tournament']['hand_number'] == 1
    assert state['tournament']['players_alive'] == (0, 1, 2)
    assert state['raw_obs']['hand'] == list(env._hand.hole_cards[actor])
    opponent = next(i for i in range(3) if i != actor)
    assert str(env._hand.hole_cards[opponent]) not in str(state)
    with pytest.raises(ValueError):
        env.observe(opponent)
    state, next_actor = env.step(1)
    assert next_actor == 1
    assert state['tournament']['hand_number'] == 1


def test_stack_persistence_button_rotation_and_rising_blinds():
    env = TournamentEnv(num_players=3, starting_stack=100,
                        blind_levels=[(1, 2), (2, 4), (4, 8)],
                        hands_per_level=1, seed=3)
    state, actor = env.reset()
    while len(env.hand_history) < 2 and not env.is_over():
        state, actor = env.step(0)
    assert len(env.hand_history) >= 2
    first, second = env.hand_history[:2]
    assert first.stacks_after == second.stacks_before
    assert (first.blinds.small_blind, first.blinds.big_blind) == (1, 2)
    assert (second.blinds.small_blind, second.blinds.big_blind) == (2, 4)
    assert first.button != second.button
    assert env.players.total_chips == 300


def test_no_ante_or_rebuy_and_invalid_input():
    with pytest.raises(ValueError):
        TournamentEnv(num_players=7)
    with pytest.raises(ValueError):
        TournamentEnv(num_players=6, starting_stacks=[10] * 5)
    with pytest.raises(ValueError):
        TournamentEnv(num_players=2, payouts=[1, -1])
    with pytest.raises(ValueError):
        TournamentEnv(num_players=2, max_hands=0)


def test_max_hands_guard_does_not_fabricate_winner():
    env = TournamentEnv(num_players=6, starting_stack=1000,
                        blind_levels=[(1, 2)], max_hands=1, seed=21)
    with pytest.raises(RuntimeError, match='max_hands'):
        env.play_tournament([FoldAgent() for _ in range(6)])


def test_restart_is_reproducible_and_rebuilds_stacks():
    env = TournamentEnv(num_players=2, starting_stack=40,
                        blind_levels=[(2, 4), (4, 8), (8, 16), (16, 32), (32, 64)],
                        hands_per_level=2, seed=19)
    outcome = env.play_tournament([RandomAgent(1), RandomAgent(2)])
    assert outcome.hands_played > 0
    before = env.reset()
    other = TournamentEnv(num_players=2, starting_stack=40,
                          blind_levels=[(2, 4), (4, 8), (8, 16), (16, 32), (32, 64)],
                          hands_per_level=2, seed=19)
    assert before == other.reset()
    assert len(env.hand_history) == 0


def test_degenerate_allin_on_reset_terminates_without_input():
    env = TournamentEnv(num_players=2, starting_stack=1, blind_levels=[(1, 2)], seed=11)
    state, actor = env.reset()
    assert env.is_over() and (state, actor) == (None, -1)
    result = env.get_result()
    assert result.hands_played >= 1
    assert sum(result.final_stacks) == 2
    assert sorted(result.ranks) == [1, 2]


def test_noncontiguous_seats_after_elimination_hand():
    # Target a breakout where seat 1 busts and seats 0 and 2 remain.
    # The engine's official UI still uses physical seats across hands.
    for seed in range(70):
        env = TournamentEnv(num_players=3, starting_stacks=[25, 1, 25],
                            blind_levels=[(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)],
                            hands_per_level=2, seed=seed)
        result = env.play_tournament([PassiveAgent()] * 3)
        if any(rec.stacks_after[1] == 0 and rec.stacks_after[0] > 0
               and rec.stacks_after[2] > 0 for rec in result.history):
            assert any(1 not in [i for i, chip in enumerate(rec.stacks_before) if chip > 0]
                       for rec in result.history[1:])
            return
    pytest.fail('no noncontiguous seat case observed across deterministic seeds')


def test_unequal_stacks_and_terminal_prize_structure():
    env = TournamentEnv(num_players=4, starting_stacks=[11, 17, 23, 29],
                        blind_levels=[(1, 2), (2, 4), (4, 8), (8, 16),
                                      (16, 32), (32, 64), (64, 128)],
                        hands_per_level=1, payouts=[70, 20, 10, 0], seed=8)
    out = env.play_tournament([RandomAgent(i + 100) for i in range(4)])
    assert sum(out.final_stacks) == 80
    assert sum(out.rewards) == 100
    assert out.rewards[out.winner] == 70
    assert len(out.ranks) == 4


def test_manual_allin_raises_and_incremental_interface():
    env = TournamentEnv(num_players=3, starting_stack=80,
                        blind_levels=[(1, 2), (2, 4), (4, 8),
                                      (8, 16), (16, 32), (32, 64), (64, 128)],
                        hands_per_level=3, seed=5)
    state, actor = env.reset()
    assert actor == 0 and state['raw_obs']['min_raise_to'] == 4
    state, actor = env.step_raise_to(6)
    assert actor == 1 and state['raw_obs']['to_call'] == 5
    actions = 1
    while not env.is_over():
        state, actor = env.step(1)
        actions += 1
        assert actions < 1000
    assert sum(env.get_result().final_stacks) == 240


def test_fuzz_unequal_short_stacks_at_large_blinds():
    for n in range(2, 7):
        for seed in range(25):
            uneven = [1 + (seed * 3 + i * 11) % 35 for i in range(n)]
            env = TournamentEnv(num_players=n, starting_stacks=uneven,
                                blind_levels=[(1, 2), (3, 6), (10, 20),
                                              (20, 40), (40, 80), (80, 160),
                                              (160, 320)], hands_per_level=2,
                                button=seed % n, seed=seed)
            out = env.play_tournament([RandomAgent(seed * 100 + i) for i in range(n)])
            assert sum(out.final_stacks) == sum(uneven)
            assert sorted(out.ranks)[0] == 1
            assert all(v >= 0 for v in out.final_stacks)
