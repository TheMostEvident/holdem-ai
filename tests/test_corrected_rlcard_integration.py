"""End-to-end corrected engine tests; run with RLCard 1.2.0 installed."""
import pytest

pytest.importorskip('rlcard')

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine
from holdem_ai.environment.rules_audit import audit_preflop


@pytest.mark.parametrize('button', [0, 1])
def test_corrected_heads_up_blinds_and_first_action(button):
    engine = HandEngine(starting_stack=200, seed=123, button=button)
    state, player = engine.reset()
    checks = audit_preflop(engine.env.game)
    assert all(c.passed for c in checks), checks
    assert player == button
    assert state['legal_actions']


@pytest.mark.parametrize('button', [0, 1])
def test_bb_acts_first_on_flop_after_preflop_call_and_check(button):
    engine = HandEngine(starting_stack=200, seed=12, button=button)
    _, player = engine.reset()
    assert player == button
    # BTN calls posted BB, BB checks its option.
    engine.step(1)
    assert engine.current_player == 1 - button
    engine.step(1)
    assert engine.current_player == 1 - button
    assert engine.env.game.stage.name == 'FLOP'
    assert len(engine.env.game.public_cards) == 3


def test_real_corrected_100_random_hands():
    engine = HandEngine(starting_stack=200, seed=42)
    agents = (RandomAgent(seed=42), RandomAgent(seed=43))
    for _ in range(100):
        result = engine.play_hand(agents)
        assert sum(result.payoffs_chips) == 0
        assert result.action_history


def test_short_blind_stack_goes_to_showdown_without_stalling():
    engine = HandEngine(starting_stack=2, seed=33)
    # With equal stacks of 2 chips, BB is all-in from posting the blind.
    state, _ = engine.reset()
    assert engine.env.game.players[1].remained_chips == 0
    for _ in range(20):
        if engine.is_over():
            break
        if 1 in state['legal_actions']:
            state, _ = engine.step(1)
        else:
            raise AssertionError('No CHECK_CALL action in nonterminal game')
    assert engine.is_over()
    assert sum(engine.get_result().payoffs_chips) == 0


def test_fold_ends_hand_without_dealing_board_cards():
    engine = HandEngine(starting_stack=200, seed=47)
    engine.reset()
    engine.step(0)
    assert engine.is_over()
    assert engine.env.game.public_cards == []
    result = engine.get_result()
    assert sum(result.payoffs_chips) == 0
    assert sorted(result.payoffs_chips) == [-1.0, 1.0]


def test_step_back_after_a_fold():
    engine = HandEngine(starting_stack=200, seed=47, allow_step_back=True)
    engine.reset()
    engine.step(0)
    assert engine.is_over()
    previous = engine.env.step_back()
    assert previous is not False
    assert not engine.env.is_over()
    assert all(x.passed for x in audit_preflop(engine.env.game))
