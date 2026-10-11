"""Checks the installed RLCard (1.2.0) rather than synthetic game objects."""

import pytest

pytest.importorskip('rlcard')

import rlcard

from holdem_ai.environment.rules_audit import audit_preflop


def test_audit_can_observe_real_rlcard_preflop_state():
    env = rlcard.make('no-limit-holdem', config={
        'seed': 123, 'game_num_players': 2, 'chips_for_each': 200,
        'dealer_id': 0,
    })
    env.reset()
    results = audit_preflop(env.game)
    assert {x.name for x in results} == {
        'button_posts_small_blind',
        'opponent_posts_big_blind',
        'button_acts_first_preflop',
    }


@pytest.mark.xfail(reason='Known upstream RLCard 1.2.0 heads-up blind/actor behavior; requires corrected hand engine', strict=False)
def test_rlcard_current_hunl_position_is_not_yet_compliant():
    env = rlcard.make('no-limit-holdem', config={
        'seed': 123, 'game_num_players': 2, 'chips_for_each': 200,
        'dealer_id': 0,
    })
    env.reset()
    assert all(x.passed for x in audit_preflop(env.game))
