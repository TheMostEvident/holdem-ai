import pytest

from holdem_ai.environment.actions import BetContext, legal_basic_actions, validate_raise_to


def test_bb_min_raise_preflop():
    ctx = BetContext(actor_commitment=1, highest_commitment=2,
                     last_full_raise=2, remaining_stack=99, big_blind=2)
    assert ctx.amount_to_call == 1
    assert ctx.min_full_raise_to == 4
    assert validate_raise_to(ctx, 4) == 'full_raise'
    with pytest.raises(ValueError, match='below minimum'):
        validate_raise_to(ctx, 3)


def test_postflop_bet_minimum_and_stack_cap():
    ctx = BetContext(actor_commitment=0, highest_commitment=0,
                     last_full_raise=2, remaining_stack=7, big_blind=2)
    assert legal_basic_actions(ctx) == ('fold', 'check', 'raise')
    assert validate_raise_to(ctx, 2) == 'full_raise'
    with pytest.raises(ValueError):
        validate_raise_to(ctx, 8)


def test_short_all_in_is_allowed_but_is_not_a_full_raise():
    ctx = BetContext(actor_commitment=0, highest_commitment=10,
                     last_full_raise=10, remaining_stack=14, big_blind=2)
    assert validate_raise_to(ctx, 14) == 'short_all_in_raise'
    with pytest.raises(ValueError):
        validate_raise_to(ctx, 13)
    already_acted = BetContext(actor_commitment=10, highest_commitment=14,
                               last_full_raise=10, remaining_stack=100, big_blind=2,
                               may_raise=False)
    assert legal_basic_actions(already_acted) == ('fold', 'call')
    with pytest.raises(ValueError, match='not reopened'):
        validate_raise_to(already_acted, 30)


def test_short_call_forces_all_in_not_raise():
    ctx = BetContext(actor_commitment=2, highest_commitment=100,
                     last_full_raise=20, remaining_stack=9, big_blind=2)
    assert ctx.amount_to_call == 9
    assert legal_basic_actions(ctx) == ('fold', 'call')
    assert not ctx.can_raise
