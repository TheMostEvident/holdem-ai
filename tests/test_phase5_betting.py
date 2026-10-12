from dataclasses import replace

import pytest

from holdem_ai.environment.bet_abstraction import BET_ACTION_LABELS, EngineBetAction, NoLimitBetContext, abstract_bet_actions


def ctx(**kwargs):
    defaults = dict(
        pot=100, committed=10, to_call=20, stack=200,
        min_raise_to=50, max_raise_to=210,
        can_fold=True, can_check=False, can_call=True, can_raise=True,
    )
    defaults.update(kwargs)
    return NoLimitBetContext(**defaults)


def test_street_raise_targets_are_integer_post_call_pot():
    actions = abstract_bet_actions(ctx())
    # P=100, call=20, committed=10 -> targets 90, 150, 270 (max 210).
    assert BET_ACTION_LABELS[3] == "raise_half_pot"
    assert actions.resolve(3) == EngineBetAction("raise_to", 90)
    assert actions.resolve(4) == EngineBetAction("raise_to", 150)
    assert actions.actions[5] is None  # capped at all-in; explicit all-in owns it
    assert actions.resolve(6) == EngineBetAction("all_in")
    assert actions.mask == (True, False, True, True, True, False, True)


def test_raise_rounds_up_to_min_and_deduplicates():
    # Raises at min => keep only first fraction with that amount.
    actions = abstract_bet_actions(ctx(pot=0, min_raise_to=80))
    assert actions.resolve(3).raise_to == 80
    assert actions.actions[4] is None
    assert actions.actions[5] is None


def test_short_all_in_allowed_only_as_explicit_action():
    actions = abstract_bet_actions(ctx(stack=30, max_raise_to=40, min_raise_to=50))
    assert actions.mask == (True, False, True, False, False, False, True)


def test_all_in_after_call_not_doubled_as_raise():
    actions = abstract_bet_actions(ctx(stack=20, max_raise_to=30))
    assert actions.mask == (True, False, True, False, False, False, False)


def test_raise_disallowed_when_betting_not_reopened():
    actions = abstract_bet_actions(ctx(can_raise=False))
    assert actions.mask == (True, False, True, False, False, False, False)


def test_check_mode_when_no_call():
    actions = abstract_bet_actions(ctx(to_call=0, can_call=False, can_check=True, min_raise_to=40))
    assert actions.mask[1]
    assert not actions.mask[2]


def test_raise_resolution_rejects_masked():
    actions = abstract_bet_actions(ctx())
    with pytest.raises(ValueError):
        actions.resolve(1)
    with pytest.raises(ValueError):
        actions.resolve(99)


@pytest.mark.parametrize("updates", [
    {"stack": -1}, {"pot": -1}, {"committed": -1},
    {"to_call": -1}, {"min_raise_to": None}, {"max_raise_to": None},
    {"max_raise_to": 211}, {"can_check": True},
])
def test_bad_bet_context_rejected(updates):
    with pytest.raises(ValueError):
        ctx(**updates)


def test_raise_to_amount_required():
    with pytest.raises(ValueError):
        EngineBetAction("raise_to")
    with pytest.raises(ValueError):
        EngineBetAction("call", 42)


def test_all_in_action_not_created_if_not_allowed():
    with pytest.raises(ValueError, match="no legal action"):
        abstract_bet_actions(ctx(can_fold=False, can_call=False, can_check=False, can_raise=False))
