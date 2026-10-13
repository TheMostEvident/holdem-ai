"""Phase 15 phase-3 safety and reward diagnostics regression tests."""
from types import SimpleNamespace

from holdem_ai.experiments.ppo_safety import safety_assessment
from holdem_ai.evaluation.reward_diagnostics import summarize_rollout_rewards
from holdem_ai.training.ppo import TrajectoryRows


def test_low_entropy_is_warning_not_stop_in_new_mode():
    result = safety_assessment(entropy=0.1, approx_kl=0.001,
        entropy_threshold=0.45, max_approx_kl=0.10,
        entropy_stop_enabled=False, consecutive=8)
    assert result["low_entropy"] is True
    assert result["high_kl"] is False
    assert result["consecutive_safety_warnings"] == 0


def test_high_kl_still_counts_with_low_entropy():
    result = safety_assessment(entropy=0.1, approx_kl=0.2,
        entropy_threshold=0.45, max_approx_kl=0.10,
        entropy_stop_enabled=False, consecutive=11)
    assert result["low_entropy"] and result["high_kl"]
    assert result["consecutive_safety_warnings"] == 12


def test_legacy_entropy_stop_can_be_preserved():
    result = safety_assessment(entropy=0.1, approx_kl=0.01,
        entropy_threshold=0.45, max_approx_kl=0.10,
        entropy_stop_enabled=True, consecutive=11)
    assert result["consecutive_safety_warnings"] == 12


def test_reward_diagnostics_fold_and_payoff():
    # Constructor invariants require aligned one-decision-per-hand fake rows.
    rows = TrajectoryRows(
        actors=(SimpleNamespace(legal_action_mask=(True, False, False, False, False, False, False)),
                SimpleNamespace(legal_action_mask=(False, False, True, False, False, False, False))),
        critics=(None, None),
        actions=(0, 2), acting_seats=(0, 1),
        returns_bb=(-0.5, 0.5), hands=2,
        hand_rewards=((-0.5, 0.5), (-1.5, 1.5)),
        terminal_actions=(0, 2), decisions_per_hand=(1, 1))
    result = summarize_rollout_rewards(rows)
    assert result["fold_terminated_hands"] == 1
    assert result["fold_termination_fraction"] == 0.5
    assert result["seat0_mean_reward_bb"] == -1.0
    assert result["showdown_fraction"] is None
