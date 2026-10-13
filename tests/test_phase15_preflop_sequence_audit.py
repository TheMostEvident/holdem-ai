"""Phase15 agent-only preflop sequence audit checks."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from holdem_ai.evaluation import preflop_sequence_audit as audit
from holdem_ai.training.ppo import TrajectoryRows


def _obs(street, seat=0, to_call=0.0, allin=True):
    features = [0.] * 20
    features[2] = float(street)
    features[1] = float(seat == 0)
    features[4] = float(to_call)
    return SimpleNamespace(
        features=tuple(features),
        legal_action_mask=(True, True, True, True, True, True, allin),
    )


def _rows():
    # (1) preflop raise_pot -> all_in -> call (last decision preflop)
    # (2) preflop fold
    # (3) preflop call -> flop check -> flop fold
    actors = (
        _obs(0, 0, 0), _obs(0, 1, 3), _obs(0, 0, 200, allin=False),
        _obs(0, 0, 0),
        _obs(0, 0, 0), _obs(1, 1, 0, allin=False),
        _obs(1, 0, 5, allin=False),
    )
    actions = (4, 6, 2, 0, 2, 1, 0)
    return TrajectoryRows(
        actors=actors, critics=(None,) * len(actions),
        actions=actions, acting_seats=(0, 1, 0, 0, 0, 1, 0),
        returns_bb=(200., -200., 200., -0.5, -3., 3., -3.),
        hands=3, hand_rewards=((200., -200.), (-0.5, 0.5), (-3., 3.)),
        terminal_actions=(2, 0, 0),
        decisions_per_hand=(3, 1, 3),
    )


def test_preflop_sequences_exact_actions_and_first_allin():
    report = audit.summarize_preflop_sequences(_rows())
    assert report["hands"] == 3
    assert report["preflop_explicit_allin_hands"] == 1
    assert report["preflop_explicit_allin_fraction"] == .333333
    assert report["preflop_terminal_fold_hands"] == 1
    assert report["preflop_terminal_call_hands"] == 1
    assert report["first_allin_decision_number_counts"] == {"2": 1}
    assert report["immediately_before_first_allin"] == {"raise_pot": 1}
    assert report["allin_with_prior_raise_head_count"] == 1
    assert report["first_actions"] == {"call": 1, "fold": 1, "raise_pot": 1}
    assert report["preflop_decisions_per_hand"]["mean"] == pytest.approx(5 / 3, abs=1e-6)
    assert report["most_common_prefixes_through_first_allin"] == [
        {"sequence": "raise_pot > all_in", "count": 1},
    ]


def test_conditional_legal_rates_at_decision_depth():
    report = audit.summarize_preflop_sequences(_rows())
    d2 = report["action_choice_by_preflop_decision_number"]["2"]
    assert d2["all_in"]["selected"] == 1
    assert d2["all_in"]["legal_decisions"] == 1
    assert d2["all_in"]["selected_given_legal"] == 1.
    d3 = report["action_choice_by_preflop_decision_number"]["3"]
    assert d3["all_in"]["selected"] == 0
    assert d3["all_in"]["legal_decisions"] == 0
    assert d3["all_in"]["selected_given_legal"] is None
    contexts = report["action_choice_by_context"]
    assert contexts["facing:facing_bet"]["all_in"]["selected"] == 1
    assert contexts["facing:not_facing_bet"]["all_in"]["selected"] == 0


def test_boundaries_required_for_sequence_audit():
    rows = _rows()
    missing = TrajectoryRows(
        rows.actors, rows.critics, rows.actions, rows.acting_seats,
        rows.returns_bb, rows.hands, rows.hand_rewards,
    )
    with pytest.raises(ValueError, match="boundaries"):
        audit.summarize_preflop_sequences(missing)


def test_mismatched_terminal_action_is_rejected():
    rows = _rows()
    invalid = TrajectoryRows(
        rows.actors, rows.critics, rows.actions, rows.acting_seats,
        rows.returns_bb, rows.hands, rows.hand_rewards,
        (0, 0, 0), rows.decisions_per_hand,
    )
    with pytest.raises(ValueError, match="terminal action mismatch"):
        audit.summarize_preflop_sequences(invalid)


def test_seeded_three_policy_comparison_protocol(monkeypatch):
    from holdem_ai.models.transformer import NetworkConfig

    net = NetworkConfig()
    trained = SimpleNamespace(config=net)
    initial = SimpleNamespace(config=net)
    seen = []
    monkeypatch.setattr(audit, "load_actor", lambda path, *, device: trained)
    monkeypatch.setattr(audit, "build_untrained_actor",
                        lambda config, *, seed, device: initial)

    def fake_collect(actor, config, *, iteration, device, env_batch_size):
        seen.append((actor, config.seed, config.hands_per_update))
        return _rows()

    monkeypatch.setattr(audit, "collect_batched_self_play", fake_collect)
    result = audit.compare_preflop_sequences(
        trained_checkpoint="phase14.pt", hands=3,
        seed=600000000, env_batch_size=2)
    assert set(result["models"]) == {
        "uniform_legal", "random_init", "trained_phase14",
    }
    assert seen[1][0] is initial
    assert seen[2][0] is trained
    assert all(item[1:] == (600000000, 3) for item in seen)


@pytest.mark.parametrize("kwargs", [
    {"hands": 0},
    {"seed": -1},
    {"env_batch_size": 0},
    {"hands": 2, "env_batch_size": 3},
])
def test_invalid_sampling_configuration(kwargs):
    with pytest.raises(ValueError):
        audit.compare_preflop_sequences(trained_checkpoint="x", **kwargs)
