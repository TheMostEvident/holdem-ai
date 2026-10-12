"""Phase17 integration regression against the real GUI-1.6 hand adapter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from holdem_ai.agents.diagnostic_opponents import CallingStation, TightPolicy
from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.evaluation.continuous_stack import (
    ContinuousEvaluationConfig, _ActionTally, _group_interval, _play_match,
    evaluate_continuous_stacks,
)
from holdem_ai.game_service.interactive_adapter import InteractiveHUNLAdapter
from scripts.phase17_evaluate import build_parser, run


class CallPolicy:
    """Stable tests: legal check/call; otherwise a legal fallback."""

    def act(self, observation, rng):
        for i in (1, 2, 0, 6, 3, 4, 5):
            if i in observation.legal_actions:
                return i
        raise AssertionError("no legal action")


class AllInPolicy:
    def act(self, observation, rng):
        for i in (6, 2, 1, 0, 3, 4, 5):
            if i in observation.legal_actions:
                return i
        raise AssertionError("no legal action")


class ObservationSpy:
    def __init__(self):
        self.calls = 0
        self.observations = []

    def act(self, obs, rng):
        self.calls += 1
        # No engine, deck, CriticState, or other player's private hole cards.
        assert not hasattr(obs, "hole_cards")
        assert not hasattr(obs, "critic_state")
        assert len(obs.private_cards) == 2
        assert len(obs.legal_action_mask) == 7
        self.observations.append((obs.player_id, tuple(obs.private_cards)))
        return CallPolicy().act(obs, rng)


def tiny_config(**kwargs):
    return ContinuousEvaluationConfig(groups=2, hands_per_match=3,
                                      candidate_stacks_bb=(50, 200), **kwargs)


def test_match_uses_variable_start_carries_stacks_and_rotates_button(monkeypatch):
    cfg = tiny_config()
    starts = []
    buttons = []
    real = InteractiveHUNLAdapter.reset_with_stacks

    def watched(self, stacks, *, seed=None):
        starts.append(tuple(stacks))
        buttons.append(self.button)
        return real(self, stacks, seed=seed)

    monkeypatch.setattr(InteractiveHUNLAdapter, "reset_with_stacks", watched)
    match = _play_match(
        CallPolicy(), CallPolicy(), cfg=cfg, candidate_seat=0,
        starting_button=1, candidate_stack_bb=50, group_seed=52,
        tally=_ActionTally(),
    )
    assert starts[0] == (5000, 35000)
    assert match["hands"] == len(starts)
    assert match["hands"] > 1
    assert buttons == [(1 + i) % 2 for i in range(match["hands"])]
    assert all(sum(pair) == 40000 and min(pair) > 0 for pair in starts)
    assert any(pair != starts[0] for pair in starts[1:])
    assert match["net_bb"] == match["final_candidate_bb"] - 50


def test_four_layout_groups_are_reproducible_and_action_masks_accounted():
    cfg = tiny_config()
    a = evaluate_continuous_stacks(RandomLegalPolicy(), CallingStation(), config=cfg)
    b = evaluate_continuous_stacks(RandomLegalPolicy(), CallingStation(), config=cfg)
    assert a == b
    assert len(a["scenarios"]) == 2
    assert a["protocol"] == "continuous_400bb_four_layout_seed_group"
    for scenario in a["scenarios"]:
        data = scenario["candidate"]
        assert data["matches"] == 8  # 2 seed groups * 2 seats * 2 buttons
        assert abs(sum(data[k] for k in ("hand_win_fraction", "hand_loss_fraction", "hand_tie_fraction")) - 1) < 1e-5
        assert 8 <= data["hands"] <= 24
        assert len(data["seed_groups"]) == cfg.groups
        assert all(x["hands"] >= 4 for x in data["seed_groups"])
        assert data["ci95_lower_bb100"] <= data["mean_bb100"] <= data["ci95_upper_bb100"]
        for band_data in data["candidate_action_profile_by_effective_remaining_stack"].values():
            n = sum(band_data["chosen_counts"].values())
            assert n == band_data["actions_taken"]
            for action, count in band_data["chosen_counts"].items():
                assert count <= band_data["legal_opportunities"][action]
                rate = band_data["conditional_action_rate"][action]
                if count:
                    assert rate is not None and 0 <= rate <= 1


def test_bankruptcy_stops_match_without_rebuy_or_truncated_return():
    result = _play_match(AllInPolicy(), CallPolicy(), cfg=tiny_config(),
                         candidate_seat=0, starting_button=0,
                         candidate_stack_bb=200, group_seed=100,
                         tally=_ActionTally())
    assert result["hands"] == 1
    assert result["candidate_bust"] or result["opponent_bust"]
    assert abs(result["net_bb"]) == 200


def test_paired_delta_matches_exact_group_differences():
    cfg = tiny_config()
    a = evaluate_continuous_stacks(RandomLegalPolicy(), CallingStation(),
                                    config=cfg, control=TightPolicy())
    for scenario in a["scenarios"]:
        x = scenario["candidate"]["seed_groups"]
        y = scenario["control"]["seed_groups"]
        deltas = scenario["paired_delta"]["group_deltas_bb100"]
        assert [r["seed"] for r in x] == [r["seed"] for r in y]
        assert deltas == [round(xi["bb100"] - yi["bb100"], 6)
                          for xi, yi in zip(x, y)]
        assert scenario["paired_delta"]["mean_bb100"] == round(sum(
            xi["bb100"] - yi["bb100"] for xi, yi in zip(x, y)) / 2, 6)


def test_policy_receives_only_information_state():
    spy = ObservationSpy()
    result = evaluate_continuous_stacks(spy, TightPolicy(), config=tiny_config())
    assert result["scenarios"]
    assert spy.calls > 0
    assert all(len(cards) == 2 for _, cards in spy.observations)


def test_original_training_adapter_keeps_independent_200bb_reset():
    adapter = HUNL200Adapter(big_blind=100, small_blind=50, stack_bb=200)
    adapter.reset(seed=55)
    assert adapter.engine.initial_stacks == (20000, 20000)
    adapter.reset(seed=56)
    assert adapter.engine.initial_stacks == (20000, 20000)


@pytest.mark.parametrize("options", [
    {"groups": 1}, {"big_blind": 1, "small_blind": 2},
    {"candidate_stacks_bb": (0,)}, {"candidate_stacks_bb": (400,)},
    {"candidate_stacks_bb": (50, 50)}, {"candidate_stacks_bb": (50.0,)},
    {"total_stack_bb": 500}, {"hands_per_match": 0},
])
def test_config_rejects_invalid(options):
    with pytest.raises(ValueError):
        ContinuousEvaluationConfig(**options)


def test_group_interval_separates_matched_groups():
    assert _group_interval([2.0, 2.0])["mean_bb100"] == 2.0
    assert _group_interval([2.0, 2.0])["ci95_lower_bb100"] == 2.0
    with pytest.raises(ValueError):
        _group_interval([2.0])


def test_cli_baseline_smoke_generates_json_csv_md_and_no_overwrite(tmp_path):
    args = build_parser().parse_args([
        "--candidate-builtin", "random", "--opponent", "calling_station",
        "--control-builtin", "tight", "--groups", "2", "--hands-per-match", "2",
        "--candidate-stacks-bb", "50,200", "--output-dir", str(tmp_path / "evaluation"),
        "--seed", "112233", "--no-progress",
    ])
    result = run(args)
    root = tmp_path / "evaluation"
    result_json = json.loads((root / "results.json").read_text(encoding="utf-8"))
    assert result_json == result
    assert len(result_json["scenarios"]) == 2
    assert (root / "report.md").read_text(encoding="utf-8").startswith("# Phase 17")
    assert (root / "summary.csv").read_text(encoding="utf-8").count("candidate,") == 2
    with pytest.raises(FileExistsError):
        run(args)


def test_cli_invalid_stacks_rejected(tmp_path):
    args = build_parser().parse_args([
        "--candidate-builtin", "random", "--output-dir", str(tmp_path / "out"),
        "--candidate-stacks-bb", "25,,300",
    ])
    with pytest.raises(ValueError, match="comma-separated"):
        run(args)
