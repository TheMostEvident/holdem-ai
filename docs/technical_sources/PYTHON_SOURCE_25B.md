# Python source volume 25B/28

[Canonical technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Immutable commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`.

## tests/test_phase17_continuous_stack.py

Git Blob `bb7feea9bf9e053cc9e3af296511cb9e8087548a` (7905 bytes).

<a name="py-295"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase17_continuous_stack.py" sha="bb7feea9bf9e053cc9e3af296511cb9e8087548a" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/test_phase17_continuous_stack.py" -->

## tests/test_phase17_gpu_retrain.py

Git Blob `b23e431ab5da1b6834e615bdaffe59f64fd7a992` (7873 bytes).

<a name="py-296"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase17_gpu_retrain.py" sha="b23e431ab5da1b6834e615bdaffe59f64fd7a992" -->

````````python
"""Phase17 5-hour PPO runner: immutable source, full resume and KL safeguards."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from holdem_ai.experiments import ppo_five_hour as p
from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.ppo_session import create_session, load_checkpoint, save_checkpoint


def _source(tmp_path):
    network = NetworkConfig(
        embed_dim=8, attention_heads=2, transformer_layers=1,
        ff_dim=16, hidden_dim=12, max_tokens=256)
    session = create_session(
        seed=41, network_config=network,
        ppo_config=PPOConfig(seed=41, hands_per_update=2,
                             minibatch_size=2, epochs=1))
    path = tmp_path / "trusted_phase14.pt"
    save_checkpoint(session, path)
    return path


def _cfg(source, root, **override):
    spec = dict(
        source_checkpoint=str(source), root=str(root), device="cpu",
        duration_hours=1., variant="a0", hands_per_update=2,
        minibatch_size=2, epochs=1, env_batch_size=2,
        eval_every=2, eval_groups=2, critic_eval_hands=2,
        save_every=1, max_updates=2,
        rollout_seed=700000000, validation_seed=80000000,
        critic_seed=710000000)
    spec.update(override)
    return p.FiveHourConfig(**spec)


@pytest.mark.parametrize("overrides", (
    {"variant": "a2"},
    {"duration_hours": 0.},
    {"duration_hours": float("nan")},
    {"duration_hours": True},
    {"max_updates": 0},
    {"eval_groups": 1},
    {"env_batch_size": 10},
    {"max_post_kl": -1.},
    {"hard_stop_kl": .01},
    {"rollout_seed": -1},
    {"critic_seed": 700000001},
    {"max_updates_this_call": 0},
))
def test_config_rejects_invalid(overrides):
    with pytest.raises(ValueError):
        _cfg("source", "root", **overrides)


def test_full_ppo_checkpoint_and_segmented_resume(tmp_path):
    source = _source(tmp_path)
    prior = source.read_bytes()
    cfg = _cfg(source, tmp_path / "segmented", max_updates_this_call=1)
    first = p.run_five_hour(cfg)
    assert first["attempted_updates"] == 1
    assert first["stop_reason"] == "invocation_update_limit"
    assert not first["complete"]
    root = tmp_path / "segmented"
    saved = torch.load(root / "latest.pt", map_location="cpu", weights_only=True)
    assert saved["retrain"]["attempted"] == 1
    assert "actor_optimizer" in saved and "critic_optimizer" in saved
    assert load_checkpoint(root / "latest.pt", device="cpu").completed_updates == 1
    assert (root / "snapshots" / "attempt_00001.pt").exists()
    assert source.read_bytes() == prior

    second = p.run_five_hour(replace(
        cfg, resume=True, max_updates_this_call=None))
    assert second["attempted_updates"] == 2
    assert second["training_hands"] == 4
    assert second["stop_reason"] == "max_updates_ceiling"
    assert (root / "snapshots" / "attempt_00002.pt").exists()
    assert (root / "summary.json").exists()
    payload = torch.load(root / "latest.pt", map_location="cpu", weights_only=True)
    assert len(payload["retrain"]["validations"]) == 1
    assert [x["attempted"] for x in payload["retrain"]["updates"]] == [1, 2]
    assert source.read_bytes() == prior

    with pytest.raises(FileExistsError):
        p.run_five_hour(cfg)


def test_resume_weights_equal_uninterrupted(tmp_path):
    source = _source(tmp_path)
    a = _cfg(source, tmp_path / "continuous")
    b = _cfg(source, tmp_path / "split", max_updates_this_call=1)
    p.run_five_hour(a)
    p.run_five_hour(b)
    p.run_five_hour(replace(b, resume=True, max_updates_this_call=None))
    aa = torch.load(tmp_path / "continuous" / "latest.pt", weights_only=True)
    bb = torch.load(tmp_path / "split" / "latest.pt", weights_only=True)
    for key in ("actor_state", "critic_state"):
        assert all(torch.equal(v, bb[key][k]) for k, v in aa[key].items())
    for key in ("actor_optimizer", "critic_optimizer"):
        assert aa[key]["param_groups"] == bb[key]["param_groups"]
        for idx, d in aa[key]["state"].items():
            for name, value in d.items():
                actual = bb[key]["state"][idx][name]
                assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
    assert [x["post_update_exact_kl"] for x in aa["retrain"]["updates"]] == [
        x["post_update_exact_kl"] for x in bb["retrain"]["updates"]]


def test_hard_stop_saves_checkpoint_and_denies_resume(tmp_path, monkeypatch):
    source = _source(tmp_path)
    source_bytes = source.read_bytes()
    real = p._one_update

    def force_large_kl(session, cfg):
        row = real(session, cfg)
        row["post_update_exact_kl"] = 0.11
        return row

    monkeypatch.setattr(p, "_one_update", force_large_kl)
    cfg = _cfg(source, tmp_path / "hard_stop")
    outcome = p.run_five_hour(cfg)
    assert outcome["stop_reason"] == "kl_hard_stop"
    assert outcome["attempted_updates"] == 1
    assert (tmp_path / "hard_stop" / "hard_stop.json").exists()
    assert load_checkpoint(tmp_path / "hard_stop" / "latest.pt").completed_updates == 1
    assert source.read_bytes() == source_bytes
    with pytest.raises(RuntimeError, match="hard-stop"):
        p.run_five_hour(replace(cfg, resume=True))


def test_very_small_time_budget_initializes_without_update(tmp_path):
    source = _source(tmp_path)
    result = p.run_five_hour(_cfg(
        source, tmp_path / "expired", duration_hours=1e-12))
    assert result["stop_reason"] == "time_budget_reached"
    assert result["attempted_updates"] == 0
    assert result["complete"]
    assert (tmp_path / "expired" / "latest.pt").exists()


def test_resume_rejects_scientific_config_change(tmp_path):
    source = _source(tmp_path)
    cfg = _cfg(source, tmp_path / "protected", max_updates_this_call=1)
    p.run_five_hour(cfg)
    with pytest.raises(ValueError, match="scientific settings"):
        p.run_five_hour(replace(cfg, resume=True, hands_per_update=3))
    with pytest.raises(ValueError, match="scientific settings"):
        p.run_five_hour(replace(cfg, resume=True, duration_hours=2.))


def test_recover_missing_mirror_after_atomic_checkpoint(tmp_path, monkeypatch):
    source = _source(tmp_path)
    cfg = _cfg(source, tmp_path / "interrupted", max_updates_this_call=1)
    real = p._log_mirrors
    seen = {"calls": 0}

    def crash_on_second_mirror(*args, **kwargs):
        seen["calls"] += 1
        if seen["calls"] == 2:
            raise RuntimeError("simulated interruption after full save")
        return real(*args, **kwargs)

    monkeypatch.setattr(p, "_log_mirrors", crash_on_second_mirror)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        p.run_five_hour(cfg)
    monkeypatch.setattr(p, "_log_mirrors", real)
    checkpoint = tmp_path / "interrupted" / "latest.pt"
    assert torch.load(checkpoint, weights_only=True)["retrain"]["attempted"] == 1
    completed = p.run_five_hour(replace(cfg, resume=True,
                                        max_updates_this_call=None))
    assert completed["attempted_updates"] == 2
    with (tmp_path / "interrupted" / "metrics.jsonl").open() as f:
        objects = [json.loads(line) for line in f]
    assert len([x for x in objects if x["kind"] == "update"]) == 2


def test_final_eval_after_time_budget_only(tmp_path):
    # Full last-evaluation is tested by max_updates ceiling: capped training
    # never invents a fresh on-policy update just to measure the final actor.
    source = _source(tmp_path)
    cfg = _cfg(source, tmp_path / "final", eval_every=3, max_updates=2)
    result = p.run_five_hour(cfg)
    assert result["attempted_updates"] == 2
    assert result["validation_count"] == 1
    v = torch.load(tmp_path / "final" / "latest.pt", weights_only=True)["retrain"]["validations"]
    assert v[0]["attempted"] == 2

````````

<!-- END_PY_SOURCE path="tests/test_phase17_gpu_retrain.py" -->

## tests/test_phase5_betting.py

Git Blob `da76a1e32f6daccc4d94ba97cd47972539938b7a` (3059 bytes).

<a name="py-297"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase5_betting.py" sha="da76a1e32f6daccc4d94ba97cd47972539938b7a" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/test_phase5_betting.py" -->

## tests/test_phase5_numpy_json.py

Git Blob `21e7b962751ffdbd96f40147564dd75f7227ab0b` (2125 bytes).

<a name="py-298"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase5_numpy_json.py" sha="21e7b962751ffdbd96f40147564dd75f7227ab0b" -->

````````python
"""Regression tests: RLCard public state may include nested NumPy scalars."""

import json

import pytest

from holdem_ai.environment.rlcard_adapter import observe_rlcard_state


def _sample_state(raw_obs):
    return {
        "obs": [0.0, 1.0],
        "legal_actions": {0: None, 2: None},
        "raw_obs": {
            "hand": ["SA", "DK"],
            "public_cards": [],
            "stage": "PREFLOP",
            **raw_obs,
        },
    }


def test_nested_numpy_integer_scalars_in_rlcard_public_state():
    np = pytest.importorskip("numpy")
    state = _sample_state({
        "all_chips": [np.int64(8), np.int32(100)],
        "stakes": {"small_blind": np.int64(5), "big_blind": np.int64(10)},
        "pot": np.int64(30),
        "current_player": np.int64(0),
    })
    observation = observe_rlcard_state(state, player_id=0, num_actions=3)
    public = {name: json.loads(value) for name, value in observation.public_state}
    assert public == {
        "all_chips": [8, 100],
        "stakes": {"big_blind": 10, "small_blind": 5},
        "pot": 30,
        "current_player": 0,
    }
    assert observation.legal_action_mask == (True, False, True)
    assert "8" in observation.infoset_key()


def test_nested_numpy_array_and_floats_are_json_safe():
    np = pytest.importorskip("numpy")
    state = _sample_state({
        "all_chips": [np.array([10, 20]), np.float32(1.25)],
        "pot": np.array(40, dtype=np.int64),
    })
    observation = observe_rlcard_state(state, 0, 3)
    public = {name: json.loads(value) for name, value in observation.public_state}
    assert public == {"all_chips": [[10, 20], 1.25], "pot": 40}


def test_invalid_public_value_rejected_rather_than_leaked_as_repr():
    state = _sample_state({"pot": object()})
    with pytest.raises(TypeError, match="not JSON serializable"):
        observe_rlcard_state(state, 0, 3)


def test_nan_public_value_still_rejected():
    np = pytest.importorskip("numpy")
    state = _sample_state({"pot": [np.float32(float("nan"))]})
    with pytest.raises(ValueError, match="Out of range float"):
        observe_rlcard_state(state, 0, 3)

````````

<!-- END_PY_SOURCE path="tests/test_phase5_numpy_json.py" -->

## tests/test_phase5_rewards.py

Git Blob `26a57ad41d9d47f6aa9ad7ccaae6462d0067cf27` (1406 bytes).

<a name="py-299"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase5_rewards.py" sha="26a57ad41d9d47f6aa9ad7ccaae6462d0067cf27" -->

````````python
import math

import pytest

from holdem_ai.training.reward_models import ChipDeltaReward, WinnerTakeAllReward


def test_cash_zero_sum_in_bb_units():
    r = ChipDeltaReward(scale=50).calculate((200, 200), (300, 100))
    assert r == pytest.approx((2, -2))
    assert sum(r) == pytest.approx(0)


def test_chip_deltas_six_seat_and_bust():
    r = ChipDeltaReward().calculate((100,) * 6, (600, 0, 0, 0, 0, 0))
    assert r == pytest.approx((500, -100, -100, -100, -100, -100))


def test_chip_delta_chips_not_conserved():
    with pytest.raises(ValueError, match="conservation"):
        ChipDeltaReward().calculate((100, 100), (105, 100))
    assert ChipDeltaReward(enforce_conservation=False).calculate((100, 100), (105, 100)) == (5, 0)


def test_bad_scale_and_shape():
    with pytest.raises(ValueError):
        ChipDeltaReward(scale=0)
    with pytest.raises(ValueError):
        ChipDeltaReward().calculate((100, 100), (200,))


def test_winner_take_all_centered_zero_sum():
    r = WinnerTakeAllReward().calculate(winner_id=4, num_players=6)
    assert sum(r) == pytest.approx(0)
    assert r[4] == pytest.approx(5 / 6)
    assert all(r[i] == pytest.approx(-1 / 6) for i in (0, 1, 2, 3, 5))


@pytest.mark.parametrize("winner, players", [(-1, 6), (6, 6), (0, 1)])
def test_invalid_winner(winner, players):
    with pytest.raises(ValueError):
        WinnerTakeAllReward().calculate(winner, players)

````````

<!-- END_PY_SOURCE path="tests/test_phase5_rewards.py" -->

## tests/test_phase5_rlcard_adapter.py

Git Blob `2320576e7001317750ee1cd779145da02ca33f0a` (5089 bytes).

<a name="py-300"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase5_rlcard_adapter.py" sha="2320576e7001317750ee1cd779145da02ca33f0a" -->

````````python
import pytest

from holdem_ai.environment.rlcard_adapter import RLCardTurnDriver, observe_rlcard_state
from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.training.episode_runner import run_self_play
from holdem_ai.utils.snapshot import snapshot


class FakeRawState:
    def __init__(self, seat, turn):
        self.seat = seat
        self.turn = turn
        self.state = {
            "obs": [float(turn), [2.0, 3.0]],
            "legal_actions": {0: None, 2: None, 5: None},
            "raw_obs": {
                "hand": [f"private_{seat}_1", f"private_{seat}_2"],
                "public_cards": ["SA", "DK", "CQ"],
                "stage": "FLOP",
                "all_chips": [100, 100],
                "pot": 25,
                "opponent_hand": ["LEAK", "LEAK"],
                "perfect_information": {"hole_cards_all": [["LEAK"]]},
                "deck": ["LEAK"],
            },
            "all_private_hands": [["LEAK"]],
        }


class FakeRLCard:
    def __init__(self):
        self.counter = 0
        self.actions = []
        self.used_seed = None

    def seed(self, seed):
        self.used_seed = seed

    def reset(self):
        self.counter = 0
        self.actions = []
        return FakeRawState(0, self.counter).state, 0

    def step(self, action):
        self.actions.append(action)
        self.counter += 1
        return FakeRawState(self.counter % 2, self.counter).state, self.counter % 2

    def is_over(self):
        return self.counter >= 4

    def get_payoffs(self):
        return [10, -10]


def test_observation_never_exposes_opponents_hidden_cards_or_engine_data():
    s = FakeRawState(0, 2).state
    o = observe_rlcard_state(s, player_id=0, num_actions=6)
    assert o.private_cards == ("private_0_1", "private_0_2")
    assert o.public_cards == ("SA", "DK", "CQ")
    assert o.legal_action_mask == (True, False, True, False, False, True)
    assert o.features == (2, 2, 3)
    assert "LEAK" not in o.infoset_key()
    assert "private_1" not in o.infoset_key()
    assert "deck" not in o.infoset_key()
    assert o.public_state == (("all_chips", "[100,100]"), ("pot", "25"))


def test_rlcard_adapter_seed_masks_and_selfplay():
    raw = FakeRLCard()
    driver = RLCardTurnDriver(raw, ("fold", "check", "call", "raise_half", "raise_pot", "allin"))
    result = run_self_play(driver, {0: RandomLegalPolicy(), 1: RandomLegalPolicy()}, seed=101)
    assert raw.used_seed == 101
    assert result.rewards == (10.0, -10.0)
    assert result.num_decisions == 4
    assert all(a in (0, 2, 5) for a in raw.actions)
    assert len(result.for_seat(0)) == 2
    assert len(result.for_seat(1)) == 2
    assert result.for_seat(0)[-1].reward == 10
    assert result.for_seat(1)[-1].reward == -10


def test_rlcard_adapter_snapshot_keeps_position_and_deep_state():
    driver = RLCardTurnDriver(FakeRLCard(), tuple(str(i) for i in range(6)))
    driver.reset(22)
    twin = snapshot(driver)
    original_state = driver.step(0)
    cloned_state = twin.step(0)
    assert original_state == cloned_state
    assert twin.env is not driver.env
    twin.env.counter = 999
    assert driver.env.counter == 1


def test_illegal_rlcard_action_rejected_before_engine():
    env = FakeRLCard()
    driver = RLCardTurnDriver(env, tuple(str(i) for i in range(6)))
    driver.reset()
    with pytest.raises(ValueError):
        driver.step(1)
    assert env.actions == []


@pytest.mark.parametrize("legal", [{-1: None}, {6: None}, {"fold": None}])
def test_out_of_range_and_wrong_action_ids_rejected(legal):
    s = FakeRawState(0, 0).state
    s["legal_actions"] = legal
    with pytest.raises((ValueError, TypeError)):
        observe_rlcard_state(s, 0, 6)


def test_rlcard_hidden_raw_history_does_not_leak():
    raw = FakeRawState(0, 0).state
    raw["raw_obs"]["history"] = [("opponent_hole_cards", "LEAK")]
    obs = observe_rlcard_state(raw, 0, 6)
    assert "LEAK" not in obs.infoset_key()


def test_terminal_replay_rejected_and_reward_override():
    env = FakeRLCard()
    driver = RLCardTurnDriver(env, tuple(str(i) for i in range(6)), terminal_rewards_fn=lambda e: [2.0, -2.0])
    driver.reset()
    for _ in range(4):
        result = driver.step(0)
    assert result.rewards == (2.0, -2.0)
    with pytest.raises(RuntimeError):
        driver.step(0)


def test_observation_requires_legal_actions_mapping():
    s = FakeRawState(0, 0).state
    s["legal_actions"] = [0, 2]
    with pytest.raises(TypeError):
        observe_rlcard_state(s, 0, 6)


def test_direct_rlcard_integration_if_library_installed():
    rlcard = pytest.importorskip("rlcard")
    env = rlcard.make("no-limit-holdem", config={"seed": 33})
    labels = tuple(f"action_{i}" for i in range(env.num_actions))
    driver = RLCardTurnDriver(env, labels)
    result = run_self_play(driver, {i: RandomLegalPolicy() for i in range(env.num_players)}, seed=33, max_decisions=500)
    assert len(result.rewards) == env.num_players
    assert result.num_decisions > 0
    assert all(t.action_index in t.observation.legal_actions for t in result.transitions)

````````

<!-- END_PY_SOURCE path="tests/test_phase5_rlcard_adapter.py" -->
