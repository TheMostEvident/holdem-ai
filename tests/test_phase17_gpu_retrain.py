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
