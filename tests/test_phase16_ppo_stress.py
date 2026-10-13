"""Phase16 A0/A3 long-stress regression and real-HUNL mini integration tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from holdem_ai.experiments import ppo_stress as stress
from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.ppo_session import create_session, save_checkpoint


def _source(tmp_path: Path) -> Path:
    net = NetworkConfig(
        embed_dim=8, attention_heads=2,
        transformer_layers=1, ff_dim=16, hidden_dim=12,
        max_tokens=256,
    )
    session = create_session(
        seed=41, network_config=net,
        ppo_config=PPOConfig(seed=41, hands_per_update=2, epochs=1,
                             minibatch_size=2),
    )
    source = tmp_path / "source.pt"
    save_checkpoint(session, source)
    return source


def _cfg(source: Path, root: Path, variant: str = "a0", **kwargs) -> stress.StressConfig:
    values = dict(
        source_checkpoint=str(source),
        root=str(root),
        variant=variant,
        device="cpu",
        updates=2,
        hands_per_update=2,
        minibatch_size=2,
        epochs=1,
        env_batch_size=2,
        eval_every=2,
        eval_groups=2,
        critic_eval_hands=2,
        save_every=1,
        rollout_seed=680000000,
        validation_seed=78000000,
        critic_seed=690000000,
    )
    values.update(kwargs)
    return stress.StressConfig(**values)


@pytest.mark.parametrize("bad", [
    {"variant": "a2"},
    {"updates": 0},
    {"hands_per_update": 0},
    {"eval_groups": 1},
    {"env_batch_size": 3},
    {"max_post_kl": 0},
    {"hard_stop_kl": .01},
    {"rollout_seed": -1},
    {"critic_seed": 680000001},
    {"max_updates_this_call": 0},
])
def test_reject_invalid_stress_settings(bad):
    values = dict(
        source_checkpoint="model.pt", root="new_run", variant="a0",
        hands_per_update=2, env_batch_size=2,
    )
    values.update(bad)
    with pytest.raises(ValueError):
        stress.StressConfig(**values)


def test_full_checkpoint_resume_no_duplicate_rollouts(tmp_path):
    source = _source(tmp_path)
    source_bytes = source.read_bytes()
    base = _cfg(source, tmp_path / "continued", max_updates_this_call=1)
    first = stress.run_stress(base)
    assert first["attempted"] == 1 and not first["completed"]
    checkpoint = tmp_path / "continued" / "latest.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["stress"]["attempted_updates"] == 1
    assert payload["completed_updates"] == 1
    assert "actor_optimizer" in payload and "critic_optimizer" in payload
    assert payload["stress"]["records"][0]["global_update"] == 1
    assert (tmp_path / "continued" / "snapshots" / "attempt_00001.pt").exists()
    assert source.read_bytes() == source_bytes
    # The checkpoint is a true PPO resume checkpoint, not an actor-only export.
    resumed_session = stress.load_checkpoint(checkpoint, device="cpu")
    assert resumed_session.completed_updates == 1

    second = stress.run_stress(
        replace(base, resume=True, max_updates_this_call=None))
    assert second["attempted"] == 2 and second["completed"]
    assert second["training_hands"] == 4
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert [r["attempted"] for r in state["stress"]["records"]] == [1, 2]
    assert [v["attempted"] for v in state["stress"]["validations"]] == [2]
    assert (tmp_path / "continued" / "snapshots" / "attempt_00002.pt").exists()
    assert source.read_bytes() == source_bytes
    assert stress.run_stress(
        replace(base, resume=True, max_updates_this_call=1))["attempted"] == 2
    with pytest.raises(FileExistsError):
        stress.run_stress(base)


def test_resumed_model_identical_to_uninterrupted(tmp_path):
    source = _source(tmp_path)
    base = _cfg(source, tmp_path / "whole")
    whole = stress.run_stress(base)
    assert whole["completed"]
    cut = _cfg(source, tmp_path / "cut", max_updates_this_call=1)
    stress.run_stress(cut)
    stress.run_stress(replace(cut, resume=True, max_updates_this_call=None))
    p = torch.load(tmp_path / "whole" / "latest.pt", map_location="cpu", weights_only=True)
    q = torch.load(tmp_path / "cut" / "latest.pt", map_location="cpu", weights_only=True)
    for name in ("actor_state", "critic_state"):
        assert all(torch.equal(p[name][k], q[name][k]) for k in p[name])
    for name in ("actor_optimizer", "critic_optimizer"):
        # Parameter-index dictionaries and Adam moments must be restored.
        assert p[name]["param_groups"] == q[name]["param_groups"]
        assert p[name]["state"].keys() == q[name]["state"].keys()
        for k in p[name]["state"]:
            for field in p[name]["state"][k]:
                left, right = p[name]["state"][k][field], q[name]["state"][k][field]
                if torch.is_tensor(left):
                    assert torch.equal(left, right)
                else:
                    assert left == right
    assert p["stress"]["records"] == q["stress"]["records"] or [
        {k: v for k, v in x.items() if k != "elapsed_seconds"}
        for x in p["stress"]["records"]
    ] == [
        {k: v for k, v in x.items() if k != "elapsed_seconds"}
        for x in q["stress"]["records"]
    ]


def test_a3_forced_reject_restores_full_actor_critic(tmp_path, monkeypatch):
    source = _source(tmp_path)
    saved = stress.load_checkpoint(source, device="cpu")
    # Force the KL gate to reject the update, regardless of its normal size.
    monkeypatch.setattr(
        stress, "exact_policy_kl",
        lambda actor, rows, *, old_probs, device: 1.0,
    )
    cfg = _cfg(source, tmp_path / "rejected", variant="a3",
               updates=1, eval_every=1)
    report = stress.run_stress(cfg)
    assert report["attempted"] == 1
    assert report["accepted"] == 0
    assert report["rejected"] == 1
    saved_actor = saved.actor.state_dict()
    saved_critic = saved.critic.state_dict()
    updated = stress.load_checkpoint(tmp_path / "rejected" / "latest.pt", device="cpu")
    assert all(torch.equal(updated.actor.state_dict()[k], value)
               for k, value in saved_actor.items())
    assert all(torch.equal(updated.critic.state_dict()[k], value)
               for k, value in saved_critic.items())
    assert updated.completed_updates == 1
    payload = torch.load(tmp_path / "rejected" / "latest.pt",
                         map_location="cpu", weights_only=True)
    assert payload["stress"]["records"][0]["rejected"] is True


def test_paired_a0_a3_report_same_validation_decks(tmp_path):
    source = _source(tmp_path)
    cfg0 = _cfg(source, tmp_path / "a0", variant="a0", updates=1, eval_every=1)
    cfg3 = _cfg(source, tmp_path / "a3", variant="a3", updates=1, eval_every=1)
    stress.run_stress(cfg0)
    stress.run_stress(cfg3)
    paired = stress.compare_stress_runs(tmp_path / "a0", tmp_path / "a3")
    assert paired["a0_attempted"] == paired["a3_attempted"] == 1
    assert "1" in paired["paired_validations"]
    for label in ("random_legal", "frozen_phase14"):
        assert paired["paired_validations"]["1"][label]["a3_minus_a0_delta_bb100"] == 0.
    with pytest.raises(ValueError, match="cannot pair"):
        # Modify only the CONFIG record on disk in a copy? The function should
        # reject a changed seed even if a maliciously edited manifest mismatches.
        different = _cfg(
            source, tmp_path / "nonmatching", variant="a3",
            updates=1, eval_every=1, validation_seed=78000001)
        stress.run_stress(different)
        stress.compare_stress_runs(tmp_path / "a0", tmp_path / "nonmatching")


def test_completed_source_checkpoint_is_never_overwritten(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError):
        stress.run_stress(_cfg(source, source.parent, variant="a0"))
    assert source.exists()


def test_recover_committed_checkpoint_when_metrics_mirror_is_missing(tmp_path):
    source = _source(tmp_path)
    cfg = _cfg(source, tmp_path / "rebuild",
               max_updates_this_call=1)
    real_mirror = stress._mirrors
    observed = {"calls": 0}

    def interrupt_mirror(*args, **kwargs):
        observed["calls"] += 1
        if observed["calls"] == 2:
            raise RuntimeError("simulated power cut after atomic checkpoint")
        return real_mirror(*args, **kwargs)

    from unittest.mock import patch
    with patch.object(stress, "_mirrors", interrupt_mirror):
        with pytest.raises(RuntimeError, match="simulated power cut"):
            stress.run_stress(cfg)
    full = torch.load(tmp_path / "rebuild" / "latest.pt",
                      map_location="cpu", weights_only=True)
    assert full["stress"]["attempted_updates"] == 1
    resumed = stress.run_stress(replace(cfg, resume=True, max_updates_this_call=None))
    assert resumed["attempted"] == 2
    assert (tmp_path / "rebuild" / "metrics.jsonl").exists()
