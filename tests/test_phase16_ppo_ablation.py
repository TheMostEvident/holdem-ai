"""Phase16 isolated PPO ablations: objectives, KL, checkpoints and safety."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from holdem_ai.encoding.observation import EncodedActor, EncodedCritic
from holdem_ai.experiments import ppo_ablation as a
from holdem_ai.models.transformer import CentralizedQCritic, NetworkConfig, TransformerActor
from holdem_ai.training.ppo import PPOConfig, TrajectoryRows
from holdem_ai.training.ppo_session import create_session, save_checkpoint


def _net():
    return NetworkConfig(
        embed_dim=8, attention_heads=2, transformer_layers=1,
        ff_dim=16, hidden_dim=12, max_tokens=256,
    )


def _rows():
    states = []
    critics = []
    for street in (0., 0., 1., 1.):
        f = [0.] * 20
        f[1] = 1.
        f[2] = street
        states.append(EncodedActor(
            (1,), (0.,), tuple(f),
            (True, False, True, True, True, False, True),
        ))
        c = [0.] * 24
        c[0] = 1.
        c[2] = 1.
        critics.append(EncodedCritic((1,), (0.,), tuple(c)))
    return TrajectoryRows(
        actors=tuple(states), critics=tuple(critics),
        actions=(3, 6, 2, 0), acting_seats=(0, 1, 0, 1),
        returns_bb=(200., -200., -2., 2.),
        hands=2, hand_rewards=((200., -200.), (-2., 2.)),
        terminal_actions=(6, 0), decisions_per_hand=(2, 2),
    )


def test_direct_v_zero_sum_warm_start_and_gradient():
    q = CentralizedQCritic(_net())
    model = a.DirectVCritic(q)
    cb = a.batch_critics(_rows().critics)
    with torch.no_grad():
        predictions = model(cb)
    assert predictions.shape == (4, 2)
    assert torch.allclose(predictions[:, 0], -predictions[:, 1], atol=1e-6)
    loss = predictions[:, 0].sum()
    # Forward is used with gradients after the no_grad correctness probe.
    model(cb)[:, 0].sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_scale_q_head_reparameterizes_all_action_outputs():
    model = CentralizedQCritic(_net())
    cb = a.batch_critics(_rows().critics)
    with torch.no_grad():
        original = model(cb)
    a.scale_q_head_(model, 200.)
    with torch.no_grad():
        scaled = model(cb)
    assert torch.allclose(scaled * 200., original, atol=5e-5)
    with pytest.raises(ValueError):
        a.scale_q_head_(model, -1)


def test_exact_post_update_kl_is_zero_then_positive():
    actor = TransformerActor(_net())
    rows = _rows()
    original = a.snapshot_probs(actor, rows, device="cpu")
    assert a.exact_policy_kl(actor, rows, old_probs=original, device="cpu") == pytest.approx(0., abs=1e-6)
    with torch.no_grad():
        actor.policy_head[-1].bias[6].add_(3.)
    changed = a.exact_policy_kl(actor, rows, old_probs=original, device="cpu")
    assert changed > .01


@pytest.mark.parametrize("variant", ["a1", "a2"])
def test_freeze_and_train_new_variants_cpu(variant):
    actor = TransformerActor(_net())
    q = CentralizedQCritic(_net())
    if variant == "a1":
        a.scale_q_head_(q, 200.)
        critic = q
    else:
        critic = a.DirectVCritic(q)
    rows = _rows()
    frozen = a.freeze_variant(
        actor, critic, rows, variant=variant, target_scale=200.,
        device="cpu", chunk_size=2,
    )
    assert frozen.old_log_probs.shape == (4,)
    assert frozen.advantages.std(unbiased=False).item() == pytest.approx(1., abs=1e-5)
    assert torch.isfinite(frozen.advantages).all()
    cfg = PPOConfig(hands_per_update=2, epochs=1, minibatch_size=4, seed=20)
    ao = torch.optim.Adam(actor.parameters(), lr=3e-4)
    co = torch.optim.Adam(critic.parameters(), lr=3e-4)
    stats = a.optimize_variant(
        actor, critic, ao, co, frozen, cfg, variant=variant,
        target_scale=200., device="cpu", shuffle_seed=123,
    )
    assert stats["minibatches"] == 1
    assert stats["decisions"] == 4
    assert math.isfinite(stats["critic_loss"])


@pytest.mark.parametrize("config", [
    {"variants": ("a0", "a0")},
    {"variants": ("broken",)},
    {"updates": 0},
    {"eval_groups": 1},
    {"env_batch_size": 1000},
    {"max_post_kl": -1.},
    {"target_scale": 0.},
    {"validation_seed": 660000000},
])
def test_invalid_ablation_config(config):
    with pytest.raises(ValueError):
        a.AblationConfig(source_checkpoint="x", root="y", **config)


def test_exclusive_run_path_rejected_prior_to_load(tmp_path):
    source = tmp_path / "dummy.pt"
    source.write_bytes(b"trusted placeholder")
    used = tmp_path / "already_used"
    used.mkdir()
    (used / "report.json").write_text("{}")
    with pytest.raises(FileExistsError):
        a.run_ablation(a.AblationConfig(
            source_checkpoint=str(source), root=str(used),
            variants=("a0",), device="cpu", updates=1,
        ))


def test_pilot_one_update_real_env_and_evaluation_cpu(tmp_path):
    """Small end-to-end new-run smoke with two-hand rollouts, no existing-run edits."""
    source = tmp_path / "source.pt"
    session = create_session(
        seed=77, network_config=_net(),
        ppo_config=PPOConfig(hands_per_update=2, epochs=1, minibatch_size=2),
    )
    save_checkpoint(session, source)
    original = source.read_bytes()
    root = tmp_path / "phase16_new"
    cfg = a.AblationConfig(
        source_checkpoint=str(source), root=str(root),
        variants=("a1",), device="cpu", hands_per_update=2,
        updates=1, minibatch_size=2, env_batch_size=2,
        eval_groups=2, critic_eval_hands=2,
    )
    result = a.run_ablation(cfg)
    assert source.read_bytes() == original
    assert result["variants"]["a1"]["critic_holdout"]["decisions"] >= 2
    assert (root / "run.json").is_file()
    assert (root / "a1" / "result.json").is_file()
    assert (root / "a1" / "final_actor.pt").is_file()
    with pytest.raises(FileExistsError):
        a.run_ablation(cfg)



def test_a3_rejected_update_restores_actor_and_keeps_source(tmp_path, monkeypatch):
    """A3 must rollback a deliberately mutated Actor when post-update KL fails."""
    from holdem_ai.evaluation.checkpoint_loader import load_actor

    path = tmp_path / "source.pt"
    session = create_session(
        seed=77, network_config=_net(),
        ppo_config=PPOConfig(hands_per_update=2, epochs=1, minibatch_size=2),
    )
    save_checkpoint(session, path)
    original_bytes = path.read_bytes()
    saved_actor = load_actor(path, device="cpu")

    def fake_optim(actor, critic, ao, co, frozen, config, *, device, shuffle_seed):
        with torch.no_grad():
            actor.policy_head[-1].bias[6].add_(25.)
            critic.q_head[-1].bias[6].add_(25.)
        return {"critic_loss": 1.0, "minibatches": 1,
                "decisions": len(frozen.rows.actions)}

    monkeypatch.setattr(a, "optimize_ppo", fake_optim)
    monkeypatch.setattr(a, "exact_policy_kl",
                        lambda actor, rows, *, old_probs, device: 10.)
    root = tmp_path / "a3_reject"
    cfg = a.AblationConfig(
        source_checkpoint=str(path), root=str(root),
        variants=("a3",), hands_per_update=2, updates=1,
        minibatch_size=2, env_batch_size=2, eval_groups=2,
        critic_eval_hands=2,
    )
    report = a.run_ablation(cfg)
    after = load_actor(root / "a3" / "final_actor.pt", device="cpu")
    assert report["variants"]["a3"]["rejected_updates"] == 1
    assert all(torch.equal(saved_actor.state_dict()[key], after.state_dict()[key])
               for key in saved_actor.state_dict())
    assert path.read_bytes() == original_bytes
