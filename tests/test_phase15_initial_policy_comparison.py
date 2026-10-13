"""Agent baseline all-in comparison regression tests (no environment edits)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holdem_ai.encoding.observation import EncodedActor
from holdem_ai.encoding.tensors import batch_actors
from holdem_ai.evaluation import initial_policy_comparison as audit
from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import TrajectoryRows


def _net():
    return NetworkConfig(embed_dim=8, attention_heads=2, transformer_layers=1,
                         ff_dim=16, hidden_dim=16, max_tokens=64)


def _row(stage, allin_legal=True):
    f = [0.] * 20
    f[1] = 1.
    f[2] = float(stage)
    mask = (True, True, True, True, True, True, allin_legal)
    return EncodedActor((1,), (0.0,), tuple(f), mask)


def _rows():
    actors = (_row(0), _row(0, False), _row(0, False),
              _row(0), _row(1, False), _row(1, False))
    return TrajectoryRows(
        actors=actors, critics=(None,) * 6,
        actions=(6, 2, 0, 2, 3, 1),
        acting_seats=(0, 1, 0, 0, 1, 0),
        returns_bb=(-200., 200., -0.5, 2., -2., 2.),
        hands=3,
        hand_rewards=((-200., 200.), (-0.5, 0.5), (2., -2.)),
        terminal_actions=(2, 0, 1),
        decisions_per_hand=(2, 1, 3),
    )


def test_uniform_actor_probability_mask_and_normalization():
    actor = audit.UniformLegalActor(_net())
    mask1 = (True, False, True, False, False, False, True)
    mask2 = (False, True, False, False, False, False, False)
    obs = [
        EncodedActor((1,), (0.,), (0.,) * 20, mask1),
        EncodedActor((1,), (0.,), (0.,) * 20, mask2),
    ]
    dist = actor.distribution(batch_actors(obs, device="cpu"))
    assert torch.allclose(dist.probs[0], torch.tensor([1/3, 0, 1/3, 0, 0, 0, 1/3]))
    assert torch.allclose(dist.probs[1], torch.tensor([0., 1., 0., 0., 0., 0., 0.]))


def test_random_initialization_is_reproducible_and_rng_safe():
    config = _net()
    torch.manual_seed(987)
    before = torch.get_rng_state().clone()
    a = audit.build_untrained_actor(config, seed=52)
    b = audit.build_untrained_actor(config, seed=52)
    c = audit.build_untrained_actor(config, seed=53)
    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k])
               for k in a.state_dict())
    assert any(not torch.equal(a.state_dict()[k], c.state_dict()[k])
               for k in a.state_dict())
    assert not a.training
    assert all(not p.requires_grad for p in a.parameters())


def test_on_policy_action_rate_uses_legal_denominator_and_hand_denominator():
    metrics = audit._action_counts_on_rows(_rows())
    assert metrics["overall"]["decisions"] == 6
    assert metrics["overall"]["all_in_legal_decisions"] == 2
    assert metrics["overall"]["all_in_selected_decisions"] == 1
    assert metrics["overall"]["all_in_rate_given_legal"] == .5
    assert metrics["preflop"]["decisions"] == 4
    assert metrics["preflop"]["all_in_rate_given_legal"] == .5
    assert metrics["per_hand"]["hands"] == 3
    assert metrics["per_hand"]["hands_with_explicit_all_in"] == 1
    assert metrics["per_hand"]["hands_with_explicit_all_in_fraction"] == .333333
    assert metrics["per_hand"]["first_decision_all_in_given_legal"] == .5
    assert metrics["flop"]["all_in_rate_given_legal"] is None
    assert metrics["river"]["decisions"] == 0


def test_shared_seed_policy_order_and_phase14_fixed_observations(monkeypatch):
    trained = SimpleNamespace(config=_net())
    initial = SimpleNamespace(config=_net())
    seen = []
    phase14_obs = _rows().actors
    def fake_collect(model, conf, *, iteration, device, env_batch_size):
        seen.append((model, conf.seed, conf.hands_per_update, env_batch_size))
        return _rows()
    def fake_fixed(models, rows, *, device):
        assert rows is phase14_obs or rows == phase14_obs
        assert set(models) == {"uniform_legal", "random_init", "trained_phase14"}
        return {"sample_count": len(rows)}
    monkeypatch.setattr(audit, "load_actor", lambda path, *, device: trained)
    monkeypatch.setattr(audit, "build_untrained_actor",
                        lambda cfg, *, seed, device: initial)
    monkeypatch.setattr(audit, "collect_batched_self_play", fake_collect)
    monkeypatch.setattr(audit, "compare_actors_on_states", fake_fixed)
    report = audit.compare_three_agent_policies(
        trained_checkpoint="phase14.pt", hands=3, seed=590000000,
        env_batch_size=2, device="cpu", init_seed=52)
    assert [name for name in report["on_policy"]] == [
        "uniform_legal", "random_init", "trained_phase14"]
    assert seen[1][0] is initial and seen[2][0] is trained
    assert all(item[1:] == (590000000, 3, 2) for item in seen)
    assert report["fixed_phase14_states"]["states"] == 6
    assert "NOT_verified" in report["initial_policy_provenance"]


def test_initial_checkpoint_architecture_mismatch_rejected(monkeypatch):
    trained = SimpleNamespace(config=_net())
    wrong = SimpleNamespace(config=NetworkConfig())
    monkeypatch.setattr(audit, "load_actor",
                        lambda path, *, device: wrong if path == "initial" else trained)
    with pytest.raises(ValueError, match="architecture differs"):
        audit.compare_three_agent_policies(
            trained_checkpoint="trained", initial_checkpoint="initial",
            hands=3, env_batch_size=2)


@pytest.mark.parametrize("kwargs", [
    {"hands": 0},
    {"seed": -1},
    {"env_batch_size": 0},
    {"hands": 2, "env_batch_size": 3},
])
def test_bad_sampling_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        audit.compare_three_agent_policies(trained_checkpoint="x", **kwargs)


def test_end_to_end_three_policy_comparison_two_real_hands_cpu(tmp_path):
    """Tiny integration run: real existing Collector, masks, engine and fixed audit."""
    from dataclasses import replace
    from holdem_ai.training.ppo import PPOConfig
    from holdem_ai.training.ppo_session import create_session, save_checkpoint

    path = tmp_path / "seeded_phase14_shape.pt"
    session = create_session(
        seed=52,
        network_config=replace(_net(), max_tokens=256),
        ppo_config=PPOConfig(seed=52, hands_per_update=2),
    )
    save_checkpoint(session, path)
    result = audit.compare_three_agent_policies(
        trained_checkpoint=path,
        hands=2,
        seed=590001000,
        env_batch_size=2,
        device="cpu",
        init_seed=53,
    )
    assert result["fixed_phase14_states"]["states"] > 0
    assert set(result["on_policy"]) == {
        "uniform_legal", "random_init", "trained_phase14",
    }
    for entry in result["on_policy"].values():
        assert entry["terminal"]["hands"] == 2
        assert entry["actions"]["overall"]["decisions"] >= 2
        assert 0 <= entry["actions"]["per_hand"]["hands_with_explicit_all_in"] <= 2
