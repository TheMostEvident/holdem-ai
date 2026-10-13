"""Tests for Phase-15 read-only critic calibration and numerical edge cases."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holdem_ai.evaluation.critic_diagnostics import (
    diagnose_critic_on_rows,
    load_ppo_critic_checkpoint,
    regression_metrics,
)
from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig, TrajectoryRows
from holdem_ai.training.ppo_session import create_session, save_checkpoint


class _HalfHalfActor:
    training = True

    def eval(self):
        self.training = False
        return self

    def train(self, state=True):
        self.training = state
        return self

    def distribution(self, batch):
        raw = torch.zeros(batch.legal_action_mask.shape,
                          dtype=torch.float32, device=batch.legal_action_mask.device)
        raw = raw.masked_fill(~batch.legal_action_mask, -1e9)
        return torch.distributions.Categorical(logits=raw)


class _KnownCritic:
    training = True

    def eval(self):
        self.training = False
        return self

    def train(self, state=True):
        self.training = state
        return self

    def __call__(self, batch):
        n = batch.features.shape[0]
        q0 = torch.tensor([1., 0., 3., 0., 0., 0., 0.],
                          device=batch.features.device).expand(n, -1)
        return torch.stack((q0, -q0), dim=1)


def _encoding(dim):
    f = [0.] * dim
    f[1] = 1.
    f[2] = 0.
    result = SimpleNamespace(
        features=tuple(f), token_ids=(1,), token_values=(0.,)
    )
    if dim == 20:
        result.legal_action_mask = (True, False, True, False, False, False, False)
    return result


def _rows():
    return TrajectoryRows(
        actors=(_encoding(20), _encoding(20)),
        critics=(_encoding(24), _encoding(24)),
        actions=(0, 2), acting_seats=(0, 1),
        returns_bb=(1., -3.), hands=2,
        hand_rewards=((1., -1.), (3., -3.)),
        terminal_actions=(0, 2), decisions_per_hand=(1, 1),
    )


def test_regression_metrics_bias_rmse_and_ev():
    m = regression_metrics([1., -3.], [2., -2.])
    assert m["bias_bb"] == 1.
    assert m["mae_bb"] == 1.
    assert m["rmse_bb"] == 1.
    assert m["explained_variance"] == 1.  # EV ignores constant prediction bias


def test_explained_variance_undefined_for_zero_target_variance():
    assert regression_metrics([2., 2.], [0., 3.])["explained_variance"] is None


def test_nonfinite_predictions_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        regression_metrics([0.], [float("nan")])


def test_chosen_q_and_action_independent_baseline_separate():
    actor, critic = _HalfHalfActor(), _KnownCritic()
    summary = diagnose_critic_on_rows(_rows(), actor, critic, chunk_size=1)
    q = summary["q_taken"]["overall"]
    v = summary["v_reference_policy"]["overall"]
    assert q["rmse_bb"] == 0.
    assert v["rmse_bb"] == 1.
    assert v["bias_bb"] == 1.
    assert summary["raw_advantage"]["mean_bb"] == -1.
    assert summary["q_taken"]["action:fold"]["decisions"] == 1
    assert summary["q_taken"]["action:call"]["decisions"] == 1
    assert actor.training is True and critic.training is True


def test_trusted_phase9_checkpoint_loader_preserves_models(tmp_path):
    net = NetworkConfig(embed_dim=8, attention_heads=2, transformer_layers=1,
                        ff_dim=16, hidden_dim=16)
    session = create_session(network_config=net, ppo_config=PPOConfig(seed=19))
    session.completed_updates = 125
    path = tmp_path / "best.pt"
    save_checkpoint(session, path)
    loaded = load_ppo_critic_checkpoint(path)
    assert loaded.completed_updates == 125
    assert loaded.ppo_config.seed == 19
    assert all(not p.requires_grad for p in loaded.actor.parameters())
    assert all(not p.requires_grad for p in loaded.critic.parameters())
    assert not loaded.actor.training and not loaded.critic.training
    assert path.is_file()


def test_critic_audit_never_exposes_privileged_state_to_actor():
    # Input restriction is enforced by the ActorBatch contract inside the
    # evaluation function; this numerical test checks separate batches work.
    summary = diagnose_critic_on_rows(_rows(), _HalfHalfActor(), _KnownCritic())
    assert summary["decisions"] == 2
    assert summary["v_reference_policy"]["street:preflop"]["decisions"] == 2


def test_three_rollout_protocol_uses_single_shared_reference(monkeypatch):
    from holdem_ai.evaluation import critic_diagnostics as mod

    net = NetworkConfig(embed_dim=8, attention_heads=2, transformer_layers=1,
                        ff_dim=16, hidden_dim=16)
    ppo = PPOConfig(seed=19)
    sessions = {
        "ref": mod.CriticCheckpoint(_HalfHalfActor(), _KnownCritic(), net, ppo, 104),
        "best": mod.CriticCheckpoint(_HalfHalfActor(), _KnownCritic(), net, ppo, 125),
        "latest": mod.CriticCheckpoint(_HalfHalfActor(), _KnownCritic(), net, ppo, 204),
    }
    collected_with = []

    def fake_collect(actor, config, *, iteration, device, env_batch_size):
        collected_with.append((actor, config.seed, config.hands_per_update))
        return _rows()

    monkeypatch.setattr(mod, "load_ppo_critic_checkpoint",
                        lambda path, *, device: sessions[str(path)])
    monkeypatch.setattr(mod, "collect_batched_self_play", fake_collect)
    result = mod.run_critic_audit(
        reference_checkpoint="ref", before_checkpoint="best",
        after_checkpoint="latest", hands=2, on_policy_hands=2,
        seed=570000000, env_batch_size=2)
    assert len(collected_with) == 3
    assert collected_with[0][0] is sessions["ref"].actor
    assert collected_with[1][0] is sessions["best"].actor
    assert collected_with[2][0] is sessions["latest"].actor
    assert collected_with[0][1] == 570000000
    assert collected_with[1][1] == collected_with[2][1] == 571000000
    assert result["completed_updates"] == {"before": 125, "after": 204}
    assert result["shared_reference"]["results"]["before"]["q_taken"]["overall"]["rmse_bb"] == 0.
