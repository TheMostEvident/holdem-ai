from __future__ import annotations

import pytest
import torch

from holdem_ai.core.schema import InformationState
from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.evaluation.strategy_diagnostics import ActionProfile, spr_bucket


def obs(*, mask=(True, False, True, True, True, True, True), street='preflop',
        button=1.0, pot=10.0, stack=20.0, opponent=30.0):
    return InformationState(player_id=0,
                            features=(0., button, 0., pot, 0., 0., stack, opponent, 0., 0., 0., 2.),
                            legal_action_mask=mask, private_cards=('sA', 'dK'),
                            stage=street)


def test_effective_spr_public_observable():
    assert spr_bucket(obs(pot=100., stack=10.)) == 'spr_lt_1'
    assert spr_bucket(obs(pot=10., stack=20.)) == 'spr_1_to_4'
    assert spr_bucket(obs(pot=5., stack=20.)) == 'spr_4_to_10'
    assert spr_bucket(obs(pot=1., stack=20.)) == 'spr_ge_10'


def test_conditional_rate_not_raw_rate_and_strata():
    profile = ActionProfile()
    profile.record(obs(mask=(True, False, True, True, True, True, True)), 0)
    profile.record(obs(mask=(False, True, False, True, True, True, True),
                       street='river', button=0.0, pot=60.), 1)
    out = profile.summary()
    assert out['decisions'] == 2
    assert out['actions']['fold'] == 1
    assert out['legal_opportunities']['fold'] == 1
    assert out['conditional_action_rate']['fold'] == 1.0
    assert out['by_street']['river']['actions']['check'] == 1
    assert out['by_position']['button']['actions']['fold'] == 1
    assert out['by_position']['big_blind']['actions']['check'] == 1
    assert out['by_spr']['spr_lt_1']['actions']['check'] == 1
    assert out['conditional_action_rate']['check'] == 1.0
    assert out['conditional_action_rate']['call'] == 0.0
    assert out['mean_policy_entropy'] is None
    assert 'private_cards' not in str(out)


def test_illegal_selection_rejected():
    profile = ActionProfile()
    with pytest.raises(ValueError):
        profile.record(obs(), 1)
    with pytest.raises(ValueError):
        profile.record(obs(street='unknown'), 0)
    with pytest.raises(ValueError):
        spr_bucket(obs(pot=-1))
    with pytest.raises(ValueError):
        ActionProfile().summary(entropy_batch_size=0)


def test_entropy_uses_only_actor_information():
    torch.manual_seed(22)
    model = TransformerActor(NetworkConfig(embed_dim=32, attention_heads=2,
                            transformer_layers=1, ff_dim=64, hidden_dim=48, max_tokens=256))
    profile = ActionProfile()
    from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
    real_obs = HUNL200Adapter().reset(seed=16).observation
    profile.record(real_obs, real_obs.legal_actions[0])
    profile.record(real_obs, real_obs.legal_actions[-1])
    model.train()
    result = profile.summary(model, device='cpu', entropy_batch_size=1)
    assert result['mean_policy_entropy'] >= 0
    assert -0.000001 <= result['mean_normalized_entropy'] <= 1.000001
    assert model.training is True
    assert 'private_cards' not in str(result)
