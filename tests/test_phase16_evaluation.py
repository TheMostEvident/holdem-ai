from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.agents.diagnostic_opponents import OpponentSpec
from holdem_ai.evaluation.opponent_pool import PoolEvaluationConfig, evaluate_opponent_pool
from holdem_ai.evaluation.seed_blocked import model_fingerprint


@pytest.fixture(scope='module')
def net():
    torch.manual_seed(16)
    return TransformerActor(NetworkConfig(embed_dim=32, attention_heads=2,
                            transformer_layers=1, ff_dim=64, hidden_dim=48,
                            max_tokens=256))


def config():
    return PoolEvaluationConfig(groups=2, seed=16001111, mode='greedy')


def test_multiplayer_pool_reproducible_and_no_mutation(net):
    before = model_fingerprint(net)
    net.train(True)
    opponents = (OpponentSpec('random'), OpponentSpec('calling_station'),
                 OpponentSpec('pot_pressure'), OpponentSpec('tight'))
    a = evaluate_opponent_pool(net, opponents, config=config())
    b = evaluate_opponent_pool(net, opponents, config=config())
    assert a == b
    assert model_fingerprint(net) == before
    assert net.training is True
    assert a['pool_summary']['total_hands'] == 32
    assert a['pool_summary']['opponent_count'] == 4
    assert len(a['opponents']) == 4
    for row in a['opponents']:
        assert row['hands'] == 8
        assert row['groups'] == 2
        assert row['ci95_lower_bb100'] <= row['mean_bb100'] <= row['ci95_upper_bb100']
        p = row['candidate_action_profile']
        assert p['decisions'] == sum(p['actions'].values())
        assert all(p['actions'][x] <= p['legal_opportunities'][x] for x in p['actions'])
        assert sum(x['decisions'] for x in p['by_street'].values()) == p['decisions']
        assert sum(x['decisions'] for x in p['by_spr'].values()) == p['decisions']
    assert 'private_cards' not in str(a)
    assert 'CriticState' not in str(a)


def test_empty_and_duplicate_opponents_rejected(net):
    with pytest.raises(ValueError):
        evaluate_opponent_pool(net, (), config=config())
    with pytest.raises(ValueError):
        evaluate_opponent_pool(net, (OpponentSpec('random'), OpponentSpec('random')),
                               config=config())


@pytest.mark.parametrize('changes', [
    {'groups': 1}, {'groups': True}, {'seed': -1}, {'mode': 'wrong'},
    {'device': 'gpu'}, {'max_decisions': 0}, {'entropy_batch_size': 0},
    {'big_blind': 1},
])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


def test_neural_checkpoint_mirror_and_profile(net, tmp_path):
    source = tmp_path / 'anchor.pt'
    torch.save({'network_config': vars(net.config),
                'actor_state': net.state_dict()}, source)
    out = evaluate_opponent_pool(net, (OpponentSpec('neural_copy', str(source)),),
                                 config=config())
    assert out['opponents'][0]['mean_bb100'] == 0.0
    assert out['opponents'][0]['opponent_id'] == model_fingerprint(net)


def test_on_group_callback(net):
    seen = []
    evaluate_opponent_pool(net, (OpponentSpec('random'),),
                           config=config(), on_group=seen.append)
    assert seen == ['random', 'random']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_only_when_available(net):
    actor = TransformerActor(net.config).cuda()
    actor.load_state_dict(net.state_dict())
    out = evaluate_opponent_pool(actor, (OpponentSpec('random'),),
                                 config=replace(config(), device='cuda'))
    assert out['pool_summary']['total_hands'] == 8
