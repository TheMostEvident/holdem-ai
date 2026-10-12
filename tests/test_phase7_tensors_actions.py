"""Action legality, padding masks, numeric tensors and gradients (GPU optional)."""
from __future__ import annotations

import json
import pytest

from holdem_ai.core.schema import InformationState
from holdem_ai.environment.hunl200_adapter import CriticState
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.encoding.actions import encode_action, masked_logits, masked_categorical
from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.encoding.vocabulary import PAD


def _observation(history=(), mask=(True, False, True, True, True, False, True)):
    return InformationState(
        0, (0.,1.,0.,1.5,.5,.5,199.5,199.,1.,.5,1.,2.), mask,
        ('sA','dK'), (), 'preflop', (('status',json.dumps(['active','active'])),), history,
    )


def _state(history=()):
    return CriticState(0,0,'preflop',(),(('sA','dK'),('hQ','cJ')),
                       (19950,19900),(50,100),(50,100),('active','active'),150,history,False)


def test_action_onehot_respects_legality():
    mask = _observation().legal_action_mask
    assert encode_action(2,mask) == (0.,0.,1.,0.,0.,0.,0.)
    with pytest.raises(ValueError):
        encode_action(1,mask)
    with pytest.raises(ValueError):
        encode_action(True,mask)
    with pytest.raises(ValueError):
        encode_action(7,mask)


def test_batch_padding_attention_and_values():
    torch = pytest.importorskip('torch')
    a = encode_actor(_observation())
    b = encode_actor(_observation(('preflop|seat=0|raise_to|raise_to=450',)))
    batch = batch_actors([a,b])
    assert batch.token_ids.shape == (2,len(b.token_ids))
    assert batch.token_values.dtype == torch.float32
    assert batch.features.shape == (2,20)
    assert batch.legal_action_mask.shape == (2,7)
    assert batch.legal_action_mask.dtype == torch.bool
    assert batch.attention_mask.dtype == torch.bool
    assert batch.token_ids[0,-1].item() == PAD
    assert not batch.attention_mask[0,-1].item()
    assert batch.token_values[1,-1].item() == 4.5


def test_critic_batch_shapes_and_amounts():
    torch = pytest.importorskip('torch')
    a = encode_critic(_state())
    b = encode_critic(_state(('preflop|seat=0|raise_to|raise_to=750',)))
    batch = batch_critics([a,b])
    assert batch.token_ids.shape[0] == 2
    assert batch.features.shape == (2,24)
    assert batch.attention_mask.dtype == torch.bool
    assert batch.token_values[1,-1].item() == 7.5


def test_masked_categorical_probabilities_and_gradient():
    torch = pytest.importorskip('torch')
    logits = torch.tensor([[0.,1000.,1.,-2.,3.,1000.,5.]], requires_grad=True)
    mask = torch.tensor([[True,False,True,True,True,False,True]])
    result = masked_logits(logits,mask)
    distribution = masked_categorical(logits,mask)
    probs = distribution.probs
    assert probs[0,1].item() == probs[0,5].item() == 0.0
    assert torch.allclose(probs.sum(-1),torch.ones(1))
    assert int(distribution.sample().item()) in (0,2,3,4,6)
    loss = -distribution.log_prob(torch.tensor([2])).sum()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0,1].item() == logits.grad[0,5].item() == 0.0
    assert result.shape == (1,7)


def test_masked_categorical_rejects_all_illegal():
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError):
        masked_categorical(torch.zeros(1,7),torch.zeros(1,7,dtype=torch.bool))
    with pytest.raises(ValueError):
        masked_logits(torch.zeros(1,6),torch.zeros(1,6,dtype=torch.bool))


def test_empty_batch_rejected():
    pytest.importorskip('torch')
    with pytest.raises(ValueError):
        batch_actors([])
    with pytest.raises(ValueError):
        batch_critics([])
