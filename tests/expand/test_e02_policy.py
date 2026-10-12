import pytest

torch = pytest.importorskip('torch')
from holdem_ai.expand.information import (make_information_state,encode_observation)
from holdem_ai.expand.information.policy_v2 import PolicyV2
from test_e02_state_and_actions import obs


def fixture():
    return encode_observation(make_information_state(
        obs(), big_blind_chips=100,
        seat_presence=(True,True,True,False,False,False)))


def test_e02_policy_logits_gradients_and_illegal_zero():
    torch.manual_seed(1)
    model = PolicyV2(width=32, heads=4, layers=1)
    x=fixture()
    logits=model([x,x])
    assert logits.shape==(2,7)
    probs=logits.softmax(-1)
    assert probs[0,1].item()==0.0
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-6)
    loss=-logits.log_softmax(-1)[:,3].mean()
    loss.backward()
    assert model.policy_head[-1].weight.grad is not None
    assert model.policy_head[-1].weight.grad.abs().sum().item()>0
    assert model.seat_project[0].weight.grad.abs().sum().item()>0
    assert model.history_project[0].weight.grad is None or model.history_project[0].weight.grad.abs().sum().item()==0


def test_e02_policy_seed_and_legal_sampling():
    model=PolicyV2(width=32,heads=4,layers=1)
    x=fixture()
    assert model.select_action(x,rng_seed=123)==model.select_action(x,rng_seed=123)
    assert model.select_action(x,rng_seed=123) != 'check'
    assert model.select_action(x,rng_seed=123,greedy=True) != 'check'
    with pytest.raises(ValueError):
        model.select_action(x,rng_seed=-1)


def test_e02_seat_permutation_equivariance():
    from dataclasses import replace
    # Rotation of physical labels by 2 with identical relative pos and
    # hero/global features preserves set-transformer output (no absolute IDs).
    x=fixture()
    rotated=replace(x,
        seat_features=x.seat_features[-2:]+x.seat_features[:-2],
        seat_mask=x.seat_mask[-2:]+x.seat_mask[:-2])
    model=PolicyV2(width=32,heads=4,layers=1)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model([x]),model([rotated]),atol=1e-5)


def test_e02_public_history_affects_policy_and_trains_gru():
    from dataclasses import replace
    from holdem_ai.expand.information.schema import PublicAction
    state=make_information_state(obs(),big_blind_chips=100,
      seat_presence=(True,True,True,False,False,False))
    raw=obs()
    raw['raw_obs']['action_history']=((1,1),)
    raw['raw_obs']['action_raise_to']=(None,)
    history=(PublicAction(actor_seat=1,street='preflop',kind='call',
              raise_to_bb=None,to_call_bb=0.5,pot_bb=2.5),)
    state2=make_information_state(raw,big_blind_chips=100,
      seat_presence=(True,True,True,False,False,False),history=history)
    h=encode_observation(state2)
    assert h.history_mask[0]
    model=PolicyV2(width=32,heads=4,layers=1)
    logits=model([h])
    assert torch.isfinite(logits[:,0]).all()
    (-logits.log_softmax(-1)[:,3]).sum().backward()
    assert model.history_project[0].weight.grad.abs().sum()>0
