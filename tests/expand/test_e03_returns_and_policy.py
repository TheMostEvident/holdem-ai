import math
import random
from dataclasses import replace
import pytest
import torch
from holdem_ai.expand.information.schema import ACTION_NAMES
from holdem_ai.expand.information.encoder import encode_observation
from holdem_ai.expand.rl import (ActorCritic, OpponentMix, RolloutConfig, PPOConfig,
    MultiAgentCollector, prepare_gae, optimize_ppo)
from holdem_ai.expand.rl.population import scripted_action, choose_role
from e03_fake_engine import FastFakeEngine


def mk_model(seed=12):
    torch.manual_seed(seed)
    return ActorCritic(width=16,heads=2,layers=1)


def test_population_probabilities_and_guards():
    with pytest.raises(ValueError): OpponentMix(shared_policy_fraction=.7,snapshot_fraction=.7)
    mix=OpponentMix(shared_policy_fraction=0.3,snapshot_fraction=0.5)
    rng=random.Random(1)
    assert 'snapshot' in {choose_role(rng,mix,has_snapshots=True) for _ in range(50)}
    assert 'snapshot' not in {choose_role(rng,mix,has_snapshots=False) for _ in range(50)}


def test_rollout_all_2_to_6_and_reward_assignment():
    config=RolloutConfig(hands_per_update=5,num_players=(2,3,4,5,6),max_warmup_hands=100,
                         mix=OpponentMix(shared_policy_fraction=1,snapshot_fraction=0))
    c=MultiAgentCollector(mk_model(),config,engine_factory=FastFakeEngine)
    for n in range(2,7):
        result=c.collect(seed=100+n,policy_version='frozen-u0',num_hands=2,force_n=n)
        assert result.total_hands==2
        assert result.hands_by_n[n]==2
        assert result.illegal_actions==0
        assert result.transitions
        assert all(len(t.observation.seats)==6 for t in result.transitions)
        assert all(math.isfinite(t.action_logprob) for t in result.transitions)
        assert all(abs(sum(rewards))<1e-10 for rewards in result.hand_rewards)
        grouped={}
        for t in result.transitions:
            grouped.setdefault((t.session_id,t.hand_id,t.seat_id),[]).append(t)
        assert all(sum(t.terminal_hand for t in stream)==1 for stream in grouped.values())
        samples=prepare_gae(result.transitions)
        assert len(samples)==len(result.transitions)
        assert all(math.isfinite(s.advantage) for s in samples)


def test_missing_terminal_rejected():
    c=MultiAgentCollector(mk_model(),RolloutConfig(hands_per_update=1),engine_factory=FastFakeEngine)
    t=c.collect(seed=21,policy_version='u0',force_n=2).transitions
    with pytest.raises(ValueError,match='unterminated'):
        prepare_gae(tuple(replace(x,terminal_hand=False) for x in t))


def test_policy_mask_gradient_and_ppo_stats():
    model=mk_model()
    config=RolloutConfig(hands_per_update=2,num_players=(2,))
    data=MultiAgentCollector(model,config,engine_factory=FastFakeEngine).collect(seed=123,policy_version='u0')
    old={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-3)
    result=optimize_ppo(model,optimizer,prepare_gae(data.transitions),PPOConfig(epochs=1,minibatch_size=2))
    assert result['samples']==len(data.transitions)
    assert math.isfinite(result['entropy'])
    assert any(not torch.equal(old[k],new) for k,new in model.state_dict().items())
    t=data.transitions[0]
    probs=model.actor([t.encoded])[0].softmax(-1).detach().tolist()
    assert sum(probs)==pytest.approx(1)
    assert all(probs[i]==0 for i,x in enumerate(t.encoded.legal_mask) if not x)
    with pytest.raises(TypeError):model.value([{'hidden_cards':['cA','dA']}])


def test_deterministic_rollout_with_frozen_policy():
    c=MultiAgentCollector(mk_model(),RolloutConfig(hands_per_update=3),engine_factory=FastFakeEngine)
    x=c.collect(seed=414,policy_version='u0')
    y=c.collect(seed=414,policy_version='u0')
    assert [t.action for t in x.transitions]==[t.action for t in y.transitions]
    assert [t.reward_bb for t in x.transitions]==[t.reward_bb for t in y.transitions]
    assert x.hands_by_n==y.hands_by_n
