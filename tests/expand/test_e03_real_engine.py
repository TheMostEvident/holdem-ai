"""Real 2-6 seat engine integration: deferred until the Windows full repo."""
import pytest
pytest.importorskip('holdem_ai.environment.multi_player_engine')

from holdem_ai.expand.rl import (ActorCritic, RolloutConfig, OpponentMix,
                                  MultiAgentCollector, prepare_gae)


@pytest.mark.parametrize('n',(2,3,4,5,6))
def test_e03_full_hands_real_base_engine(n):
    import torch
    torch.manual_seed(3)
    actor=ActorCritic(width=16,heads=2,layers=1)
    collector=MultiAgentCollector(actor,
       RolloutConfig(hands_per_update=2,num_players=(n,),max_warmup_hands=100,
                     mix=OpponentMix(shared_policy_fraction=0,snapshot_fraction=0,
                         scripted_names=('calling_station',))))
    result=collector.collect(seed=40000+n,policy_version='real-u0')
    assert result.hands_by_n[n]==2
    assert result.illegal_actions==0
    assert all(abs(sum(x))<1e-8 for x in result.hand_rewards)
    assert len(prepare_gae(result.transitions))>0
