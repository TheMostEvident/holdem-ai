from dataclasses import replace
import random
import pytest
from holdem_ai.expand.pluribus.blueprint import KuhnGame,LeducGame
from holdem_ai.expand.pluribus.search import ToyRootBelief,InconsistentPublicHistory

@pytest.mark.parametrize('name', [KuhnGame,LeducGame])
def test_toy_belief_fix_hero_and_exclude_collisions(name):
    game=name(2)
    initial=game.start(random.Random(55))
    hero=initial.current_player
    own=initial.hole[hero] if game.name=='kuhn' else initial.hole[hero]//2
    belief=ToyRootBelief(game,hero,own)
    states=[belief.sample(random.Random(seed)) for seed in range(20)]
    for state in states:
        assert state.current_player==hero
        assert len(set(state.hole))==2
        assert (state.hole[hero] if game.name=='kuhn' else state.hole[hero]//2)==own
    # No reference to the original other player's private card is required.
    assert belief.fingerprint==belief.fingerprint

def test_replay_history_only_public_actions():
    game=KuhnGame(2)
    root=game.start(random.Random(10))
    hero=1
    sampled=ToyRootBelief(game,hero,root.hole[hero],history=('check',)).sample(random.Random(18))
    assert sampled.current_player==1
    assert sampled.history[0][-1]=='check'

def test_reject_impossible_trace():
    b=ToyRootBelief(KuhnGame(2),0,0,history=('raise_pot',),max_attempts=4)
    with pytest.raises(InconsistentPublicHistory):b.sample(random.Random(1))
