import copy
import math
import random
import pytest
from holdem_ai.expand.pluribus.blueprint import KuhnGame,LeducGame,MCCFRSolver,MCCFRConfig
from holdem_ai.expand.pluribus.search import (BlueprintPolicy,SubgameSolver,SearchBudget,ToyRootBelief)


def belief(game,seed=17):
    state=game.start(random.Random(seed))
    hero=state.current_player
    private=state.hole[hero] if game.name=='kuhn' else state.hole[hero]//2
    return ToyRootBelief(game,hero,private)

@pytest.mark.parametrize('name',[KuhnGame,LeducGame])
def test_valid_probabilities_and_blueprint_unchanged(name):
    game=name(2)
    base=MCCFRSolver(game,MCCFRConfig(seed=77))
    base.train(200)
    snapshot=copy.deepcopy(base.store.as_dict())
    cfg=SearchBudget(max_iterations=30,max_time_ms=4000,max_nodes=6000,depth=2)
    out=SubgameSolver(BlueprintPolicy(base.store),cfg,seed=51).search(belief(game))
    assert out.status in ('solved','timeout','resource_guard')
    assert out.iterations>0
    assert out.nodes>0
    assert all(math.isfinite(v) and v>=0 for v in out.action_probabilities.values())
    assert sum(out.action_probabilities.values())==pytest.approx(1.)
    assert base.store.as_dict()==snapshot


def test_reproducible_with_seed_and_large_budget():
    game=KuhnGame(2)
    cfg=SearchBudget(max_time_ms=10000,max_iterations=40,max_nodes=10000,depth=2)
    a=SubgameSolver(BlueprintPolicy(),cfg,seed=41).search(belief(game))
    b=SubgameSolver(BlueprintPolicy(),cfg,seed=41).search(belief(game))
    assert a.status==b.status=='solved'
    assert a.nodes==b.nodes and a.action_probabilities==b.action_probabilities
    assert a.continuation_samples==b.continuation_samples


def test_node_cap_fallback_to_blueprint():
    game=KuhnGame(2)
    base=BlueprintPolicy()
    out=SubgameSolver(base,SearchBudget(max_nodes=1,max_iterations=30),seed=1).search(belief(game))
    assert out.status=='resource_guard' and out.iterations==0
    assert out.action_probabilities=={'check':0.5,'bet':0.5}


def test_time_budget_always_valid():
    cfg=SearchBudget(max_time_ms=1,max_iterations=10000,max_nodes=50000,depth=3)
    out=SubgameSolver(BlueprintPolicy(),cfg,seed=2).search(belief(KuhnGame(2)))
    assert out.status in ('timeout','solved','resource_guard')
    assert sum(out.action_probabilities.values())==pytest.approx(1)


def test_3_player_multiway_toy():
    game=KuhnGame(3)
    out=SubgameSolver(BlueprintPolicy(),SearchBudget(max_time_ms=10000,max_iterations=12,
                             max_nodes=5000,depth=2),seed=3).search(belief(game))
    assert out.iterations==12
    assert sum(out.action_probabilities.values())==pytest.approx(1)


def test_abstraction_mismatch_uses_only_valid_actions():
    game=KuhnGame(2)
    base=BlueprintPolicy()
    assert base.policy('missing',('fold','call','raise_to_300')) == dict.fromkeys(('fold','call','raise_to_300'),1/3)
