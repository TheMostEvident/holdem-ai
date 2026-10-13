# Complete Python source volume 20/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## tests/expand/test_e04_real_engine.py

Original blob `75b70b607ab96e34fdae64dc4c785d025eb69480` (1393 bytes).

<a id="py-229"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e04_real_engine.py" sha="75b70b607ab96e34fdae64dc4c785d025eb69480" -->

````````python
"""Testable on Windows after installing actual Base Phase 3 engine."""
import random
import pytest
pytest.importorskip('holdem_ai.environment.multi_player_engine')
from holdem_ai.expand.pluribus.blueprint import HoldemGame,MCCFRSolver,MCCFRConfig,SearchBudgetExceeded


@pytest.mark.parametrize('n',[2,3,4,5,6])
def test_actual_holdem_engine_rules_and_chips(n):
    game=HoldemGame(n,stack_bb=3)
    state=game.start(random.Random(n))
    assert len(state.player_ids)==n
    steps=0
    while not state.is_terminal():
        steps+=1
        assert steps<256
        legal=state.legal_actions()
        assert legal
        assert isinstance(state.infoset(state.current_player),str)
        state=state.next_state(random.Random(100+n+steps).choice(legal))
    assert sum(state.terminal_payoffs())==pytest.approx(0)


def test_actual_engine_two_player_traversal():
    solver=MCCFRSolver(HoldemGame(2,stack_bb=2),MCCFRConfig(seed=400,max_nodes_per_iteration=50000))
    solver.train(2)
    assert solver.completed_iterations==2


def test_actual_engine_six_player_guard_not_silent():
    solver=MCCFRSolver(HoldemGame(6,stack_bb=2),MCCFRConfig(seed=404,max_nodes_per_iteration=1500))
    try:
        solver.iterate()
    except SearchBudgetExceeded:
        assert solver.completed_iterations==0
        assert not solver.store.entries
    else:
        assert solver.completed_iterations==1

````````

<!-- END_PY_SOURCE path="tests/expand/test_e04_real_engine.py" -->

## tests/expand/test_e04_toy_mccfr.py

Original blob `a9a18e22a831df1c502e0d5b7d4cd0c63a190ca3` (4100 bytes).

<a id="py-230"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e04_toy_mccfr.py" sha="a9a18e22a831df1c502e0d5b7d4cd0c63a190ca3" -->

````````python
import math
import random
import pytest
from holdem_ai.expand.pluribus.blueprint import (KuhnGame,LeducGame,MCCFRConfig,MCCFRSolver,
                                                   SearchBudgetExceeded)

@pytest.mark.parametrize('players',[2,3])
@pytest.mark.parametrize('name',[KuhnGame,LeducGame])
def test_toy_terminal_and_reward_zero_sum(name,players):
    game=name(players)
    for seed in range(10):
        state=game.start(random.Random(seed))
        seen=0
        while not state.is_terminal():
            seen+=1
            assert seen<25
            state=state.next_state(random.Random(seed+seen).choice(state.legal_actions()))
        assert len(state.terminal_payoffs())==players
        assert sum(state.terminal_payoffs())==pytest.approx(0)

@pytest.mark.parametrize('players',[2,3])
@pytest.mark.parametrize('name',[KuhnGame,LeducGame])
def test_mccfr_2_and_3_player_updates(name,players):
    solver=MCCFRSolver(name(players), MCCFRConfig(seed=12,max_nodes_per_iteration=10000))
    stats=solver.train(60)
    assert len(stats)==60 and solver.completed_iterations==60
    assert solver.store.entries
    for entry in solver.store.entries.values():
        assert all(math.isfinite(v) for v in entry['regret_sum'])
        assert sum(entry['strategy_sum'])>=0
        assert sum(solver.store.policy('',('a','b')))==pytest.approx(1)
    assert any(sum(e['strategy_sum'])>0 for e in solver.store.entries.values())


def test_infosets_hide_other_hole_and_future_community():
    s=KuhnGame().start(random.Random(10))
    from dataclasses import replace
    i=s.current_player
    cards=list(s.hole)
    cards[1-i]=99
    assert s.infoset(i)==replace(s,hole=tuple(cards)).infoset(i)
    led=LeducGame().start(random.Random(11))
    assert led.infoset(0)==replace(led,board=999).infoset(0)


def test_illegal_actions_rejected():
    st=KuhnGame().start(random.Random(1))
    with pytest.raises(ValueError):st.next_state('all_in')


def test_budget_error_does_not_change_store_or_iteration():
    solver=MCCFRSolver(KuhnGame(2), MCCFRConfig(max_nodes_per_iteration=1))
    with pytest.raises(SearchBudgetExceeded):solver.iterate()
    assert solver.completed_iterations==0 and not solver.store.entries


def test_budget_infosets_atomic():
    solver=MCCFRSolver(KuhnGame(2),MCCFRConfig(max_infosets=1))
    with pytest.raises(SearchBudgetExceeded):solver.iterate()
    assert solver.completed_iterations==0 and not solver.store.entries


def test_toy_linear_update_weights():
    solver=MCCFRSolver(KuhnGame(2),MCCFRConfig(seed=444))
    stats=solver.train(6)
    assert [e['weight'] for e in stats]==[1,1,2,2,3,3]


def test_uniform_and_average_distribution():
    solver=MCCFRSolver(KuhnGame(2))
    entry=solver.game.start(random.Random(3))
    key=entry.infoset(0)
    acts=entry.legal_actions()
    assert solver.store.policy(key,acts)==(0.5,0.5)
    solver.train(100)
    for k,e in solver.store.entries.items():
        for avg in (True,False):
            prob=solver.store.policy(k,tuple(e['actions']),average=avg)
            assert all(p>=0 for p in prob)
            assert sum(prob)==pytest.approx(1)


def test_same_seed_same_result_different_seed_changes():
    a=MCCFRSolver(KuhnGame(3),MCCFRConfig(seed=7))
    b=MCCFRSolver(KuhnGame(3),MCCFRConfig(seed=7))
    c=MCCFRSolver(KuhnGame(3),MCCFRConfig(seed=8))
    a.train(100); b.train(100); c.train(100)
    assert a.store.as_dict()==b.store.as_dict()
    assert a.store.as_dict()!=c.store.as_dict()


def test_kuhn_exact_best_response_improves():
    from holdem_ai.expand.pluribus.blueprint.kuhn_reference import kuhn_nash_conv
    solver=MCCFRSolver(KuhnGame(2),MCCFRConfig(seed=42))
    before=kuhn_nash_conv(solver.store)['nash_conv']
    solver.train(10000)
    after=kuhn_nash_conv(solver.store)['nash_conv']
    assert before==pytest.approx(11/12)
    assert 0 <= after < 0.05


def test_depth_budget_is_safe():
    solver=MCCFRSolver(KuhnGame(2),MCCFRConfig(max_depth=1))
    with pytest.raises(SearchBudgetExceeded):solver.iterate()
    assert solver.completed_iterations==0 and not solver.store.entries

````````

<!-- END_PY_SOURCE path="tests/expand/test_e04_toy_mccfr.py" -->

## tests/expand/test_e05_beliefs.py

Original blob `01584c20fe897a65d5774b007dc969a440c66417` (1406 bytes).

<a id="py-231"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e05_beliefs.py" sha="01584c20fe897a65d5774b007dc969a440c66417" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/expand/test_e05_beliefs.py" -->

## tests/expand/test_e05_cli.py

Original blob `0f4a5514e42fd45bbae35196353636d5a21aeacc` (595 bytes).

<a id="py-232"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e05_cli.py" sha="0f4a5514e42fd45bbae35196353636d5a21aeacc" -->

````````python
from argparse import Namespace
import json
import pytest
from scripts.expand_phase05_search import run

def test_cli_toy_json_and_no_overwrite(tmp_path):
    path=tmp_path/'out.json'
    args=Namespace(game='kuhn',players=2,checkpoint=None,seed=9,root_seed=10,
                   stack_bb=2,time_ms=1000,nodes=2500,memory_mb=64,iterations=10,depth=2,
                   output_json=str(path))
    result=run(args)
    assert result['iterations']>=1
    assert json.loads(path.read_text())['action_probabilities']==result['action_probabilities']
    with pytest.raises(FileExistsError):run(args)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e05_cli.py" -->

## tests/expand/test_e05_continuations.py

Original blob `b1adb846c3e81ea8e62cf69e9e0f994774ebfed5` (1417 bytes).

<a id="py-233"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e05_continuations.py" sha="b1adb846c3e81ea8e62cf69e9e0f994774ebfed5" -->

````````python
import pytest
from holdem_ai.expand.pluribus.search import CONTINUATIONS,continuation_policy

@pytest.mark.parametrize('variant',CONTINUATIONS)
def test_all_continuations_have_valid_legal_distribution(variant):
    acts=('fold','check','call','raise_half_pot','raise_pot','all_in')
    probs=continuation_policy(acts,[1]*len(acts),variant)
    assert set(probs)==set(acts)
    assert all(0<=v<=1 for v in probs.values())
    assert sum(probs.values())==pytest.approx(1)

def test_fivefold_weighting_is_over_original_mass():
    acts=('fold','call','raise_pot')
    b=[0.2,0.3,0.5]
    fold=continuation_policy(acts,b,'fold_bias')
    assert fold['fold']==pytest.approx(1/1.8)
    call=continuation_policy(acts,b,'call_bias')
    assert call['call']==pytest.approx(1.5/2.2)
    r=continuation_policy(acts,b,'raise_bias')
    assert r['raise_pot']==pytest.approx(2.5/3)
    assert continuation_policy(acts,b,'blueprint')['call']==pytest.approx(.3)

def test_raise_bias_affects_all_raise_sizes_not_check():
    p=continuation_policy(('check','raise_half_pot','raise_pot','all_in'),(1,1,1,1),'raise_bias')
    assert p['raise_half_pot']==p['raise_pot']==p['all_in']
    assert p['raise_pot']==pytest.approx(5*p['check'])

@pytest.mark.parametrize('bad',[(-1,1),(0,0),(float('nan'),1)])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError): continuation_policy(('fold','call'),bad,'blueprint')

````````

<!-- END_PY_SOURCE path="tests/expand/test_e05_continuations.py" -->

## tests/expand/test_e05_evaluate.py

Original blob `a80fcd07603d3c4f58a21ce2b8b0fa1759b4c8dc` (757 bytes).

<a id="py-234"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e05_evaluate.py" sha="a80fcd07603d3c4f58a21ce2b8b0fa1759b4c8dc" -->

````````python
from argparse import Namespace
import random
import pytest
from holdem_ai.expand.pluribus.blueprint import KuhnGame,MCCFRSolver,MCCFRConfig,save_checkpoint
from scripts.expand_phase05_evaluate import run

def test_evaluation_not_modifying_checkpoint(tmp_path):
    path=tmp_path/'blueprint.json.gz'
    solver=MCCFRSolver(KuhnGame(2),MCCFRConfig(seed=1))
    solver.train(50)
    save_checkpoint(path,solver)
    original=path.read_bytes()
    args=Namespace(game='kuhn',checkpoint=str(path),hands=3,seed=111,iterations=4,
                   time_ms=300,nodes=500,output_json=str(tmp_path/'out.json'))
    out=run(args)
    assert out['hands']==6
    assert out['search_calls']>0
    assert len(out['ci95_normal'])==2
    assert path.read_bytes()==original

````````

<!-- END_PY_SOURCE path="tests/expand/test_e05_evaluate.py" -->

## tests/expand/test_e05_holdem_deck.py

Original blob `3bbe41adc0b10940eb20c2e0622adbaac5e4f1f0` (1611 bytes).

<a id="py-235"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e05_holdem_deck.py" sha="3bbe41adc0b10940eb20c2e0622adbaac5e4f1f0" -->

````````python
import random
import pytest
from holdem_ai.expand.pluribus.search.belief import DECK,HoldemObservationBelief

@pytest.mark.parametrize('n',[2,3,4,5,6])
def test_card_layout_exact_hero_visible_board_and_no_collision(n):
    occupied=([0,2,3,4,5] if n<=5 else [0,1,2,3,4,5])[:n]
    button=occupied[0]
    hero=occupied[-1]
    raw={'player_id':hero,'button':button,'hand':['sA','cK'],
         'public_cards':['d2','h3','s4','c5','d6'],
         'street':'river','stacks':[40 if i in occupied else 0 for i in range(6)],
         'street_bets':[0]*6,'action_history':(), 'action_raise_to':()}
    s=HoldemObservationBelief(raw,initial_stacks=tuple(raw['stacks']),
                              small_blind=1,big_blind=2)
    for seed in range(3):
        cards=s._random_deck(random.Random(seed))
        assert len(cards)==52 and set(cards)==DECK and len(set(cards))==52
        dealing=[]
        pos=button
        for _ in occupied:
            pos=next((pos+d)%6 for d in range(1,7) if (pos+d)%6 in occupied)
            dealing.append(pos)
        index=dealing.index(hero)
        assert cards[index]=='sA'
        assert cards[n+index]=='cK'
        assert cards[2*n+1:2*n+4]==['d2','h3','s4']
        assert cards[2*n+5]=='c5' and cards[2*n+7]=='d6'


def test_card_collisions_and_bad_hole_rejected():
    x={'player_id':0,'button':0,'hand':['sA','sA'], 'public_cards':[],
       'stacks':[20,20,0,0,0,0], 'street_bets':[0]*6,'action_history':(),
       'action_raise_to':()}
    with pytest.raises(ValueError):
        HoldemObservationBelief(x,initial_stacks=(20,20,0,0,0,0),small_blind=1,big_blind=2)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e05_holdem_deck.py" -->

## tests/expand/test_e05_nested.py

Original blob `14700f44656c84ed80c63e488e64304661259b61` (885 bytes).

<a id="py-236"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e05_nested.py" sha="14700f44656c84ed80c63e488e64304661259b61" -->

````````python
import pytest
from holdem_ai.expand.pluribus.search import NestedSearchSession

def test_off_tree_re_root_and_committed_own_actions():
    s=NestedSearchSession(hero=0)
    s.observe_action(actor=1,street='preflop',action='raise_to',raise_to=75,abstract_actions=('fold','call','raise_pot'))
    assert s.off_tree_count==1
    s.observe_action(actor=0,street='preflop',action='call',infoset_key='hero:AA')
    with pytest.raises(ValueError):
        s.observe_action(actor=0,street='preflop',action='fold',infoset_key='hero:AA')
    s.observe_action(actor=1,street='flop',action='check')
    assert s.street_resets==1
    assert s.own_commitments[('preflop','hero:AA')]=='call'
    assert len(s.public_trace())==3

def test_reject_bad_off_tree_size():
    s=NestedSearchSession(0)
    with pytest.raises(ValueError):s.observe_action(actor=1,action='raise_to',street='flop',raise_to=-1)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e05_nested.py" -->

## tests/expand/test_e05_real_engine.py

Original blob `1051eb0cdb66f828eb017aa23dcd3cc11ef746b0` (2180 bytes).

<a id="py-237"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e05_real_engine.py" sha="1051eb0cdb66f828eb017aa23dcd3cc11ef746b0" -->

````````python
"""Runs only on user's real base Phase3 repository install."""
import random
import pytest
pytest.importorskip('holdem_ai.environment.multi_player_engine')
from holdem_ai.expand.pluribus.blueprint import HoldemGame
from holdem_ai.expand.pluribus.search import HoldemObservationBelief,BlueprintPolicy,SearchBudget,SubgameSolver

@pytest.mark.parametrize('n',[2,3,4,5,6])
def test_holdem_root_rebuild_and_legality(n):
    game=HoldemGame(n,stack_bb=2)
    st=game.start(random.Random(300+n))
    raw=st.engine.observe(st.current_player)['raw_obs']
    belief=HoldemObservationBelief(raw,initial_stacks=st.engine.initial_stacks,
                                   small_blind=st.engine.small_blind,big_blind=st.engine.big_blind)
    sample=belief.sample(random.Random(1000+n))
    actual=sample.engine.observe(sample.current_player)['raw_obs']
    assert actual['hand']==raw['hand']
    assert actual['public_cards']==raw['public_cards']
    solver=SubgameSolver(BlueprintPolicy(abstraction_id=game.abstraction.fingerprint),
             SearchBudget(max_time_ms=1000,max_iterations=3,max_nodes=300,depth=1),seed=77)
    out=solver.search(belief)
    assert out.action_probabilities
    assert set(out.action_probabilities)==set(st.legal_actions())
    assert sum(out.action_probabilities.values())==pytest.approx(1)


def test_replay_midstreet_offtree_raise():
    game=HoldemGame(3,stack_bb=20)
    st=game.start(random.Random(900))
    # Reconstruct a public history containing exact raise_to, conditional on
    # the acting hero's actually visible hole cards.
    e=st.engine
    e.step_raise_to(e.min_raise_to())
    while not e.is_over() and e.current_player!=st.player_ids[0]:
        e.step(1)
    if e.is_over() or e.current_player!=st.player_ids[0]:
        pytest.skip('seed produced no later turn for target seat')
    raw=e.observe(e.current_player)['raw_obs']
    belief=HoldemObservationBelief(raw,initial_stacks=e.initial_stacks,
                                   small_blind=e.small_blind,big_blind=e.big_blind)
    reconstructed=belief.sample(random.Random(101))
    assert reconstructed.engine.observe(reconstructed.current_player)['raw_obs']['pot']==raw['pot']

````````

<!-- END_PY_SOURCE path="tests/expand/test_e05_real_engine.py" -->

## tests/expand/test_e05_solver.py

Original blob `07ae3f973be2b48bff5249e244ae621cba6b10c9` (2866 bytes).

<a id="py-238"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e05_solver.py" sha="07ae3f973be2b48bff5249e244ae621cba6b10c9" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/expand/test_e05_solver.py" -->

## tests/expand/test_e06_adaptive.py

Original blob `5b733c4a6ded7ea0584a70a3e68204e4d2fabeeb` (3226 bytes).

<a id="py-239"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e06_adaptive.py" sha="5b733c4a6ded7ea0584a70a3e68204e4d2fabeeb" -->

````````python
import pytest
from holdem_ai.expand.opponent import (OpponentConfig,OpponentRegistry,PublicDecision,SafeAdaptivePolicy,
                                       GuardConfig,kl_divergence,tv_distance)

LEGAL=('fold','call','raise_pot')
BASE={'fold':.2,'call':.6,'raise_pot':.2}
PROPOSAL={'fold':.05,'call':.05,'raise_pot':.90}

def fitted(n=75):
    cfg=OpponentConfig(min_adapt_samples=12)
    r=OpponentRegistry(cfg);r.bind_seat(1,'opp','tenure-1')
    x=None
    for i in range(n):
        x=r.observe(PublicDecision('opp','tenure-1',f'h{i}',i,1,'preflop','button','4to10',LEGAL,'call'))
    return cfg,x

def test_no_evidence_preserves_base():
    gate=SafeAdaptivePolicy()
    ans=gate.adjust(BASE,PROPOSAL,None,legal=LEGAL)
    assert ans.probabilities==BASE and not ans.used_proposal

def test_low_samples_no_adapt():
    cfg,e=fitted(4)
    ans=SafeAdaptivePolicy(cfg).adjust(BASE,PROPOSAL,e,legal=LEGAL)
    assert ans.probabilities==BASE and ans.reason=='insufficient_evidence'

def test_trust_kl_tv_bounds_and_probability_simplex():
    cfg,e=fitted()
    limits=GuardConfig(kl_budget=.002,tv_budget=.011,max_mix=.8)
    ans=SafeAdaptivePolicy(cfg,limits).adjust(BASE,PROPOSAL,e,legal=LEGAL)
    assert ans.used_proposal
    assert 0<ans.mixture_weight<.8
    assert sum(ans.probabilities.values())==pytest.approx(1)
    assert kl_divergence(ans.probabilities,BASE)<=limits.kl_budget+1e-10
    assert tv_distance(ans.probabilities,BASE)<=limits.tv_budget+1e-10
    assert ans.probabilities['raise_pot']>BASE['raise_pot']

def test_zero_support_cannot_create_probability():
    cfg,e=fitted()
    base={'fold':0.,'call':1.,'raise_pot':0.}
    out=SafeAdaptivePolicy(cfg).adjust(base,PROPOSAL,e,legal=LEGAL)
    assert out.probabilities==base
    assert not out.used_proposal

def test_changepoint_fallback():
    cfg,e=fitted()
    from dataclasses import replace
    e=replace(e,shift_active=True)
    ans=SafeAdaptivePolicy(cfg).adjust(BASE,PROPOSAL,e,legal=LEGAL)
    assert ans.reason=='change_point_fallback'
    assert ans.probabilities==BASE

def test_no_proposal_unchanged_even_with_evidence():
    cfg,e=fitted()
    ans=SafeAdaptivePolicy(cfg).adjust(BASE,None,e,legal=LEGAL)
    assert ans.reason=='no_exploitation_proposal' and not ans.used_proposal

def test_illegal_action_or_invalid_distribution_rejected():
    cfg,e=fitted()
    with pytest.raises(ValueError):SafeAdaptivePolicy(cfg).adjust(BASE,{'fold':.1,'call':.9,'all_in':.0},e,legal=LEGAL)
    with pytest.raises(ValueError):SafeAdaptivePolicy(cfg).adjust(BASE,{'fold':.1,'call':-.9,'raise_pot':1.8},e,legal=LEGAL)

def test_context_mismatch_is_fallback():
    cfg,e=fitted()
    ans=SafeAdaptivePolicy(cfg).adjust({'fold':.5,'call':.5}, {'fold':.2,'call':.8},e,
        legal=('fold','call'),expected_opponent_context=('river','button','4to10',LEGAL))
    assert ans.reason=='opponent_context_mismatch'


def test_hero_legal_and_opponent_legal_may_differ():
    cfg,e=fitted()
    legal=('fold','call')
    ans=SafeAdaptivePolicy(cfg).adjust({'fold':.5,'call':.5}, {'fold':.1,'call':.9},e,legal=legal,
                                       expected_opponent_context=e.context)
    assert ans.used_proposal and set(ans.probabilities)==set(legal)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e06_adaptive.py" -->

## tests/expand/test_e06_bridge.py

Original blob `f54c57842e482cc7cb6b599ae7ae17f0ec1460fb` (2084 bytes).

<a id="py-240"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e06_bridge.py" sha="f54c57842e482cc7cb6b599ae7ae17f0ec1460fb" -->

````````python
import pytest
from holdem_ai.expand.opponent import PublicOpponentTracker,position_for_seat,count_prior_preflop_raises
from holdem_ai.expand.environment.table_state import HandHandle, PlayerSeat,TableSnapshot,TableConfig
from holdem_ai.expand.information.schema import PublicAction
from test_e06_state_bridge import state_with_hole


def hand(players=((0,'a'),(1,'b'),(2,'c'),(3,'d'),(4,'e'),(5,'f'))):
    return HandHandle('hand1',1,0,1,2,players,77,False)

def test_positions_six_and_four():
    h=hand()
    assert [position_for_seat(h,i) for i in range(6)]==[
       'button','small_blind','big_blind','early','middle','cutoff']
    h4=hand(((0,'a'),(1,'b'),(2,'c'),(5,'f')))
    assert position_for_seat(h4,5)=='cutoff'

def test_prior_raises_counts_exact_raise_to():
    a=PublicAction(1,'preflop','raise_to',3.5,1.,1.5)
    b=PublicAction(2,'flop','raise_pot',7.,0.,3.)
    assert count_prior_preflop_raises((a,b))==1

def test_registry_tracks_join_tenures_from_snapshot():
    snap=TableSnapshot(TableConfig(),tuple(PlayerSeat(i,'a' if i==0 else 'b' if i==1 else None,
       10000 if i<2 else 0,'seated' if i<2 else 'vacant') for i in range(6)),0,3,'session',False,None,0,20000)
    tracker=PublicOpponentTracker()
    tracker.sync_seats(snap,{'a':'e1','b':'e2'})
    assert tracker.registry.active_by_seat[0]==('a','e1')
    tracker.sync_seats(snap,{'a':'e99','b':'e2'})
    assert tracker.registry.active_by_seat[0]==('a','e99')
    assert ('a','e1') in tracker.registry.models

def test_accepted_action_only_after_caller_commits():
    tracker=PublicOpponentTracker()
    tracker.registry.bind_seat(0,'a','e1')
    state=state_with_hole(('sA','hA'))
    h=hand(((0,'a'),(1,'b')))
    output=tracker.record_accepted_action(state,handle=h,player_id='a',player_session_id='e1',
                                          action_index=0,action='fold')
    assert output.raw_events==1
    with pytest.raises(ValueError):tracker.record_accepted_action(state,handle=h,player_id='b',
                                           player_session_id='e1',action_index=1,action='fold')

````````

<!-- END_PY_SOURCE path="tests/expand/test_e06_bridge.py" -->
