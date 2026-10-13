# Complete Python source volume 21/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## tests/expand/test_e06_cli.py

Original blob `c215129d87babef830400232d7a5874daf986196` (991 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e06_cli.py" sha="c215129d87babef830400232d7a5874daf986196" -->

````````python
import argparse,json
import pytest
from scripts.expand_phase06_benchmark import run


def test_synthetic_blocked_benchmark(tmp_path):
    args=argparse.Namespace(blocks=4,warmup=80,post_shift=80,initial_style='tight',new_style='aggressive',seed=71006,
                            output_json=str(tmp_path/'out.json'),checkpoint=str(tmp_path/'stats.json'))
    out=run(args)
    assert out['blocks']==4
    assert len(out['rows'])==4
    assert out['max_observed_kl']<=.025+1e-9
    assert out['max_observed_tv']<=.075+1e-9
    assert (tmp_path/'stats.json').is_file()
    assert out['shift_detected_blocks']>=1
    with pytest.raises(FileExistsError):run(args)


def test_determinism(tmp_path):
    args=argparse.Namespace(blocks=3,warmup=50,post_shift=50,initial_style='tight',new_style='aggressive',seed=71006,
                            output_json=str(tmp_path/'out.json'),checkpoint=None)
    a=run(args)
    args.output_json=str(tmp_path/'again.json')
    b=run(args)
    assert a==b

````````

<!-- END_PY_SOURCE path="tests/expand/test_e06_cli.py" -->

## tests/expand/test_e06_models.py

Original blob `dff4941daf0d6b723bd62d0be6204ee2aa3389b5` (4350 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e06_models.py" sha="dff4941daf0d6b723bd62d0be6204ee2aa3389b5" -->

````````python
import math
import pytest
from holdem_ai.expand.opponent import OpponentConfig,OpponentRegistry, PublicDecision, spr_bin

A=('fold','call','raise_pot')

def event(i,action='call',*,player='alice',tenure='join1',seat=2,street='preflop',position='button',legal=A,hand=None,pre=0):
    return PublicDecision(player,tenure,hand or f'h{i}',i,seat,street,position,'4to10',legal,action,pre)

def test_spr_buckets_and_validation():
    assert [spr_bin(x) for x in (0,1,4,10)]==['lt1','1to4','4to10','ge10']
    for v in (-1,math.inf,math.nan):
        with pytest.raises(ValueError):spr_bin(v)
    with pytest.raises(ValueError):event(1,'all_in')
    with pytest.raises(ValueError):event(1,legal=('call','fold'))
    with pytest.raises(ValueError):event(1,seat=6)

def test_session_isolation_across_same_seat():
    r=OpponentRegistry()
    r.bind_seat(2,'alice','join1')
    for i in range(40):r.observe(event(i,'fold'))
    old=r.evidence(2,event(1).context)
    assert old.posterior['fold']>.95
    r.unbind_seat(2)
    r.bind_seat(2,'bob','join2')
    fresh=r.evidence(2,event(1).context)
    assert fresh.raw_events==0 and fresh.posterior['fold']==pytest.approx(1/3)
    with pytest.raises(ValueError):r.observe(event(41,'fold'))
    r.unbind_seat(2)
    r.bind_seat(2,'alice','join3')
    assert r.evidence(2,event(1).context).raw_events==0
    assert ('alice','join1') in r.models

def test_no_seat_lease_overlap():
    r=OpponentRegistry();r.bind_seat(0,'a','t1')
    with pytest.raises(ValueError):r.bind_seat(1,'a','t1')

def test_legal_context_and_street_conditioned():
    r=OpponentRegistry();r.bind_seat(2,'alice','join1')
    for i in range(25):r.observe(event(i,'fold'))
    alt=event(27,'call',street='river')
    assert r.evidence(2,alt.context).raw_events==0
    alt2=event(28,'call',legal=('fold','call'))
    assert r.evidence(2,alt2.context).raw_events==0
    alt3=event(29,'call',position='early')
    assert r.evidence(2,alt3.context).raw_events==0

def test_duplicate_decisions_fail_closed():
    r=OpponentRegistry();r.bind_seat(2,'alice','join1')
    x=event(1)
    r.observe(x)
    with pytest.raises(ValueError):r.observe(x)

def test_dirichlet_smoothing_and_decay():
    r=OpponentRegistry();r.bind_seat(2,'alice','join1')
    x=event(0)
    assert r.evidence(2,x.context).posterior['fold']==pytest.approx(1/3)
    for i in range(55):r.observe(event(i,'fold'))
    p=r.evidence(2,x.context).posterior
    assert p['fold']>p['call']>0
    assert sum(p.values())==pytest.approx(1)
    assert r.evidence(2,x.context).effective_recent<r.evidence(2,x.context).effective_long

def test_style_shift_detection_and_recovery():
    cfg=OpponentConfig(long_decay=.999,recent_decay=.89,min_long=25,min_recent=4,
                       shift_kl_threshold=.08,shift_consecutive=2,shift_hold_events=12)
    r=OpponentRegistry(cfg);r.bind_seat(2,'alice','join1')
    active_before=False
    for i in range(110):
        out=r.observe(event(i,'fold'))
        active_before |= out.shift_active
    assert not active_before
    signals=[]
    for i in range(110,185):
        out=r.observe(event(i,'raise_pot'))
        signals.append(out.shift_active)
    assert any(signals), 'must detect large probability shift'
    assert out.shift_events>0

def test_vpip_pfr_3bet_and_multi_decisions_one_hand():
    r=OpponentRegistry();r.bind_seat(2,'alice','join1')
    r.observe(event(1,'call',hand='h1'))
    r.observe(event(2,'raise_pot',hand='h1',pre=1))
    r.models[('alice','join1')].end_hand('h1')
    r.models[('alice','join1')].end_hand('h1')
    r.observe(event(3,'fold',hand='h2'))
    r.models[('alice','join1')].end_hand('h2')
    out=r.models[('alice','join1')].diagnostics()
    assert out['preflop_hands']==2
    assert out['vpip_rate']==pytest.approx(.5)
    assert out['pfr_rate']==pytest.approx(.5)
    assert out['threebet_rate']==pytest.approx(.5)

def test_capacity_no_silent_eviction():
    cfg=OpponentConfig(max_event_ids=2)
    r=OpponentRegistry(cfg);r.bind_seat(2,'alice','join1')
    r.observe(event(0));r.observe(event(1))
    with pytest.raises(RuntimeError):r.observe(event(2))

def test_model_does_not_contain_opponent_private_cards():
    r=OpponentRegistry();r.bind_seat(2,'alice','join1')
    r.observe(event(1))
    assert not any(k in r.models[('alice','join1')].__dict__ for k in ('hole','deck','board','critic'))

````````

<!-- END_PY_SOURCE path="tests/expand/test_e06_models.py" -->

## tests/expand/test_e06_persistence.py

Original blob `1fe169488bf1f9b3c0b535cebec5701a6f35c4ac` (1702 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e06_persistence.py" sha="1fe169488bf1f9b3c0b535cebec5701a6f35c4ac" -->

````````python
import json
import pytest
from holdem_ai.expand.opponent import OpponentRegistry,PublicDecision,save_registry,load_registry


def test_checkpoint_roundtrip_resume(tmp_path):
    r=OpponentRegistry();r.bind_seat(3,'bob','join-123')
    for i in range(40):
        r.observe(PublicDecision('bob','join-123',f'h{i}',i,3,'flop','middle','ge10',('fold','call'),'call'))
    file=tmp_path/'stats.json'
    save_registry(file,r)
    restored=load_registry(file)
    assert restored.evidence(3,('flop','middle','ge10',('fold','call'))).posterior==pytest.approx(
        r.evidence(3,('flop','middle','ge10',('fold','call'))).posterior)
    assert restored.active_by_seat==r.active_by_seat
    with pytest.raises(ValueError):restored.observe(PublicDecision('bob','join-123','h2',2,3,'flop','middle','ge10',('fold','call'),'call'))
    restored.observe(PublicDecision('bob','join-123','new',41,3,'flop','middle','ge10',('fold','call'),'fold'))
    assert restored.evidence(3,('flop','middle','ge10',('fold','call'))).raw_events==41
    with pytest.raises(FileExistsError):save_registry(file,r)

def test_checkpoint_tamper_detection(tmp_path):
    r=OpponentRegistry();r.bind_seat(1,'id','tenure')
    target=tmp_path/'registry.json';save_registry(target,r)
    text=target.read_text();target.write_text(text.replace('"player_id": "id"','"player_id": "evil"'))
    with pytest.raises(ValueError):load_registry(target)

def test_checkpoint_contains_no_card_or_deck(tmp_path):
    r=OpponentRegistry();r.bind_seat(1,'id','tenure')
    target=tmp_path/'registry.json';save_registry(target,r)
    payload=json.loads(target.read_text())['payload']
    assert set(payload)=={'version','config','active_seats','models'}

````````

<!-- END_PY_SOURCE path="tests/expand/test_e06_persistence.py" -->

## tests/expand/test_e06_real_engine.py

Original blob `1367ee49057e39736e899bcbc850693f11c2cb36` (1922 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e06_real_engine.py" sha="1367ee49057e39736e899bcbc850693f11c2cb36" -->

````````python
"""Runs only once Base Phase 3 engine is present in user's real repository."""
import pytest
pytest.importorskip('holdem_ai.environment.multi_player_engine')
from holdem_ai.expand.environment import DynamicTable
from holdem_ai.expand.information import Phase2TableAdapter
from holdem_ai.expand.opponent import OpponentRegistry,public_decision_from_state


def test_public_event_hook_with_real_hand_engine():
    table=DynamicTable()
    tenures={}
    for i in range(4):
        tenures[f'p{i}']=table.request_join(f'p{i}',10000,i)
    bridge=Phase2TableAdapter(table)
    reg=OpponentRegistry()
    used=0
    for hand_seed in range(100,115):
        hand=bridge.start_hand(hand_seed)
        assert hand is not None
        for seat,pid in hand.participants:
            if seat not in reg.active_by_seat:reg.bind_seat(seat,pid,tenures[pid])
        for action_index in range(180):
            if table.current_player_id is None:break
            info=bridge.observe()
            action='check' if info.legal_actions.allows('check') else 'call'
            pid=table.current_player_id
            event=public_decision_from_state(info,player_id=pid,player_session_id=tenures[pid],
                hand_id=hand.hand_id,action_index=action_index,action=action,
                position=('button' if info.hero_seat==hand.button_seat else
                    'small_blind' if info.hero_seat==hand.small_blind_seat else
                    'big_blind' if info.hero_seat==hand.big_blind_seat else 'other'))
            reg.observe(event)
            bridge.act_abstract(action)
            used+=1
        else:raise AssertionError('engine decisions exceeded')
        result=bridge.settle_hand()
        assert sum(result.payoffs_chips)==0
        for seat,pid in hand.participants:reg.models[(pid,tenures[pid])].end_hand(hand.hand_id)
    assert used>0
    assert sum(len(m.seen_events) for m in reg.models.values())==used

````````

<!-- END_PY_SOURCE path="tests/expand/test_e06_real_engine.py" -->

## tests/expand/test_e06_router.py

Original blob `4a30bbaad60b00c1e946f639fbd187bdd6b19ac2` (1797 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e06_router.py" sha="4a30bbaad60b00c1e946f639fbd187bdd6b19ac2" -->

````````python
import pytest
from holdem_ai.expand.opponent import (OpponentRegistry,OpponentConfig,PublicDecision,StyleExpertRouter,
                                       SafeAdaptivePolicy)

OPP_LEGAL=('fold','call','raise_pot')
HERO_LEGAL=('check','raise_half_pot','raise_pot')
EXPERTS={
 'fold_heavy':{'check':.1,'raise_half_pot':.3,'raise_pot':.6},
 'calling':{'check':.2,'raise_half_pot':.6,'raise_pot':.2},
 'aggressive':{'check':.65,'raise_half_pot':.30,'raise_pot':.05},
}

def fitted(action):
    cfg=OpponentConfig(min_adapt_samples=12)
    reg=OpponentRegistry(cfg);reg.bind_seat(4,'a','tenure1')
    evidence=None
    for i in range(65):
        evidence=reg.observe(PublicDecision('a','tenure1',f'h{i}',i,4,'flop','cutoff','1to4',OPP_LEGAL,action))
    return cfg,evidence

@pytest.mark.parametrize('action,style', [('fold','fold_heavy'),('call','calling'),('raise_pot','aggressive')])
def test_router_uses_different_experts_for_styles(action,style):
    cfg,evidence=fitted(action)
    route=StyleExpertRouter(cfg).propose(evidence,EXPERTS,hero_legal=HERO_LEGAL)
    assert route.style==style and route.reason=='routed_existing_expert'
    assert route.proposal==EXPERTS[style]
    base={'check':.6,'raise_half_pot':.2,'raise_pot':.2}
    # Hero and opponent legal action signatures differ, which MUST be allowed.
    guarded=SafeAdaptivePolicy(cfg).adjust(base,route.proposal,evidence,legal=HERO_LEGAL)
    assert guarded.used_proposal
    assert sum(guarded.probabilities.values())==pytest.approx(1)

def test_router_uncertain_fallback():
    assert StyleExpertRouter().propose(None,EXPERTS,hero_legal=HERO_LEGAL).proposal is None
    cfg,e=fitted('fold')
    route=StyleExpertRouter(cfg).propose(e,{},hero_legal=HERO_LEGAL)
    assert route.reason=='missing_expert' and route.proposal is None

````````

<!-- END_PY_SOURCE path="tests/expand/test_e06_router.py" -->

## tests/expand/test_e06_state_bridge.py

Original blob `c2e0d68de00bf2d3cd2529bc0f1bef04c7d1edda` (1643 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e06_state_bridge.py" sha="c2e0d68de00bf2d3cd2529bc0f1bef04c7d1edda" -->

````````python
"""Public-information bridge test: Phase2 private hero hole must never pass to Phase6."""
from dataclasses import replace
from holdem_ai.expand.information.schema import (InformationState,PublicSeatFeatures,LegalActionSpec)
from holdem_ai.expand.opponent import public_decision_from_state,OpponentRegistry


def state_with_hole(hole):
    seats=tuple(PublicSeatFeatures(occupied=i<2,dealt_in=i<2,folded=False,all_in=False,
                seat_status='active' if i<2 else 'out',table_status='seated' if i<2 else 'vacant',
                stack_bb=100 if i<2 else 0, invested_bb=0,street_bet_bb=0,
                relative_position=i,is_hero=i==0) for i in range(6))
    mask=(True,False,True,False,True,False,False)
    spec=LegalActionSpec(mask,(None,)*7,100,10000,100)
    return InformationState(hero_seat=0,hero_hole=hole,board=(),street='preflop',button_seat=0,
            seats=seats,action_history=(),legal_actions=spec,pot_bb=1.5,to_call_bb=1.0,
            big_blind_chips=100,num_dealt_in=2)


def test_public_projection_ignores_private_hole():
    a=public_decision_from_state(state_with_hole(('sA','hA')),player_id='a',
        player_session_id='join1',hand_id='h1',action_index=0,action='fold',position='button')
    b=public_decision_from_state(state_with_hole(('c2','d7')),player_id='a',
        player_session_id='join1',hand_id='h1',action_index=0,action='fold',position='button')
    assert a==b
    assert 'sA' not in repr(a) and 'c2' not in repr(b)
    assert a.legal==('fold','call','raise_pot')
    r=OpponentRegistry();r.bind_seat(0,'a','join1')
    r.observe(a)
    assert 'hole' not in repr(r.models[('a','join1')])

````````

<!-- END_PY_SOURCE path="tests/expand/test_e06_state_bridge.py" -->

## tests/expand/test_e07_agent.py

Original blob `4221f8954d5de530d16ca860876aeca09c6ea74c` (3183 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e07_agent.py" sha="4221f8954d5de530d16ca860876aeca09c6ea74c" -->

````````python
import pytest
from holdem_ai.expand.integration import IntegratedPokerAgent,RoutingConfig,uniform_legal
from holdem_ai.expand.information import make_information_state


def state():
    raw={'legal_actions':{0:None,1:None},'raw_obs':{'hand':['cA','hK'],
       'public_cards':[],'street':'preflop','button':0,'player_id':0,
       'stacks':(10000,10000,0,0,0,0), 'street_bets':(0,0,0,0,0,0),
       'total_contributions':(0,0,0,0,0,0),'status':('active','active','out','out','out','out'),
       'pot':0,'to_call':0,'min_raise_to':100,'action_history':(), 'action_raise_to':(),
       'future_deck':['sA'],'opponent_holes':[('hA','dA')]}}
    return make_information_state(raw,big_blind_chips=100,
                  seat_presence=(True,True,False,False,False,False))


def test_e07_uniform_reproducible_and_legal():
    a=IntegratedPokerAgent(uniform_legal)
    obs=state()
    assert a.act(obs,123)==a.act(obs,123)
    assert a.act(obs,123).action.kind in ('fold','check')
    assert set(a.act(obs,123).probabilities)=={'fold','check'}


def test_e07_priority_and_fallback():
    calls=[]
    def src(name,good):
        def f(obs):
            calls.append(name)
            return {'fold':.2,'check':.8} if good else {'call':1}
        return f
    a=IntegratedPokerAgent(src('neural',True),blueprint=src('blueprint',True),
        search=src('search',False),config=RoutingConfig(use_search=True,use_blueprint=True,
        verified_search=True,verified_blueprint=True))
    d=a.act(state(),123)
    assert d.source=='blueprint' and calls==['search','blueprint']
    assert 'search:invalid_or_failed:ValueError' in d.fallback_reasons


def test_e07_unverified_abstraction_never_executed():
    def unsafe(_):raise AssertionError('unverified source MUST NOT be invoked')
    a=IntegratedPokerAgent(uniform_legal,search=unsafe,
         config=RoutingConfig(use_search=True))
    d=a.act(state(),2)
    assert d.source=='neural'
    assert 'search:unverified_abstraction' in d.fallback_reasons


def test_e07_untrusted_or_privileged_input_rejected():
    a=IntegratedPokerAgent(uniform_legal)
    with pytest.raises(TypeError):a.act({'raw_obs':{'opponent_holes':'secret'}},1)
    with pytest.raises(ValueError):a.act(state(),-1)


def test_e07_fallback_uniform_when_neural_bad():
    a=IntegratedPokerAgent(lambda obs: {'not_legal':1})
    d=a.act(state(),123)
    assert d.source=='uniform_legal_emergency'
    assert set(d.probabilities)=={'fold','check'}


def test_e07_adaptation_without_proven_experts_keeps_base():
    a=IntegratedPokerAgent(uniform_legal,config=RoutingConfig(use_adaptation=True))
    d=a.act(state(),123)
    assert d.adaptation_reason.startswith('no_evidence')
    assert d.probabilities==uniform_legal(state())


def test_e07_neural_fallback_reserved_after_two_failed_optional_sources():
    def bad(obs):raise TimeoutError('timeout')
    a=IntegratedPokerAgent(uniform_legal,search=bad,blueprint=bad,
        config=RoutingConfig(use_search=True,use_blueprint=True,
                             verified_search=True,verified_blueprint=True,max_policy_calls=2))
    d=a.act(state(),91)
    assert d.source=='neural'
    assert len(d.fallback_reasons)==2

````````

<!-- END_PY_SOURCE path="tests/expand/test_e07_agent.py" -->

## tests/expand/test_e07_benchmark.py

Original blob `c572cc6e9f115ab5e595dbd56762331b1098852b` (1777 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e07_benchmark.py" sha="c572cc6e9f115ab5e595dbd56762331b1098852b" -->

````````python
from dataclasses import replace
import pytest
from holdem_ai.expand.integration import IntegratedPokerAgent,uniform_legal
from holdem_ai.expand.integration.benchmark import BenchmarkConfig,cash_benchmark,tournament_benchmark,paired_cash_difference
from e03_fake_engine import FastFakeEngine
from e07_fake_tournament import KnockoutEngine


def test_e07_cash_scenarios_and_independent_clustering():
    cfg=BenchmarkConfig(groups=2,players=(2,3),stacks_bb=(20,),opponents=('calling_station',))
    agent=IntegratedPokerAgent(uniform_legal)
    r=cash_benchmark(agent,cfg,engine_factory=FastFakeEngine)
    r2=cash_benchmark(agent,cfg,engine_factory=FastFakeEngine)
    assert r==r2
    assert r['scored_hands']==4
    assert r['overall_bb100']['n_independent_sessions']==4
    assert paired_cash_difference(r,r)['paired_difference_bb100']['mean']==0


def test_e07_cash_paired_rejects_different_seed():
    a=IntegratedPokerAgent(uniform_legal)
    config=BenchmarkConfig(groups=2,players=(2,),stacks_bb=(20,),opponents=('tight',))
    x=cash_benchmark(a,config,engine_factory=FastFakeEngine)
    y=cash_benchmark(a,replace(config,seed=4),engine_factory=FastFakeEngine)
    with pytest.raises(ValueError):paired_cash_difference(x,y)


def test_e07_tournament_sessions_and_payouts():
    cfg=BenchmarkConfig(mode='tournament',groups=2,players=(2,3),
                        opponents=('calling_station',),max_tournament_hands=20)
    r=tournament_benchmark(IntegratedPokerAgent(uniform_legal),cfg,engine_factory=KnockoutEngine)
    assert r['tournaments']==4
    assert r['overall_prize_ev']['n_independent_sessions']==4
    assert all(x['win_rate'] in (0,.5,1.) for x in r['strata'])


def test_e07_invalid_groups():
    with pytest.raises(ValueError):BenchmarkConfig(groups=1)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e07_benchmark.py" -->

## tests/expand/test_e07_bridges_and_cli.py

Original blob `9f0cf98d2052a7e9c4801dfb95214b2332c35341` (2195 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e07_bridges_and_cli.py" sha="9f0cf98d2052a7e9c4801dfb95214b2332c35341" -->

````````python
import argparse
import json
from types import SimpleNamespace
import pytest

from holdem_ai.expand.integration import from_e05_solver,from_e04_checkpoint
from test_e07_agent import state
from scripts.expand_phase07_benchmark import run,parser
from e03_fake_engine import FastFakeEngine


def test_e07_search_source_rejects_timeout_without_mask_override():
    obs=state()
    class Solver:
        def search(self,root):
            return SimpleNamespace(status='timeout',action_probabilities={'fold':.5,'check':.5})
    source=from_e05_solver(Solver(),verified_root_belief=lambda state:'belief')
    with pytest.raises(TimeoutError):source(obs)


def test_e07_search_source_rejects_nonmatching_action_support():
    class Solver:
        def search(self,root):
            return SimpleNamespace(status='solved',action_probabilities={'call':1.})
    source=from_e05_solver(Solver(),verified_root_belief=lambda state:'belief')
    with pytest.raises(ValueError):source(state())


def test_e07_missing_checkpoint_refuses_fake_blueprint(tmp_path):
    with pytest.raises(FileNotFoundError):from_e04_checkpoint(str(tmp_path/'missing.json.gz'),exact_infoset_key=lambda s:'x')


def test_e07_cli_validation_adaptive_requires_expert(tmp_path):
    args=parser().parse_args(['--groups','2','--players','2','--adaptive',
                              '--output-json',str(tmp_path/'out.json')])
    with pytest.raises(ValueError,match='requires|needs'):run(args)


def test_e07_cli_completes_fake_cash_with_manifest(tmp_path,monkeypatch):
    import scripts.expand_phase07_benchmark as cli
    old_cash=cli.cash_benchmark
    def fake_cash(agent,cfg):return old_cash(agent,cfg,engine_factory=FastFakeEngine)
    monkeypatch.setattr(cli,'cash_benchmark',fake_cash)
    output=tmp_path/'bench.json'
    args=parser().parse_args(['--groups','2','--players','2','--stacks-bb','20',
        '--opponents','calling_station','--seed','42','--output-json',str(output)])
    report=run(args)
    assert output.exists()
    assert report['cash']['scored_hands']==2
    assert json.loads(output.read_text())['release_manifest']['format']=='expand-e07-manifest-v1'
    with pytest.raises(FileExistsError):run(args)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e07_bridges_and_cli.py" -->

## tests/expand/test_e07_cash_bridge.py

Original blob `3882cb83ec7615a6b4b8bc8e909d0d9a814352ad` (584 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e07_cash_bridge.py" sha="3882cb83ec7615a6b4b8bc8e909d0d9a814352ad" -->

````````python
from holdem_ai.expand.integration import IntegratedPokerAgent,uniform_legal
from holdem_ai.expand.integration.benchmark import _cash_session
from e03_fake_engine import FastFakeEngine


def test_e07_e1_e2_e6_accepts_public_actions():
    out=_cash_session(IntegratedPokerAgent(uniform_legal),n=4,stack_bb=20,
        style='calling_station',scenario_seed=70770,hero_seat=0,
        engine_factory=FastFakeEngine,max_actions=200,max_warmup_hands=70)
    assert out['hands']==1
    assert out['illegal_actions']==0
    assert out['opponent_events']>=1
    assert out['warmup_hands']>=1

````````

<!-- END_PY_SOURCE path="tests/expand/test_e07_cash_bridge.py" -->

## tests/expand/test_e07_manifest.py

Original blob `f805ce2e305669d0da8e0d29e2069e368187e94f` (847 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e07_manifest.py" sha="f805ce2e305669d0da8e0d29e2069e368187e94f" -->

````````python
import json
import pytest
from holdem_ai.expand.integration import release_manifest,write_new_json,sha256_file


def test_e07_manifest_checkpoint_hash_and_no_overwrite(tmp_path):
    src=tmp_path/'test.pt'
    src.write_bytes(b'checkpoint-v1')
    script=tmp_path/'script.py'
    script.write_text('print(1)')
    metadata=release_manifest(project_root=tmp_path,config={'a':1},seeds=[7],
             checkpoints={'ppo':str(src)},abstraction_id='test',scripts=['script.py'])
    assert metadata['checkpoints']['ppo']['sha256']==sha256_file(src)
    assert metadata['evaluation_script_sha256']['script.py']==sha256_file(script)
    output=tmp_path/'result.json'
    write_new_json(output,metadata)
    with pytest.raises(FileExistsError):write_new_json(output,metadata)
    assert json.loads(output.read_text())['format']=='expand-e07-manifest-v1'

````````

<!-- END_PY_SOURCE path="tests/expand/test_e07_manifest.py" -->

## tests/expand/test_e07_real_engine.py

Original blob `0621abe66f07e4425ec2ffef85e3ac04fd3db97a` (1008 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e07_real_engine.py" sha="0621abe66f07e4425ec2ffef85e3ac04fd3db97a" -->

````````python
"""Gated: run only when Base Phase3 engine is installed in user's repository."""
import pytest
pytest.importorskip('holdem_ai.environment.multi_player_engine')
from holdem_ai.expand.integration import IntegratedPokerAgent,uniform_legal
from holdem_ai.expand.integration.benchmark import BenchmarkConfig,cash_benchmark,tournament_benchmark


def test_e07_real_cash_holdem_2_to_6():
    cfg=BenchmarkConfig(groups=2,players=(2,3,4,5,6),stacks_bb=(100,),
                        opponents=('calling_station',),max_warmup_hands=120)
    out=cash_benchmark(IntegratedPokerAgent(uniform_legal),cfg)
    assert out['scored_hands']==10
    assert all(x['bb100']['n_independent_sessions']==2 for x in out['strata'])


def test_e07_real_tournament_chip_prize():
    cfg=BenchmarkConfig(mode='tournament',groups=2,players=(2,3),
                        opponents=('calling_station',),max_tournament_hands=150)
    result=tournament_benchmark(IntegratedPokerAgent(uniform_legal),cfg)
    assert result['tournaments']==4

````````

<!-- END_PY_SOURCE path="tests/expand/test_e07_real_engine.py" -->
