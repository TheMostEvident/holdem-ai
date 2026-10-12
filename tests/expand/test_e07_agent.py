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
