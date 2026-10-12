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
