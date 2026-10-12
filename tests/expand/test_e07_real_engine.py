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
