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
