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
