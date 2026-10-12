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
