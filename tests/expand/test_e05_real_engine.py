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
