import random
import pytest
from holdem_ai.expand.pluribus.abstraction import AbstractionConfig, holdem_infoset_key
from holdem_ai.expand.pluribus.blueprint import HoldemGame,MCCFRSolver,MCCFRConfig,save_checkpoint,load_checkpoint
from e03_fake_engine import FastFakeEngine

@pytest.mark.parametrize('n',[2,3,4,5,6])
def test_real_interface_on_fast_deterministic_engine(n):
    game=HoldemGame(n,engine_factory=FastFakeEngine)
    state=game.start(random.Random(n))
    assert len(state.player_ids)==n and len(set(state.player_ids))==n
    key=state.infoset(state.current_player)
    assert 'e04-holdem' in key
    solver=MCCFRSolver(game,MCCFRConfig(seed=n,max_nodes_per_iteration=10000))
    stats=solver.train(12)
    assert stats[-1]['iteration']==12
    assert any(sum(e['strategy_sum'])>0 for e in solver.store.entries.values())


def test_private_rival_cards_do_not_affect_hero_infoset():
    game=HoldemGame(3,engine_factory=FastFakeEngine)
    state=game.start(random.Random(1))
    k=state.infoset(state.current_player)
    state.engine.some_private_opponent_cards = ['sA','cK']
    state.engine._future_deck = ['hA']
    assert k==state.infoset(state.current_player)


def test_postflop_card_buckets_and_abstraction_id():
    from copy import deepcopy
    game=HoldemGame(2,engine_factory=FastFakeEngine)
    state=game.start(random.Random(10))
    raw=state.engine.observe(state.current_player)['raw_obs']
    raw=deepcopy(raw)
    raw['street']='flop'
    raw['public_cards']=['hA','d2','s7']
    ident=holdem_infoset_key(raw,('fold','check'),AbstractionConfig(postflop_buckets=16))
    assert 'prototype_bucket' in ident
    assert AbstractionConfig(16).fingerprint != AbstractionConfig(32).fingerprint
    assert 'cA' not in ident # bucket, not exact private card


def test_holdem_save_and_restore(tmp_path):
    solver=MCCFRSolver(HoldemGame(3,engine_factory=FastFakeEngine), MCCFRConfig(seed=123))
    solver.train(9)
    path=tmp_path/'p.json.gz'
    save_checkpoint(path,solver)
    loaded=load_checkpoint(path,engine_factory=FastFakeEngine)
    assert loaded.game.abstraction.fingerprint==solver.game.abstraction.fingerprint
    assert loaded.store.as_dict()==solver.store.as_dict()
    loaded.train(2)
    assert loaded.completed_iterations==11


def test_no_illegal_abstract_actions_on_mock():
    state=HoldemGame(6,engine_factory=FastFakeEngine).start(random.Random(5))
    while not state.is_terminal():
        for x in state.legal_actions():
            assert x in ('fold','check','call','raise_half_pot','raise_pot','all_in')
        state=state.next_state(state.legal_actions()[-1])
    assert sum(state.terminal_payoffs())==pytest.approx(0)
