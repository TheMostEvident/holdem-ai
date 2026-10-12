from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import pytest

from holdem_ai.expand.information import (make_information_state, legal_action_spec,
    to_table_action, ACTION_NAMES, TrainingOnlyCriticState, encode_observation)
from holdem_ai.expand.information.encoder import SEAT_DIM, GLOBAL_DIM, HISTORY_DIM, MAX_HISTORY


def obs(*, hero=0, to_call=100, stack=10000, pot=150, street='preflop',
        legal=(0, 1, 2, 3, 4)):
    stacks = [10000,10000,10000,0,0,0]
    stacks[hero] = stack
    invested = [0, 50, 100, 0,0,0]
    bets = list(invested)
    if hero == 0:
        assert to_call==100
    if hero == 1:
        assert to_call==50
    if hero == 2:
        assert to_call==0
    return {
        'legal_actions': {i: None for i in legal},
        'raw_obs': {'hand':['cA','hK'],'public_cards':[],
                    'street':street,'button':0,'player_id':hero,
                    'stacks':tuple(stacks),'street_bets':tuple(bets),
                    'total_contributions':tuple(invested),'status':('active','active','active','out','out','out'),
                    'pot':pot,'to_call':to_call,'min_raise_to':200,
                    'action_history':(), 'action_raise_to':(),
                    # These must never reach actor even if supplied to builder.
                    'opponent_holes':(('dA','sA'),), 'future_deck':('c2','h2'), 'seed':123},
        'hidden_engine_state': {'hole_cards': 'DANGER'},
    }


def mk():
    return make_information_state(obs(), big_blind_chips=100,
        seat_presence=(True,True,True,False,False,False))


def test_e02_only_acting_player_and_public_fields():
    state = mk()
    assert state.hero_hole == ('cA', 'hK')
    assert state.num_dealt_in == 3
    assert state.seats[3].relative_position == 3
    assert not state.seats[3].occupied
    assert not state.seats[3].dealt_in
    assert state.seats[0].is_hero
    assert 'opponent' not in repr(state)
    assert 'future_deck' not in repr(state)
    assert 'hidden_engine_state' not in repr(state)
    assert 'seed' not in repr(state)
    encoded = encode_observation(state)
    assert len(encoded.seat_features) == 6
    assert all(len(row) == SEAT_DIM for row in encoded.seat_features)
    assert len(encoded.global_features) == GLOBAL_DIM
    assert len(encoded.history_features) == MAX_HISTORY
    assert all(len(x) == HISTORY_DIM for x in encoded.history_features)
    assert sum(encoded.seat_mask) == 3
    assert sum(encoded.history_mask) == 0


def test_e02_invariance_to_opponent_hole_and_future_deck():
    a, b = obs(), obs()
    b['raw_obs']['opponent_holes'] = [('c3','d3'),('c6','h6')]
    b['raw_obs']['future_deck'] = ['hT']
    b['raw_obs']['seed'] = 9999
    b['hidden_engine_state'] = {"critic": "nothing"}
    left = make_information_state(a,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    right = make_information_state(b,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    assert left == right
    assert encode_observation(left) == encode_observation(right)
    with pytest.raises((AttributeError, FrozenInstanceError)):
        left.hero_hole = ('dA','hA')


def test_e02_street_hole_and_board_validation():
    o = obs()
    o['raw_obs']['public_cards'] = ['sA']
    with pytest.raises(ValueError, match='board'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    o=obs()
    o['raw_obs']['hand'] = ['cA','cA']
    with pytest.raises(ValueError, match='duplicate'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    o=obs()
    o['raw_obs']['street'] = 'river'
    with pytest.raises(ValueError, match='street/board'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)


def test_e02_bad_public_chip_state_rejected():
    o = obs()
    o['raw_obs']['pot'] = 151
    with pytest.raises(ValueError,match='pot/contributions'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    o = obs()
    o['raw_obs']['status'] = ('active','active','active','active','out','out')
    with pytest.raises(ValueError,match='registry'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)


def test_e02_action_translation_exact_pot_math():
    state = mk()
    spec = state.legal_actions
    assert spec.mask == (True, False, True, True, True, True, True)
    assert spec.raise_to_chips[3] == 225
    assert spec.raise_to_chips[4] == 350
    assert spec.raise_to_chips[5] == 600
    assert spec.raise_to_chips[6] == 10000
    assert to_table_action(spec, 'raise_half_pot').raise_to_chips == 225
    assert to_table_action(spec, 'raise_pot').raise_to_chips == 350
    assert to_table_action(spec, 'raise_double_pot').raise_to_chips == 600
    assert to_table_action(spec, 'all_in').kind == 'all_in'
    with pytest.raises(ValueError,match='illegal'):
        to_table_action(spec,'check')


def test_e02_short_allin_call_is_not_duplicate_allin_action():
    o=obs(stack=50,legal=(0,1))
    state=make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    spec=state.legal_actions
    assert spec.allows('call')
    assert not spec.allows('all_in')
    assert not spec.allows('raise_double_pot')


def test_e02_check_vs_call_legality():
    o=obs(hero=2,to_call=0)
    state=make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    assert state.legal_actions.allows('check')
    assert not state.legal_actions.allows('call')


def test_e02_deduplicate_wager_target():
    o=obs(stack=225)
    # Half-pot raise to == all-in, canonical half-pot action only
    spec=legal_action_spec(o['raw_obs'],tuple(o['legal_actions']),100)
    assert spec.allows('raise_half_pot')
    assert not spec.allows('all_in')


def test_e02_all_sitting_out_retained_as_occupied_token():
    o=obs()
    state=make_information_state(o,big_blind_chips=100,
          seat_presence=(True,True,True,True,False,False))
    assert state.seats[3].occupied and not state.seats[3].dealt_in
    assert encode_observation(state).seat_mask[3]


def test_e02_no_training_critic_in_encoder():
    with pytest.raises(TypeError,match='InformationState'):
        encode_observation(TrainingOnlyCriticState(full_hand_state={'secret': 'x'}))


def test_e02_empty_action_raises():
    o=obs(legal=())
    with pytest.raises(ValueError,match='unsupported engine'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)


def test_e02_table_occupancy_status_are_separate_and_consistent():
    o=obs()
    with pytest.raises(ValueError,match='presence/status'):
        make_information_state(o,big_blind_chips=100,
          seat_presence=(True,True,True,False,False,False),
          seat_statuses=('seated','seated','seated','waiting_bb','vacant','vacant'))
    state=make_information_state(o,big_blind_chips=100,
          seat_presence=(True,True,True,True,False,False),
          seat_statuses=('seated','seated','seated','sitting_out','vacant','vacant'))
    assert state.seats[3].table_status=='sitting_out'
    assert not state.seats[3].dealt_in
    x=encode_observation(state)
    assert x.seat_features[3][-2] == 1.0  # sitting-out token
