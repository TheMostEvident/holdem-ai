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
