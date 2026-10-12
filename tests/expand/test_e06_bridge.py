import pytest
from holdem_ai.expand.opponent import PublicOpponentTracker,position_for_seat,count_prior_preflop_raises
from holdem_ai.expand.environment.table_state import HandHandle, PlayerSeat,TableSnapshot,TableConfig
from holdem_ai.expand.information.schema import PublicAction
from test_e06_state_bridge import state_with_hole


def hand(players=((0,'a'),(1,'b'),(2,'c'),(3,'d'),(4,'e'),(5,'f'))):
    return HandHandle('hand1',1,0,1,2,players,77,False)

def test_positions_six_and_four():
    h=hand()
    assert [position_for_seat(h,i) for i in range(6)]==[
       'button','small_blind','big_blind','early','middle','cutoff']
    h4=hand(((0,'a'),(1,'b'),(2,'c'),(5,'f')))
    assert position_for_seat(h4,5)=='cutoff'

def test_prior_raises_counts_exact_raise_to():
    a=PublicAction(1,'preflop','raise_to',3.5,1.,1.5)
    b=PublicAction(2,'flop','raise_pot',7.,0.,3.)
    assert count_prior_preflop_raises((a,b))==1

def test_registry_tracks_join_tenures_from_snapshot():
    snap=TableSnapshot(TableConfig(),tuple(PlayerSeat(i,'a' if i==0 else 'b' if i==1 else None,
       10000 if i<2 else 0,'seated' if i<2 else 'vacant') for i in range(6)),0,3,'session',False,None,0,20000)
    tracker=PublicOpponentTracker()
    tracker.sync_seats(snap,{'a':'e1','b':'e2'})
    assert tracker.registry.active_by_seat[0]==('a','e1')
    tracker.sync_seats(snap,{'a':'e99','b':'e2'})
    assert tracker.registry.active_by_seat[0]==('a','e99')
    assert ('a','e1') in tracker.registry.models

def test_accepted_action_only_after_caller_commits():
    tracker=PublicOpponentTracker()
    tracker.registry.bind_seat(0,'a','e1')
    state=state_with_hole(('sA','hA'))
    h=hand(((0,'a'),(1,'b')))
    output=tracker.record_accepted_action(state,handle=h,player_id='a',player_session_id='e1',
                                          action_index=0,action='fold')
    assert output.raw_events==1
    with pytest.raises(ValueError):tracker.record_accepted_action(state,handle=h,player_id='b',
                                           player_session_id='e1',action_index=1,action='fold')
