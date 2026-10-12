from types import SimpleNamespace
import pytest
from holdem_ai.expand.environment import Action, DynamicTable
from holdem_ai.expand.information import Phase2TableAdapter


class PublicRichEngine:
    def __init__(self, *, stacks, button, small_blind, big_blind, seed):
        self.initial_stacks=tuple(stacks)
        self.stacks=list(stacks)
        self.contributions=[0]*6
        self.bets=[0]*6
        self.status=['active' if x>0 else 'out' for x in stacks]
        self.button=button
        occupied=[i for i,x in enumerate(stacks) if x>0]
        self.current_player=occupied[0]
        def nxt(i):
            return next((i+k)%6 for k in range(1,7) if (i+k)%6 in occupied)
        self.seats=SimpleNamespace(small_blind=button if len(occupied)==2 else nxt(button),
                                    big_blind=nxt(button) if len(occupied)==2 else nxt(nxt(button)))
        self._order=occupied
        self._history=[]
        self._raise_history=[]
        self.street='preflop'
        self.board=[]
        self._done=False

    def reset(self):
        return self.observe(self.current_player), self.current_player

    def is_over(self):
        return self._done

    def observe(self, player_id):
        assert player_id==self.current_player and not self._done
        return {'legal_actions':{0:None,1:None},'raw_obs':{
            'hand':['cA','dK'], 'public_cards':list(self.board),
            'street':self.street,'button':self.button,'player_id':player_id,
            'stacks':tuple(self.stacks),'street_bets':tuple(self.bets),
            'total_contributions':tuple(self.contributions),'status':tuple(self.status),
            'pot':0,'to_call':0,'min_raise_to':2,
            'action_history':tuple(self._history),'action_raise_to':tuple(self._raise_history)}}

    def step(self, action_id):
        assert action_id==1
        player=self.current_player
        self._history.append((player,action_id))
        self._raise_history.append(None)
        if len(self._history)==1:
            self.street='flop'
            self.board=['c2','d4','h5']
            self.current_player=self._order[1]
        else:
            self._done=True
            self.current_player=-1

    def get_result(self):
        assert self._done
        return SimpleNamespace(final_stacks=tuple(self.stacks),payoffs_chips=(0,)*6,
          board=tuple(self.board),action_history=tuple(self._history),
          action_raise_to=tuple(self._raise_history))


def mk_table():
    table=DynamicTable(engine_factory=PublicRichEngine)
    table.request_join('a',10000,0)
    table.request_join('b',10000,1)
    return table


def test_e02_tracker_carries_street_across_actions():
    a=Phase2TableAdapter(mk_table())
    a.start_hand(1)
    before=a.observe()
    assert before.action_history==()
    a.act_abstract('check')
    after=a.observe()
    assert after.street=='flop'
    assert after.action_history[0].street=='preflop'
    assert after.action_history[0].kind=='check'
    a.act_abstract('check')
    settlement=a.settle_hand()
    assert settlement.hand_number==1
    assert not a.table.snapshot().hand_in_progress
    a.start_hand(2)
    assert a.observe().action_history==()


def test_e02_rejected_action_does_not_corrupt_tracker():
    a=Phase2TableAdapter(mk_table())
    a.start_hand(1)
    with pytest.raises(ValueError):
        a.act_abstract('call')
    assert len(a.observe().action_history)==0


def test_e02_external_action_detected_without_leak():
    a=Phase2TableAdapter(mk_table())
    a.start_hand(1)
    # Caller bypasses adapter; no guessed/mislabelled history permitted.
    a.table.act(a.table.current_player_id,Action('check'))
    with pytest.raises(ValueError,match='mismatched'):
        a.observe()


def test_e02_attaching_midhand_rejected():
    table=mk_table()
    table.start_hand(1)
    with pytest.raises(ValueError,match='mid-hand'):
        Phase2TableAdapter(table)
