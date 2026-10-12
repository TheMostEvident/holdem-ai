"""Deterministic test-only 2-6 player game. NOT a Hold'em rules engine."""
from types import SimpleNamespace


class FastFakeEngine:
    def __init__(self,*,stacks,button,small_blind,big_blind,seed):
        self.initial_stacks=tuple(stacks)
        self.stacks=list(stacks)
        self.contributions=[0]*6
        self.street_bets=[0]*6
        self.status=['active' if s else 'out' for s in stacks]
        self._seats=[j for j,x in enumerate(stacks) if x>0]
        self.button=button
        self.seed=seed
        def nxt(x):
            return next((x+k)%6 for k in range(1,7) if (x+k)%6 in self._seats)
        sb=button if len(self._seats)==2 else nxt(button)
        bb=nxt(button) if len(self._seats)==2 else nxt(sb)
        self.seats=SimpleNamespace(small_blind=sb,big_blind=bb)
        self.current_player=self._seats[0]
        self.history=[]
        self.raises=[]
        self._done=False
    def reset(self):
        return self.observe(self.current_player),self.current_player
    def observe(self,pid):
        assert pid==self.current_player
        cards=['cA','dK','cT','dJ','h3','s7']
        return {'legal_actions':{0:None,1:None},'raw_obs':{'hand':[cards[pid],cards[(pid+1)%6]],
         'public_cards':[],'street':'preflop','button':self.button,
         'player_id':pid,'stacks':tuple(self.stacks),'street_bets':tuple(self.street_bets),
         'total_contributions':tuple(self.contributions),'status':tuple(self.status),
         'pot':0,'to_call':0,'min_raise_to':2,'action_history':tuple(self.history),
         'action_raise_to':tuple(self.raises)}}
    def step(self,a):
        assert a in (0,1)
        self.history.append((self.current_player,a))
        self.raises.append(None)
        i=self._seats.index(self.current_player)
        if i+1==len(self._seats):
            self._done=True
            self.current_player=-1
        else:self.current_player=self._seats[i+1]
    def is_over(self):return self._done
    def get_result(self):
        assert self._done
        awards=[0]*6
        # Deterministic, zero-sum, signed hand payoff to each physical seat.
        if self.seed%2:
            if self._seats[1]!=self._seats[0]:
                awards[self._seats[0]]+=10
                awards[self._seats[1]]-=10
        else:
            awards[self._seats[0]]-=10
            awards[self._seats[1]]+=10
        return SimpleNamespace(final_stacks=tuple(x+y for x,y in zip(self.stacks,awards)),
           payoffs_chips=tuple(awards),board=(),action_history=tuple(self.history),
           action_raise_to=tuple(self.raises))

# Test-only isolated shim: absent Base Phase3 files in sandbox. Never installed
# in source distribution and never used when the real engine is importable.
try:
    from holdem_ai.environment.multi_player_seats import multi_player_seats
except ModuleNotFoundError:
    import holdem_ai.expand.environment.multiway_adapter as _adapter
    def _test_blind_seats(button, active):
        def nxt(seat):
            return next((seat + k) % 6 for k in range(1,7) if (seat + k) % 6 in active)
        if len(active)==2:
            return button, nxt(button)
        return nxt(button), nxt(nxt(button))
    _adapter.blind_seats=_test_blind_seats
    import holdem_ai.expand.environment.table_manager as _table_manager
    _table_manager.blind_seats=_test_blind_seats
