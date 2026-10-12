"""A deterministic side-effect-free test double, NEVER a poker engine."""
from types import SimpleNamespace


class KnockoutEngine:
    def __init__(self, *, stacks, button, small_blind, big_blind, seed):
        self.initial=tuple(stacks)
        self.stacks=list(stacks)
        self.active=[i for i,s in enumerate(stacks) if s>0]
        self.button=button
        self.big_blind=big_blind
        self.small_blind=small_blind
        self.seed=seed
        self.current_player=self.active[0]
        self.done=False
        def nxt(pos):return next((pos+i)%6 for i in range(1,7) if (pos+i)%6 in self.active)
        sb=button if len(self.active)==2 else nxt(button)
        bb=nxt(button) if len(self.active)==2 else nxt(sb)
        self.seats=SimpleNamespace(small_blind=sb,big_blind=bb)
    def reset(self):return self.observe(self.current_player),self.current_player
    def observe(self,p):
        assert p==self.current_player
        return {'legal_actions':{0:None,1:None},
            'raw_obs':{'hand':['cA','dK'],'public_cards':[], 'street':'preflop',
            'button':self.button,'player_id':p,'stacks':tuple(self.stacks),
            'street_bets':(0,)*6,'total_contributions':(0,)*6,
            'status':tuple('active' if i in self.active else 'out' for i in range(6)),
            'pot':0,'to_call':0,'min_raise_to':self.big_blind,
            'action_history':(), 'action_raise_to':()}}
    def step(self,a):
        if a not in (0,1):raise ValueError('fake legal action')
        self.current_player=-1
        self.done=True
    def is_over(self):return self.done
    def get_result(self):
        assert self.done
        # Knock out one (nonwinner) seat; deterministic and zero-sum.
        loser=self.active[0]
        winner=self.active[-1]
        final=list(self.stacks)
        final[winner]+=final[loser]
        final[loser]=0
        payoffs=tuple(a-b for a,b in zip(final,self.stacks))
        return SimpleNamespace(final_stacks=tuple(final),payoffs_chips=payoffs)
