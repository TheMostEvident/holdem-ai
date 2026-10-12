"""Single-table 2-6 seat tournament adapter over Base Phase3 Hand Engine.

Keeps per-ENTRY records (re-entry is a NEW entry, not revival), persistent
chips across hands, scheduled blinds, optional registration window, finish
rank + prize ledger. Ante support is explicitly refused until underlying hand
engine offers antes. This is NOT a multi-table tournament director or an ICM
solver. All registration and re-entry take effect at hand boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from holdem_ai.expand.environment.multiway_adapter import make_hand_engine, moving_button_after
from holdem_ai.expand.environment.table_state import Action


@dataclass(frozen=True)
class TournamentConfig:
    starting_stack: int = 2000
    blind_levels: tuple[tuple[int,int], ...] = ((10,20),(20,40),(40,80),(80,160),(160,320),(320,640),(640,1280),(1280,2560))
    hands_per_level: int = 4
    registration_until_hand: int = 0  # inclusive hand index before which new entry accepted
    max_reentries_per_player: int = 0
    payouts: tuple[float,...] = (1.,)  # index 0 is first place, missing positions 0
    max_hands: int = 10000
    ante_chips: int = 0

    def __post_init__(self):
        if type(self.starting_stack) is not int or self.starting_stack <= 0:
            raise ValueError('starting_stack >0 required')
        if not self.blind_levels or any(type(a) is not int or type(b) is not int or not 0<a<b
                                        for a,b in self.blind_levels):
            raise ValueError('valid blind level (SB, BB) tuples required')
        for key in ('hands_per_level','max_hands'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('registration_until_hand','max_reentries_per_player','ante_chips'):
            if type(getattr(self,key)) is not int or getattr(self,key)<0:
                raise ValueError(f'{key} must be nonnegative integer')
        if self.ante_chips:
            raise NotImplementedError('antes require ante support in Base Phase3 engine')
        if not self.payouts or any(type(x) not in (int,float) or x<0 for x in self.payouts):
            raise ValueError('invalid nonnegative prize ladder')
        if any(self.payouts[i]<self.payouts[i+1] for i in range(len(self.payouts)-1)):
            raise ValueError('payouts must be non-increasing')


@dataclass
class Entry:
    entry_id: str
    player_id: str
    seat: int
    stack: int
    reentry_number: int
    eliminated_hand: int | None = None
    pre_bust_stack: int | None = None
    rank: int | None = None
    prize: float = 0.


@dataclass(frozen=True)
class TournamentHand:
    hand_number: int
    participants: tuple[tuple[int,str],...]
    button: int
    sb: int
    bb: int
    big_blind: int
    seed: int
    terminal: bool


@dataclass(frozen=True)
class TournamentSettlement:
    hand_number: int
    payoffs_chips: tuple[int,...]
    final_stacks: tuple[int,...]
    eliminated_entries: tuple[str,...]
    complete: bool


@dataclass(frozen=True)
class TournamentReport:
    winner_entry_id: str
    hands_played: int
    entries: tuple[tuple[str,str,int,float,int|None],...]  # entry_id, player_id, rank, prize, bust hand
    total_prizes: float


class SingleTableTournament:
    def __init__(self, config:TournamentConfig|None=None, *, engine_factory:Callable|None=None):
        self.config=config or TournamentConfig()
        self.factory=engine_factory or make_hand_engine
        self.entries: list[Entry]=[]
        self.seats: list[Entry|None]=[None]*6
        self.hand=None
        self.button=None
        self.hands_played=0
        self._before=None
        self._roster: tuple[int,...]=()
        self._registered_by_player:dict[str,int]={}
        self._closed=False
        self._complete=False
        self._report=None
        self.events: list[dict]=[]

    @property
    def registration_open(self)->bool:
        return not self._closed and not self._complete and self.hands_played<self.config.registration_until_hand or (
            not self._closed and not self._complete and self.hands_played==0 and self.hand is None)

    def close_registration(self):
        if self.hand is not None: raise RuntimeError('close registration only between hands')
        self._closed=True
        self._try_finish()

    def register(self,player_id:str,*,seat:int|None=None,reentry:bool=False)->str:
        """Register at boundary. Same player may reenter after a confirmed bust."""
        if self.hand is not None:raise RuntimeError('register only between hands')
        if not self.registration_open:raise RuntimeError('late registration closed')
        if type(player_id) is not str or not player_id.strip():raise ValueError('valid player ID required')
        count=self._registered_by_player.get(player_id,0)
        if bool(count)!=bool(reentry):
            raise ValueError('reentry flag must exactly match prior entry existence')
        if reentry and count>self.config.max_reentries_per_player:
            raise ValueError('player reentry cap reached')
        if any(e.player_id==player_id and e.eliminated_hand is None for e in self.entries):
            raise ValueError('player already active')
        free=[i for i,e in enumerate(self.seats) if e is None]
        if seat is None:
            if not free:raise ValueError('table full')
            seat=free[0]
        if type(seat) is not int or seat not in free:raise ValueError('seat unavailable')
        entry=Entry(f'{player_id}#entry{count+1}',player_id,seat,self.config.starting_stack,count)
        self.seats[seat]=entry
        self.entries.append(entry)
        self._registered_by_player[player_id]=count+1
        self.events.append({'type':'registered','entry_id':entry.entry_id,'seat':seat,'after_hand':self.hands_played})
        return entry.entry_id

    @property
    def alive(self)->tuple[Entry,...]:
        return tuple(e for e in self.seats if e is not None and e.stack>0)

    @property
    def current_player_id(self)->str|None:
        if self.hand is None or self.hand.is_over():return None
        e=self.seats[self.hand.current_player]
        assert e is not None
        return e.player_id

    def blind_level(self)->tuple[int,int]:
        idx=min(self.hands_played//self.config.hands_per_level,len(self.config.blind_levels)-1)
        return self.config.blind_levels[idx]

    def start_hand(self,seed:int)->TournamentHand|None:
        if type(seed) is not int or seed<0:raise ValueError('seed must be >=0 integer')
        if self.hand is not None:raise RuntimeError('previous hand not settled')
        if self._complete:return None
        self._try_finish()
        if self._complete:return None
        if len(self.alive)<2:return None  # late-registration waiting, not artificial result
        if self.hands_played>=self.config.max_hands:
            raise RuntimeError('max_hands reached; cannot award arbitrary champion')
        active=tuple(e.seat for e in self.alive)
        button=moving_button_after(self.button,active)
        stacks=tuple(e.stack if e is not None and e.stack>0 else 0 for e in self.seats)
        sb,bb=self.blind_level()
        engine=self.factory(stacks=stacks,button=button,small_blind=sb,big_blind=bb,seed=seed)
        engine.reset()
        self.hand=engine
        self.button=button
        self._before=stacks
        self._roster=active
        self.events.append({'type':'hand_started','number':self.hands_played+1,'seed':seed,
                            'button':button,'sb':engine.seats.small_blind,'bb':engine.seats.big_blind,
                            'blinds':[sb,bb],'roster':list(active)})
        return TournamentHand(self.hands_played+1,tuple((i,self.seats[i].entry_id) for i in active),
                              button,engine.seats.small_blind,engine.seats.big_blind,bb,seed,engine.is_over())

    def observe_current(self)->dict:
        if self.current_player_id is None:raise RuntimeError('no acting player')
        return dict(self.hand.observe(self.hand.current_player))

    def act(self,action:Action)->None:
        if self.current_player_id is None:raise RuntimeError('no acting player')
        if type(action) is not Action:raise TypeError('requires E1 Action')
        raw=self.hand.observe(self.hand.current_player)['raw_obs']
        to_call=raw['to_call']
        if action.kind=='check' and to_call:raise ValueError('check not legal')
        if action.kind=='call' and not to_call:raise ValueError('call not legal')
        if action.kind=='raise_to':self.hand.step_raise_to(action.raise_to_chips)
        else:
            lookup={'fold':0,'check':1,'call':1,'raise_half_pot':2,'raise_pot':3}
            name=action.kind
            code=lookup.get(name,1 if name=='all_in' and self.hand.stacks[self.hand.current_player]<=to_call else 4)
            self.hand.step(code)

    def settle_hand(self)->TournamentSettlement:
        if self.hand is None or not self.hand.is_over():raise RuntimeError('hand not terminal')
        result=self.hand.get_result()
        final=tuple(result.final_stacks)
        if len(final)!=6 or sum(final)!=sum(self._before) or sum(result.payoffs_chips)!=0:
            raise AssertionError('hand chip conservation failure')
        if any(final[i]!=0 for i in range(6) if i not in self._roster):
            raise AssertionError('inactive seat received chips')
        busts=[]
        for i in self._roster:
            entry=self.seats[i]
            entry.stack=final[i]
            if final[i]==0:
                entry.eliminated_hand=self.hands_played+1
                entry.pre_bust_stack=self._before[i]
                busts.append(entry.entry_id)
                self.seats[i]=None
        self.hands_played+=1
        self.hand=None
        self.events.append({'type':'hand_settled','number':self.hands_played,
                            'busts':list(busts),'final_stacks':list(final)})
        self._try_finish()
        return TournamentSettlement(self.hands_played,tuple(result.payoffs_chips),final,
                                    tuple(busts),self._complete)

    def _try_finish(self):
        if self._complete or self.hand is not None:return
        if len(self.entries)<2:return
        if len(self.alive)!=1:return
        if self.registration_open:return
        winner=self.alive[0]
        # Eliminations in later hands outrank earlier; among simultaneous
        # busts, larger pre-hand stacks outrank smaller ones. Equal share ranks.
        losers=[e for e in self.entries if e is not winner]
        losers.sort(key=lambda e:(e.eliminated_hand or -1,e.pre_bust_stack or 0),reverse=True)
        ordered=[winner]+losers
        cursor=0
        while cursor<len(ordered):
            group=[ordered[cursor]]
            if cursor>0:
                k=(ordered[cursor].eliminated_hand,ordered[cursor].pre_bust_stack)
                j=cursor+1
                while j<len(ordered) and (ordered[j].eliminated_hand,ordered[j].pre_bust_stack)==k:
                    group.append(ordered[j]);j+=1
            start=cursor
            prizes=[self.config.payouts[j] if j<len(self.config.payouts) else 0.
                    for j in range(start,start+len(group))]
            shared=sum(prizes)/len(group)
            for e in group:e.rank=start+1;e.prize=shared
            cursor+=len(group)
        self._complete=True
        self._report=TournamentReport(winner.entry_id,self.hands_played,
                   tuple((e.entry_id,e.player_id,e.rank,e.prize,e.eliminated_hand) for e in self.entries),
                   sum(e.prize for e in self.entries))
        self.events.append({'type':'finished','winner':winner.entry_id})

    @property
    def finished(self)->bool:return self._complete

    def result(self)->TournamentReport:
        if not self.finished:raise RuntimeError('tournament unfinished')
        return self._report
