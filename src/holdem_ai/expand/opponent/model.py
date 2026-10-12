"""Per-player Bayesian-smoothed decayed action model; only public observations.

Counts are conditioned on ALL four: street / public position / SPR / legal set.
Different legal-action sets are never spuriously pooled. No opponent hole
cards, board runout, future deck or hidden simulator variables are read/stored.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from .schema import PublicDecision, RAISE_ACTIONS


@dataclass(frozen=True)
class OpponentConfig:
    alpha: float = 0.5
    long_decay: float = 0.995
    recent_decay: float = 0.88
    min_recent: float = 8.0
    min_long: float = 20.0
    shift_kl_threshold: float = 0.13
    shift_consecutive: int = 2
    shift_hold_events: int = 10
    min_adapt_samples: float = 18.0
    max_contexts_per_player: int = 1024
    max_event_ids: int = 200000

    def __post_init__(self):
        for f in ('alpha','min_recent','min_long','shift_kl_threshold','min_adapt_samples'):
            x=getattr(self,f)
            if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or (x <= 0 if f=='alpha' else x < 0):
                raise ValueError(f'{f} invalid')
        for f in ('long_decay','recent_decay'):
            x=getattr(self,f)
            if not isinstance(x,(float,int)) or isinstance(x,bool) or not 0<x<1:
                raise ValueError('decays must lie strictly within (0,1)')
        if self.recent_decay >= self.long_decay:raise ValueError('recent decay must be faster than long decay')
        for f in ('shift_consecutive','shift_hold_events','max_contexts_per_player','max_event_ids'):
            x=getattr(self,f)
            if type(x) is not int or x < 1:raise ValueError(f'{f} must be positive integer')


@dataclass
class ContextCounts:
    actions: tuple[str,...]
    long: list[float]
    recent: list[float]
    raw_events: int = 0
    over_streak: int = 0
    shift_hold: int = 0
    shift_events: int = 0

    @classmethod
    def empty(cls,actions):return cls(actions,[0.]*len(actions),[0.]*len(actions))

    def posterior(self,counts:list[float],alpha:float)->dict[str,float]:
        den = sum(counts)+alpha*len(counts)
        return {a:(n+alpha)/den for a,n in zip(self.actions,counts)}

    def kl(self,alpha:float)->float:
        # Put BOTH windows on the same prior-to-data ratio. Otherwise a
        # stationary deterministic opponent generates a spurious KL increase:
        # recent counts saturate (~9) while long counts continue growing.
        long_total=sum(self.long)
        recent_total=sum(self.recent)
        if long_total==0 or recent_total==0:
            return 0.0
        matching_prior=alpha*recent_total/long_total
        a=self.posterior(self.recent,matching_prior)
        b=self.posterior(self.long,alpha)
        return max(0.,sum(x*math.log(x/b[k]) for k,x in a.items()))

    def add(self,action:str,cfg:OpponentConfig)->None:
        i=self.actions.index(action)
        self.long=[x*cfg.long_decay for x in self.long]
        self.recent=[x*cfg.recent_decay for x in self.recent]
        self.long[i]+=1.
        self.recent[i]+=1.
        self.raw_events+=1
        if self.shift_hold:
            self.shift_hold-=1
        enough = sum(self.long)>=cfg.min_long and sum(self.recent)>=cfg.min_recent
        if enough and self.kl(cfg.alpha)>cfg.shift_kl_threshold:
            self.over_streak+=1
        else:
            self.over_streak=0
        if self.over_streak>=cfg.shift_consecutive:
            if self.shift_hold==0:self.shift_events+=1
            self.shift_hold=cfg.shift_hold_events
            self.over_streak=0
            # Start a fresh regime from *recent public evidence*. The longer
            # baseline must not keep contaminating predictions after a
            # confidently detected switch. Evidence temporarily drops below
            # the adaptation minimum so the basic policy is used during
            # recovery; new observations must re-earn the right to adapt.
            self.long=list(self.recent)


@dataclass(frozen=True)
class ModelEvidence:
    player_id: str
    player_session_id: str
    context: tuple[str,str,str,tuple[str,...]]
    posterior: dict[str,float]
    recent_posterior: dict[str,float]
    raw_events: int
    effective_long: float
    effective_recent: float
    kl_recent_long: float
    shift_active: bool
    shift_events: int

    def probability(self,kind:str)->float:
        if kind in self.posterior:return self.posterior[kind]
        raise ValueError('action not legal in this context')


@dataclass
class OpponentModel:
    player_id: str
    player_session_id: str
    config: OpponentConfig
    contexts: dict[tuple[str,str,str,tuple[str,...]],ContextCounts] = field(default_factory=dict)
    seen_events: set[tuple[str,str,str,int]] = field(default_factory=set)
    hands: dict[str,dict[str,bool]] = field(default_factory=dict)
    finished_hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    threebet_hands: int = 0

    def observe(self,event:PublicDecision)->ModelEvidence:
        if event.identity!=(self.player_id,self.player_session_id):
            raise ValueError('identity/tenure mismatch')
        if event.event_id in self.seen_events:raise ValueError('duplicate decision event')
        if len(self.seen_events)>=self.config.max_event_ids:
            raise RuntimeError('event ID budget reached; archive and start new analysis session')
        if event.context not in self.contexts and len(self.contexts)>=self.config.max_contexts_per_player:
            raise RuntimeError('context budget reached; archive and start new analysis session')
        self.seen_events.add(event.event_id)
        cell=self.contexts.setdefault(event.context,ContextCounts.empty(event.legal))
        cell.add(event.action,self.config)
        if event.street=='preflop':
            hand=self.hands.setdefault(event.hand_id,{'vpip':False,'pfr':False,'threebet':False})
            is_raise=event.action in RAISE_ACTIONS
            hand['vpip'] |= event.action=='call' or is_raise
            hand['pfr'] |= is_raise
            hand['threebet'] |= is_raise and event.preflop_raises_before>=1
        return self.evidence(event.context)

    def end_hand(self,hand_id:str)->None:
        if type(hand_id) is not str or not hand_id:raise ValueError('hand ID required')
        hand=self.hands.pop(hand_id,None)
        if hand is None:return
        self.finished_hands+=1
        self.vpip_hands+=int(hand['vpip'])
        self.pfr_hands+=int(hand['pfr'])
        self.threebet_hands+=int(hand['threebet'])

    def diagnostics(self)->dict[str,float|int]:
        n=self.finished_hands
        return {'preflop_hands':n,'vpip_rate':self.vpip_hands/n if n else 0.0,
                'pfr_rate':self.pfr_hands/n if n else 0.0,
                'threebet_rate':self.threebet_hands/n if n else 0.0,
                'total_contexts':len(self.contexts),'total_events':len(self.seen_events)}

    def evidence(self,context:tuple[str,str,str,tuple[str,...]])->ModelEvidence:
        counts=self.contexts.get(context)
        if counts is None:
            counts=ContextCounts.empty(context[-1])
        cfg=self.config
        return ModelEvidence(self.player_id,self.player_session_id,context,
                             counts.posterior(counts.long,cfg.alpha),
                             counts.posterior(counts.recent,cfg.alpha),counts.raw_events,
                             sum(counts.long),sum(counts.recent),
                             counts.kl(cfg.alpha),counts.shift_hold>0,counts.shift_events)


class OpponentRegistry:
    def __init__(self,config:OpponentConfig|None=None):
        self.config=config or OpponentConfig()
        self.models:dict[tuple[str,str],OpponentModel]={}
        self.active_by_seat:dict[int,tuple[str,str]]={}

    def bind_seat(self,seat_id:int,player_id:str,player_session_id:str)->None:
        if type(seat_id) is not int or not 0<=seat_id<6:raise ValueError('invalid seat')
        if not player_id or not player_session_id:raise ValueError('bad identity')
        identity=(player_id,player_session_id)
        if seat_id in self.active_by_seat and self.active_by_seat[seat_id]!=identity:
            raise ValueError('unbind previous seat occupant before a new join')
        if any(k!=seat_id and v==identity for k,v in self.active_by_seat.items()):
            raise ValueError('player tenure is already seated elsewhere')
        self.active_by_seat[seat_id]=identity
        self.models.setdefault(identity,OpponentModel(*identity,self.config))

    def unbind_seat(self,seat_id:int)->None:
        self.active_by_seat.pop(seat_id,None)

    def observe(self,event:PublicDecision)->ModelEvidence:
        if self.active_by_seat.get(event.seat_id)!=event.identity:
            raise ValueError('seat player/tenure mismatch: refuse to mix histories')
        return self.models[event.identity].observe(event)

    def evidence(self,seat_id:int,context)->ModelEvidence|None:
        identity=self.active_by_seat.get(seat_id)
        if identity is None:return None
        return self.models[identity].evidence(context)
