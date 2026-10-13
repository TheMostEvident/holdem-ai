# Python full source volume 12/28

[Canonical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) — original commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`; exact code with individual blob SHA.

## src/holdem_ai/expand/integration/manifest.py

Original SHA `4350626ed7f40434e89fb59101b34a9afd004dbe` (2055 bytes).

<a id="py-133"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/integration/manifest.py" sha="4350626ed7f40434e89fb59101b34a9afd004dbe" -->

````````python
"""Reproducible experiment metadata; no checkpoint modification."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Mapping


def sha256_file(path: str|Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def release_manifest(*, project_root: str|Path, config: Mapping,
                     seeds: list[int], checkpoints: Mapping[str,str] | None = None,
                     abstraction_id: str='none', scripts: list[str] | None = None) -> dict:
    root=Path(project_root)
    paths={}
    for name,path in sorted((checkpoints or {}).items()):
        p=Path(path).resolve(strict=True)
        paths[name]={'absolute_path':str(p),'sha256':sha256_file(p)}
    files={}
    for name in scripts or []:
        p=(root/name).resolve(strict=True)
        if not p.is_relative_to(root.resolve()):
            raise ValueError('script must be beneath project root')
        files[name]=sha256_file(p)
    try:
        res=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,timeout=5,check=True)
        commit=res.stdout.strip()
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        commit=None
    return {'format':'expand-e07-manifest-v1','git_commit':commit,
            'checkpoints':paths,'config':dict(config),'seeds':list(seeds),
            'abstraction_id':abstraction_id,'evaluation_script_sha256':files}


def write_new_json(path: str|Path, record: Mapping) -> None:
    """Refuse overwrite using exclusive creation; interruption may leave partial output."""
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with dest.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/integration/manifest.py" -->

## src/holdem_ai/expand/integration/source_bridges.py

Original SHA `889f28dd20f20843e255fbdf84624eb4d76a2819` (2477 bytes).

<a id="py-134"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/integration/source_bridges.py" sha="889f28dd20f20843e255fbdf84624eb4d76a2819" -->

````````python
"""Opt-in E4/E5 source bridges with explicit infoset/belief contract.

There is NO generic safe conversion from E2 PublicAction to E4 exact engine-ID
history in E1-6; caller must supply a verified matching infoset-key factory.
Never infer opponent hole cards or reconstruct hidden engine internals here.
"""
from __future__ import annotations
from typing import Callable
from holdem_ai.expand.information.schema import InformationState, ACTION_NAMES


def from_e04_checkpoint(path: str, *, exact_infoset_key:Callable[[InformationState],str]):
    from holdem_ai.expand.pluribus.blueprint.checkpoint import load_checkpoint
    solved=load_checkpoint(path)
    if solved.game.name!='holdem':
        raise ValueError('cannot deploy toy-game blueprint into Holdem')
    if not callable(exact_infoset_key):raise TypeError('verified exact_infoset_key required')
    def distribution(state:InformationState):
        if state.num_dealt_in!=solved.game.num_players:
            raise ValueError('blueprint player count incompatible with observation')
        key=exact_infoset_key(state)
        entry=solved.store.entries.get(key)
        if entry is None:raise ValueError('unseen blueprint infoset; do not pretend uniform is trained')
        available=tuple(a for a,can in zip(ACTION_NAMES,state.legal_actions.mask) if can)
        if set(entry['actions'])!=set(available):
            raise ValueError('action abstraction incompatibility: exact action set differs')
        weights=solved.store.policy(key,tuple(entry['actions']),average=True)
        table=dict(zip(entry['actions'],weights))
        return {a:table[a] for a in available}
    return distribution,solved.game.abstraction.fingerprint


def from_e05_solver(solver, *, verified_root_belief:Callable[[InformationState],object]):
    """Budget-limited search source; throws on fallback/timeout for agent router."""
    if not callable(verified_root_belief):raise TypeError('verified root belief callable required')
    def distribution(state:InformationState):
        result=solver.search(verified_root_belief(state))
        if result.status!='solved':
            raise TimeoutError(f'search incomplete: {result.status}')
        legal=tuple(a for a,yes in zip(ACTION_NAMES,state.legal_actions.mask) if yes)
        if set(result.action_probabilities)!=set(legal):
            raise ValueError('search root action abstraction incompatible')
        return dict(result.action_probabilities)
    return distribution

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/integration/source_bridges.py" -->

## src/holdem_ai/expand/integration/tournament.py

Original SHA `40ba6c58a98e71246bb4009056cf34549e877959` (12012 bytes).

<a id="py-135"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/integration/tournament.py" sha="40ba6c58a98e71246bb4009056cf34549e877959" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/integration/tournament.py" -->

## src/holdem_ai/expand/opponent/__init__.py

Original SHA `0781f89737e471d844d38978b8b5c066e8266799` (948 bytes).

<a id="py-136"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/opponent/__init__.py" sha="0781f89737e471d844d38978b8b5c066e8266799" -->

````````python
"""Expand Phase 6: public-only per-tenure opponents and bounded adaptation."""
from .schema import PublicDecision, public_decision_from_state, spr_bin
from .model import (OpponentConfig, ContextCounts, ModelEvidence, OpponentModel, OpponentRegistry)
from .adaptive import GuardConfig, AdaptationResult, SafeAdaptivePolicy, kl_divergence, tv_distance
from .persistence import save_registry,load_registry
from .bridge import PublicOpponentTracker,position_for_seat,count_prior_preflop_raises
from .router import StyleExpertRouter,StyleRoutingResult

__all__=['PublicDecision','public_decision_from_state','spr_bin','OpponentConfig','ContextCounts',
         'ModelEvidence','OpponentModel','OpponentRegistry','GuardConfig','AdaptationResult',
         'SafeAdaptivePolicy','kl_divergence','tv_distance','save_registry','load_registry','PublicOpponentTracker','position_for_seat','count_prior_preflop_raises','StyleExpertRouter','StyleRoutingResult']

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/opponent/__init__.py" -->

## src/holdem_ai/expand/opponent/adaptive.py

Original SHA `3051fef7434e35b8e11b9aa5aaa7e8b38733a661` (5634 bytes).

<a id="py-137"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/opponent/adaptive.py" sha="3051fef7434e35b8e11b9aa5aaa7e8b38733a661" -->

````````python
"""Trust-region gate around an already valid public-information policy.

This module NEVER produces a stand-alone profitable exploitative policy.
An independent candidate policy must supply `proposal`. Only then can the
opponent-conditioned evidence gate authorize a conservative interpolation.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping
from .model import ModelEvidence, OpponentConfig


@dataclass(frozen=True)
class GuardConfig:
    kl_budget: float = 0.025
    tv_budget: float = 0.075
    max_mix: float = 0.30
    confidence_pseudocount: float = 20.0
    min_base_probability: float = 0.0

    def __post_init__(self):
        for key in ('kl_budget','tv_budget','max_mix','confidence_pseudocount','min_base_probability'):
            x=getattr(self,key)
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:
                raise ValueError(f'{key} must be nonnegative finite')
        if not 0<=self.max_mix<=1 or self.min_base_probability>1:
            raise ValueError('invalid mix or support bound')


def normalize_policy(legal:tuple[str,...],raw:Mapping[str,float],*,strict:bool=True)->dict[str,float]:
    if not legal or len(set(legal))!=len(legal):raise ValueError('nonempty unique actions required')
    if strict and set(raw)!=set(legal):raise ValueError('policy keys must match legal actions exactly')
    if not isinstance(raw,Mapping):raise TypeError('policy distribution must be mapping')
    entries={a:float(raw.get(a,0.)) for a in legal}
    if any(not math.isfinite(v) or v<0 for v in entries.values()):
        raise ValueError('probabilities must be finite nonnegative')
    s=sum(entries.values())
    if s<=0:raise ValueError('no positive probability')
    return {a:v/s for a,v in entries.items()}


def kl_divergence(p:Mapping[str,float],q:Mapping[str,float])->float:
    if set(p)!=set(q):raise ValueError('mismatched policy supports')
    out=0.
    for a,x in p.items():
        if x>0:
            if q[a]<=0:return math.inf
            out+=x*math.log(x/q[a])
    return max(out,0.)


def tv_distance(p:Mapping[str,float],q:Mapping[str,float])->float:
    if set(p)!=set(q):raise ValueError('mismatched policy supports')
    return .5*sum(abs(p[a]-q[a]) for a in p)


@dataclass(frozen=True)
class AdaptationResult:
    probabilities: dict[str,float]
    used_proposal: bool
    reason: str
    mixture_weight: float
    kl_from_base: float
    tv_from_base: float
    evidence_count: float
    shift_active: bool


class SafeAdaptivePolicy:
    def __init__(self,model_config:OpponentConfig|None=None,guard:GuardConfig|None=None):
        self.model_config=model_config or OpponentConfig()
        self.guard=guard or GuardConfig()

    def adjust(self,base:Mapping[str,float],proposal:Mapping[str,float]|None,
               evidence:ModelEvidence|None,*,legal:tuple[str,...],
               expected_opponent_context:tuple|None=None)->AdaptationResult:
        """Use opponent evidence without conflating its legal set with HERO legal set.

        `proposal` is an external candidate (e.g. separate neural exploiter).
        It must use hero-visible features only; a hidden-information oracle
        must NEVER be wired into this adapter.
        """
        p=normalize_policy(legal,base)
        amount=0. if evidence is None else evidence.effective_long
        shift=False if evidence is None else evidence.shift_active
        def untouched(reason):
            return AdaptationResult(p,False,reason,0.,0.,0.,amount,shift)
        if proposal is None:return untouched('no_exploitation_proposal')
        if evidence is None:return untouched('no_opponent_evidence')
        # Opponent's observed legal set normally differs from HERO's current
        # legal set. Never compare these. The optional expected context guards
        # callers against accidentally routing an unrelated opponent sample.
        if expected_opponent_context is not None and evidence.context!=expected_opponent_context:
            return untouched('opponent_context_mismatch')
        if shift:return untouched('change_point_fallback')
        if amount<self.model_config.min_adapt_samples:return untouched('insufficient_evidence')
        q=normalize_policy(legal,proposal)
        # Never put mass on a baseline-zero action. No artificial epsilon can
        # turn an illegal/suppressed baseline action into a new supported one.
        for a in legal:
            if p[a]<=self.guard.min_base_probability:
                q[a]=0.
        if not sum(q.values()):return untouched('proposal_unsupported')
        q=normalize_policy(legal,q)
        if all(abs(p[a]-q[a])<1e-14 for a in legal):return untouched('proposal_equals_base')
        max_w=self.guard.max_mix*amount/(amount+self.guard.confidence_pseudocount) if amount else 0.
        if max_w<=0:return untouched('zero_mixture_budget')
        def blend(w):return {a:(1-w)*p[a]+w*q[a] for a in legal}
        def allowed(w):
            cand=blend(w)
            return kl_divergence(cand,p)<=self.guard.kl_budget+1e-14 and tv_distance(cand,p)<=self.guard.tv_budget+1e-14
        if not allowed(max_w):
            lo,hi=0.,max_w
            for _ in range(60):
                mid=(lo+hi)/2
                if allowed(mid):lo=mid
                else:hi=mid
            max_w=lo
        adapted=blend(max_w)
        kl=kl_divergence(adapted,p)
        tv=tv_distance(adapted,p)
        if max_w<=1e-12:return untouched('trust_region_zero')
        return AdaptationResult(adapted,True,'bounded_adaptation',max_w,kl,tv,amount,False)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/opponent/adaptive.py" -->

## src/holdem_ai/expand/opponent/bridge.py

Original SHA `4b2526201ee399b4013557420e8e797fee2c1742` (4284 bytes).

<a id="py-138"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/opponent/bridge.py" sha="4b2526201ee399b4013557420e8e797fee2c1742" -->

````````python
"""Phase1/Phase2 public event bridge, called ONLY after a legal action commits.

Caller flow:
  before = phase2.observe()
  phase2.act_abstract(action)             # may raise; no Phase6 state written
  tracker.record_accepted_action(before, handle=..., ...)  # now commit evidence

Physical seat IDs and player-session IDs must come from the trusted Phase1
seat registry and unique request_join event token (not from hole cards).
"""
from __future__ import annotations
from typing import Mapping
from holdem_ai.expand.information.schema import InformationState, PublicAction
from holdem_ai.expand.environment.table_state import HandHandle, TableSnapshot
from .model import OpponentRegistry, ModelEvidence
from .schema import PublicDecision, public_decision_from_state, RAISE_ACTIONS


def position_for_seat(handle:HandHandle,seat:int)->str:
    occupied=set(x for x,_ in handle.participants)
    if seat not in occupied:raise ValueError('seat not in dealt hand')
    if seat==handle.button_seat:return 'button'
    if seat==handle.small_blind_seat:return 'small_blind'
    if seat==handle.big_blind_seat:return 'big_blind'
    others=[]
    i=handle.big_blind_seat
    while True:
        i=(i+1)%6
        if i==handle.button_seat:break
        if i in occupied and i not in (handle.small_blind_seat,handle.big_blind_seat):
            others.append(i)
    if seat not in others:raise ValueError('nonstandard hand position')
    if seat==others[-1]:return 'cutoff'
    if seat==others[0]:return 'early'
    return 'middle'


def count_prior_preflop_raises(history:tuple[PublicAction,...])->int:
    return sum(x.street=='preflop' and (x.kind in RAISE_ACTIONS or x.kind=='raise_to') for x in history)


class PublicOpponentTracker:
    def __init__(self, registry:OpponentRegistry|None=None):
        self.registry=registry or OpponentRegistry()

    def sync_seats(self,snapshot:TableSnapshot,tenure_by_player:Mapping[str,str])->None:
        """Sync public Phase1 roster, using unique join request IDs as tenures.

        Must be called AFTER applying table-boundary events. A rejoin gets a
        new tenure even if physical seat and display player ID are unchanged.
        """
        wanted={}
        for s in snapshot.seats:
            if s.player_id is None:continue
            if s.player_id not in tenure_by_player:
                raise ValueError('tenure token not supplied for seated player')
            wanted[s.seat_id]=(s.player_id,tenure_by_player[s.player_id])
        for seat,identity in list(self.registry.active_by_seat.items()):
            if wanted.get(seat)!=identity:self.registry.unbind_seat(seat)
        for seat,identity in wanted.items():
            self.registry.bind_seat(seat,*identity)

    def record_accepted_action(self, before:InformationState, *,
                               handle:HandHandle, player_id:str,player_session_id:str,
                               action_index:int, action:str)->ModelEvidence:
        """Commit action evidence ONLY after the game engine accepts the action.

        `action` must be one of the Phase2 abstract action names and legal in
        the *before* observation. Arbitrary exact off-tree bet amounts cannot
        be assigned to this discrete action set without a separate translator.
        """
        if handle.terminal or before.hero_seat not in dict(handle.participants):
            raise ValueError('not an eligible hand decision')
        if dict(handle.participants)[before.hero_seat]!=player_id:
            raise ValueError('player ID mismatch with hand roster')
        evidence=public_decision_from_state(before,player_id=player_id,player_session_id=player_session_id,
                            hand_id=handle.hand_id,action_index=action_index,action=action,
                            position=position_for_seat(handle,before.hero_seat),
                            preflop_raises_before=count_prior_preflop_raises(before.action_history))
        return self.registry.observe(evidence)

    def close_hand(self,handle:HandHandle)->None:
        for _,player_id in handle.participants:
            for (pid,session),model in self.registry.models.items():
                if pid==player_id and handle.hand_id in model.hands:
                    model.end_hand(handle.hand_id)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/opponent/bridge.py" -->

## src/holdem_ai/expand/opponent/model.py

Original SHA `058f1f471dc675c56cd74aee29784015ca6dab2c` (9149 bytes).

<a id="py-139"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/opponent/model.py" sha="058f1f471dc675c56cd74aee29784015ca6dab2c" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/opponent/model.py" -->

## src/holdem_ai/expand/opponent/persistence.py

Original SHA `f57069dbc8a2abdb18729f26203ddf18f6a5a6e8` (5976 bytes).

<a id="py-140"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/opponent/persistence.py" sha="f57069dbc8a2abdb18729f26203ddf18f6a5a6e8" -->

````````python
"""Deterministic allowlisted JSON checkpoint, not Python pickle. Atomic writes.

The snapshot is a research artifact containing public action histories only.
Never accept untrusted checkpoint bytes as an authoritative model without
validating checksum and schema. Player IDs may nevertheless be personal data:
keep the resulting files private and/or use pseudonymous IDs.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import os
from .model import ContextCounts, OpponentConfig, OpponentModel, OpponentRegistry
from .schema import PublicDecision

FORMAT=1


def _row(c:tuple[str,str,str,tuple[str,...]],v:ContextCounts)->dict:
    return {'context':[c[0],c[1],c[2],list(c[3])], 'actions':list(v.actions),
            'long':v.long,'recent':v.recent,'raw_events':v.raw_events,
            'over_streak':v.over_streak,'shift_hold':v.shift_hold,'shift_events':v.shift_events}


def _payload(reg:OpponentRegistry)->dict:
    return {'version':FORMAT,'config':asdict(reg.config),
            'active_seats':[[k,*v] for k,v in sorted(reg.active_by_seat.items())],
            'models':[{'player_id':m.player_id,'player_session_id':m.player_session_id,
                       'contexts':[_row(c,v) for c,v in sorted(m.contexts.items())],
                       'seen_events':[list(e) for e in sorted(m.seen_events)],
                       'hands':m.hands,'finished_hands':m.finished_hands,
                       'vpip_hands':m.vpip_hands,'pfr_hands':m.pfr_hands,
                       'threebet_hands':m.threebet_hands}
                      for _,m in sorted(reg.models.items())]}


def save_registry(path:str|Path,registry:OpponentRegistry)->None:
    file=Path(path)
    if file.exists():raise FileExistsError('do not overwrite analysis checkpoints')
    file.parent.mkdir(parents=True,exist_ok=True)
    payload=_payload(registry)
    body=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    doc={'sha256':hashlib.sha256(body.encode()).hexdigest(),'payload':payload}
    temp=file.with_suffix(file.suffix+'.tmp')
    try:
        with temp.open('x',encoding='utf-8') as out:
            json.dump(doc,out,ensure_ascii=False,allow_nan=False,indent=2)
            out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(temp,file)
    finally:
        if temp.exists():temp.unlink()


def load_registry(path:str|Path)->OpponentRegistry:
    doc=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(doc,dict) or set(doc)!= {'sha256','payload'}:raise ValueError('invalid registry wrapper')
    payload=doc['payload']
    body=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    if doc['sha256']!=hashlib.sha256(body.encode()).hexdigest():raise ValueError('registry checksum mismatch')
    if payload.get('version')!=FORMAT:raise ValueError('unknown registry format')
    reg=OpponentRegistry(OpponentConfig(**payload['config']))
    for item in payload['models']:
        identity=(item['player_id'],item['player_session_id'])
        if identity in reg.models:raise ValueError('duplicate identity')
        m=OpponentModel(*identity,reg.config)
        for row in item['contexts']:
            st,pos,spr,acts=row['context']
            c=(st,pos,spr,tuple(acts))
            # Use public action schema for context validation without storing
            # private information; the observed action is a context action.
            PublicDecision(identity[0],identity[1],'validation',0,0,st,pos,spr,tuple(acts),acts[0])
            if tuple(row['actions'])!=c[-1] or c in m.contexts:raise ValueError('invalid context')
            counters=ContextCounts(c[-1],list(row['long']),list(row['recent']),
                                    row['raw_events'],row['over_streak'],row['shift_hold'],row['shift_events'])
            if (any(type(n) not in (float,int) or not math.isfinite(n) or n<0 for n in (*counters.long,*counters.recent))
                    or len(counters.long)!=len(c[-1]) or len(counters.recent)!=len(c[-1])):
                raise ValueError('invalid persisted counts')
            if any(type(getattr(counters,key)) is not int or getattr(counters,key)<0 for key in
                   ('raw_events','over_streak','shift_hold','shift_events')):raise ValueError('invalid persisted counters')
            m.contexts[c]=counters
        for event in item['seen_events']:
            if not isinstance(event,list) or len(event)!=4:raise ValueError('bad event ID')
            eid=tuple(event)
            if eid[:2]!=identity or type(eid[2]) is not str or type(eid[3]) is not int or eid[3]<0:raise ValueError('bad event identity')
            m.seen_events.add(eid)
        if len(m.seen_events)!=len(item['seen_events']) or len(m.seen_events)>reg.config.max_event_ids:
            raise ValueError('duplicate/oversized events')
        if len(m.contexts)>reg.config.max_contexts_per_player:raise ValueError('too many contexts')
        m.hands=item['hands']
        if not isinstance(m.hands,dict) or any(not isinstance(v,dict) or set(v)!= {'vpip','pfr','threebet'} or
                any(type(a) is not bool for a in v.values()) for v in m.hands.values()):
            raise ValueError('invalid hand statistics')
        for key in ('finished_hands','vpip_hands','pfr_hands','threebet_hands'):
            n=item[key]
            if type(n) is not int or n<0:raise ValueError('invalid stat')
            setattr(m,key,n)
        if max(m.vpip_hands,m.pfr_hands,m.threebet_hands)>m.finished_hands:
            raise ValueError('invalid hand stat count')
        reg.models[identity]=m
    for x in payload['active_seats']:
        seat,pid,sid=x
        if type(seat) is not int or not 0<=seat<6 or seat in reg.active_by_seat or (pid,sid) not in reg.models:
            raise ValueError('invalid active seats')
        reg.bind_seat(seat,pid,sid)
    return reg

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/opponent/persistence.py" -->

## src/holdem_ai/expand/opponent/router.py

Original SHA `712aebfe46d84e8854a56ddc7d20aa188339face` (2247 bytes).

<a id="py-141"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/opponent/router.py" sha="712aebfe46d84e8854a56ddc7d20aa188339face" -->

````````python
"""Optional model-conditioned routing among PRE-EXISTING hero policies.

Experts are externally supplied hero-visible (card-conditioned) policies.
This router NEVER calculates which poker bet is profitable and does not train
experts. Deployment requires actual held-out hand-EV validation in Phase7.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from .schema import RAISE_ACTIONS
from .model import ModelEvidence, OpponentConfig
from .adaptive import normalize_policy

STYLES=('fold_heavy','calling','aggressive')


@dataclass(frozen=True)
class StyleRoutingResult:
    style: str|None
    proposal: dict[str,float]|None
    reason: str
    style_probability: float


class StyleExpertRouter:
    def __init__(self,config:OpponentConfig|None=None,*,style_threshold:float=.55):
        self.config=config or OpponentConfig()
        if not (isinstance(style_threshold,(int,float)) and not isinstance(style_threshold,bool)
                and .5<=style_threshold<=1.):raise ValueError('style threshold must be in [.5,1]')
        self.style_threshold=float(style_threshold)

    def propose(self,evidence:ModelEvidence|None,experts:Mapping[str,Mapping[str,float]],*,
                hero_legal:tuple[str,...])->StyleRoutingResult:
        if evidence is None:return StyleRoutingResult(None,None,'no_evidence',0.)
        if evidence.shift_active:return StyleRoutingResult(None,None,'style_shift_fallback',0.)
        if evidence.effective_long<self.config.min_adapt_samples:
            return StyleRoutingResult(None,None,'insufficient_evidence',0.)
        p=evidence.posterior
        candidates={'fold_heavy':p.get('fold',0.),
                    'calling':p.get('call',0.),
                    'aggressive':sum(p.get(a,0.) for a in RAISE_ACTIONS)}
        style=max(STYLES,key=lambda k:candidates[k])
        prob=candidates[style]
        if prob<self.style_threshold:
            return StyleRoutingResult(None,None,'ambiguous_style',prob)
        if style not in experts:
            return StyleRoutingResult(style,None,'missing_expert',prob)
        candidate=normalize_policy(hero_legal,experts[style])
        return StyleRoutingResult(style,candidate,'routed_existing_expert',prob)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/opponent/router.py" -->

## src/holdem_ai/expand/opponent/schema.py

Original SHA `670e6e55f25527d02b601548caaace7d49b66a76` (4004 bytes).

<a id="py-142"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/opponent/schema.py" sha="670e6e55f25527d02b601548caaace7d49b66a76" -->

````````python
"""Allowlisted public-only events and fixed-cardinality legal-action context.

player_session_id is a *join tenure ID*, not a physical seat or table session.
A returning player must receive a new tenure, even when player_id is reused.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from holdem_ai.expand.information.schema import ACTION_NAMES, STREETS, InformationState

POSITIONS = ('button', 'small_blind', 'big_blind', 'early', 'middle', 'cutoff', 'other')
RAISE_ACTIONS = frozenset({'raise_half_pot','raise_pot','raise_double_pot','all_in'})


def spr_bin(spr: float) -> str:
    if not isinstance(spr, (float,int)) or isinstance(spr,bool) or not math.isfinite(spr) or spr < 0:
        raise ValueError('SPR must be finite nonnegative')
    return 'lt1' if spr < 1 else '1to4' if spr < 4 else '4to10' if spr < 10 else 'ge10'


@dataclass(frozen=True)
class PublicDecision:
    player_id: str
    player_session_id: str
    hand_id: str
    action_index: int
    seat_id: int
    street: str
    position: str
    spr_bucket: str
    legal: tuple[str, ...]
    action: str
    preflop_raises_before: int = 0

    def __post_init__(self):
        for x in (self.player_id,self.player_session_id,self.hand_id):
            if type(x) is not str or not x.strip():raise ValueError('nonempty identity and hand IDs required')
        if type(self.action_index) is not int or self.action_index < 0:raise ValueError('invalid action_index')
        if type(self.seat_id) is not int or not 0 <= self.seat_id < 6:raise ValueError('invalid seat')
        if self.street not in STREETS or self.position not in POSITIONS:
            raise ValueError('invalid street/position')
        if self.spr_bucket not in ('lt1','1to4','4to10','ge10'):
            raise ValueError('invalid SPR bucket')
        if (not isinstance(self.legal,tuple) or not self.legal
                or len(self.legal)!=len(set(self.legal))
                or any(a not in ACTION_NAMES for a in self.legal)
                or tuple(a for a in ACTION_NAMES if a in self.legal)!=self.legal):
            raise ValueError('legal actions must be unique in canonical order')
        if self.action not in self.legal:raise ValueError('observed action must have been legal')
        if type(self.preflop_raises_before) is not int or self.preflop_raises_before < 0:
            raise ValueError('invalid raise count')

    @property
    def identity(self) -> tuple[str,str]:
        return (self.player_id,self.player_session_id)

    @property
    def context(self) -> tuple[str,str,str,tuple[str,...]]:
        return (self.street,self.position,self.spr_bucket,self.legal)

    @property
    def event_id(self) -> tuple[str,str,str,int]:
        return (*self.identity,self.hand_id,self.action_index)


def public_decision_from_state(state: InformationState, *, player_id: str,
                               player_session_id: str, hand_id: str,
                               action_index: int, action: str, position: str,
                               preflop_raises_before: int = 0) -> PublicDecision:
    """Immediately project acting-player state into public fields ONLY.

    Caller must supply the seat's authenticated player/tenure identity and the
    publicly observable action. The state's private hero_hole is NEVER used.
    """
    if type(state) is not InformationState:
        raise TypeError('requires Phase2 InformationState, not an engine/private state')
    seat = state.seats[state.hero_seat]
    if not seat.dealt_in or not seat.occupied:raise ValueError('decision seat unavailable')
    legal = tuple(a for a,allowed in zip(ACTION_NAMES,state.legal_actions.mask) if allowed)
    stack = seat.stack_bb
    total_pot = state.pot_bb
    spr = stack / max(total_pot,1e-9)
    return PublicDecision(player_id,player_session_id,hand_id,action_index,
                          state.hero_seat,state.street,position,spr_bin(spr),legal,
                          action,preflop_raises_before)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/opponent/schema.py" -->

## src/holdem_ai/expand/pluribus/__init__.py

Original SHA `5edd06d348b06035af6eaccb95c812469f2491df` (73 bytes).

<a id="py-143"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/__init__.py" sha="5edd06d348b06035af6eaccb95c812469f2491df" -->

````````python
"""Pluribus-inspired research branch; NOT Pluribus original software."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/__init__.py" -->

## src/holdem_ai/expand/pluribus/abstraction/__init__.py

Original SHA `eff43e43b9ff875c6ff887d87cd530b56b28add7` (111 bytes).

<a id="py-144"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/abstraction/__init__.py" sha="eff43e43b9ff875c6ff887d87cd530b56b28add7" -->

````````python
from .cards import AbstractionConfig, holdem_infoset_key
__all__ = ['AbstractionConfig', 'holdem_infoset_key']

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/abstraction/__init__.py" -->
