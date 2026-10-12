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
