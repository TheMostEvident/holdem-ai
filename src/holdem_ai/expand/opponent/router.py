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
