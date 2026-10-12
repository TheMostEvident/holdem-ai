"""Four Pluribus-inspired continuation policy families.

A continuation type is sampled per player *information set*, never per hidden
world.  The Phase5 rollout estimator is an approximation of Pluribus' leaf
continuation subgame, not a full joint leaf-continuation solver.
"""
from __future__ import annotations
import math
from typing import Mapping

CONTINUATIONS = ('blueprint', 'fold_bias', 'call_bias', 'raise_bias')


def _kind(action: str) -> str:
    a = str(action).strip().lower()
    if a == 'fold': return 'fold'
    if a in ('call', 'check', 'check_call'): return 'call'
    if a in ('bet', 'all_in') or a.startswith('raise'): return 'raise'
    return 'other'


def normalize(actions: tuple[str, ...], policy: Mapping[str,float] | tuple[float,...]) -> dict[str,float]:
    if not actions or len(set(actions)) != len(actions):
        raise ValueError('requires unique nonempty actions')
    try:
        values = [float(policy[a]) for a in actions] if isinstance(policy,Mapping) else [float(x) for x in policy]
    except (KeyError,TypeError) as exc:
        raise ValueError('policy must cover legal actions') from exc
    if len(values)!=len(actions) or any(not math.isfinite(x) or x<0 for x in values):
        raise ValueError('invalid action probabilities')
    z = sum(values)
    if z<=0: raise ValueError('zero-mass policy')
    return dict(zip(actions,(x/z for x in values)))


def continuation_policy(actions: tuple[str,...], blueprint, variant: str, bias: float=5.0) -> dict[str,float]:
    if variant not in CONTINUATIONS: raise ValueError('unknown continuation')
    if not math.isfinite(bias) or bias < 1: raise ValueError('bias must be >=1')
    base = normalize(actions,blueprint)
    tag = {'fold_bias':'fold','call_bias':'call','raise_bias':'raise'}.get(variant)
    weighted = {a:p*(bias if tag is not None and _kind(a)==tag else 1.0) for a,p in base.items()}
    return normalize(actions,weighted)
