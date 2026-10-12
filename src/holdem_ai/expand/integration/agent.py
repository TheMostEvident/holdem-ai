"""E7 routing: strictly actor-visible InformationState -> one legal Phase1 Action.

Policy sources are explicitly supplied/validated; neither this module nor E6
invent an exploitative policy. Prototype CFR/Search sources are opt-in and
incompatible abstractions MUST fall back instead of silently extrapolating.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
import math
import random

from holdem_ai.expand.information.schema import InformationState, ACTION_NAMES
from holdem_ai.expand.information.actions import to_table_action
from holdem_ai.expand.information.encoder import encode_observation
from holdem_ai.expand.opponent.adaptive import SafeAdaptivePolicy, normalize_policy
from holdem_ai.expand.opponent.router import StyleExpertRouter

PolicySource = Callable[[InformationState], Mapping[str, float]]


@dataclass(frozen=True)
class RoutingConfig:
    use_blueprint: bool = False
    use_search: bool = False
    use_adaptation: bool = False
    # A compatible abstraction+action mapping must be externally verified.
    verified_blueprint: bool = False
    verified_search: bool = False
    max_policy_calls: int = 2

    def __post_init__(self):
        if any(type(getattr(self, x)) is not bool for x in ('use_blueprint','use_search','use_adaptation',
                                                           'verified_blueprint','verified_search')):
            raise ValueError('routing flags must be bool')
        if type(self.max_policy_calls) is not int or self.max_policy_calls < 1:
            raise ValueError('max_policy_calls must be positive')


@dataclass(frozen=True)
class RoutedDecision:
    action_name: str
    action: object  # Phase1 Action
    probabilities: dict[str, float]
    source: str
    fallback_reasons: tuple[str, ...]
    adaptation_reason: str
    rng_seed: int


class IntegratedPokerAgent:
    """Deterministic seeded routing, support preserving fallback, no hidden inputs."""

    def __init__(self, neural: PolicySource, *, blueprint: PolicySource | None = None,
                 search: PolicySource | None = None,
                 expert_policies: Mapping[str, PolicySource] | None = None,
                 config: RoutingConfig | None = None, adaptive: SafeAdaptivePolicy | None = None,
                 expert_router: StyleExpertRouter | None = None):
        if not callable(neural):
            raise TypeError('neural fallback policy callable required')
        self.neural = neural
        self.blueprint = blueprint
        self.search = search
        self.experts = dict(expert_policies or {})
        if any(not callable(p) for p in self.experts.values()):
            raise TypeError('expert must be a callable')
        self.config = config or RoutingConfig()
        self.adaptive = adaptive or SafeAdaptivePolicy()
        self.expert_router = expert_router or StyleExpertRouter()
        self.last_decision: RoutedDecision | None = None

    def act(self, obs: InformationState, rng_seed: int, *, evidence=None) -> RoutedDecision:
        if type(obs) is not InformationState:
            raise TypeError('requires E2 InformationState; full engine state prohibited')
        if type(rng_seed) is not int or rng_seed < 0:
            raise ValueError('rng_seed must be nonnegative integer')
        legal = tuple(a for a, allowed in zip(ACTION_NAMES, obs.legal_actions.mask) if allowed)
        if not legal:
            raise ValueError('no legal action')
        reasons: list[str] = []
        selected = None
        probs = None
        candidates = (
            ('search', self.search, self.config.use_search, self.config.verified_search),
            ('blueprint', self.blueprint, self.config.use_blueprint, self.config.verified_blueprint),
            ('neural', self.neural, True, True),
        )
        calls = 0
        for name, source, enabled, verified in candidates:
            if not enabled:
                continue
            if not verified:
                reasons.append(f'{name}:unverified_abstraction')
                continue
            if source is None:
                reasons.append(f'{name}:unavailable')
                continue
            # Always reserve a chance for the trusted neural fallback, even
            # if earlier optional policy calls exhausted the optional budget.
            if name != 'neural' and calls >= self.config.max_policy_calls:
                reasons.append(f'{name}:policy_call_budget')
                continue
            calls += 1
            try:
                raw = source(obs)
                probs = normalize_policy(legal, raw, strict=True)
            except (ValueError, TypeError, RuntimeError, TimeoutError, OverflowError, ArithmeticError) as e:
                reasons.append(f'{name}:invalid_or_failed:{type(e).__name__}')
                continue
            selected = name
            break
        if selected is None or probs is None:
            # If a trusted source is broken or unavailable, do not pick illegal
            # actions or invent strategy. Uniform legal is explicitly labelled.
            selected = 'uniform_legal_emergency'
            probs = {a:1/len(legal) for a in legal}
            reasons.append('no_valid_policy_source')
        adapt_reason = 'disabled'
        if self.config.use_adaptation:
            proposals: dict[str,dict[str,float]] = {}
            for style, source in self.experts.items():
                try:
                    proposals[style] = normalize_policy(legal,source(obs))
                except (ValueError,TypeError,RuntimeError,TimeoutError,ArithmeticError):
                    reasons.append(f'adaptive_expert:{style}:invalid_or_failed')
            chosen = self.expert_router.propose(evidence, proposals, hero_legal=legal)
            adjusted = self.adaptive.adjust(probs, chosen.proposal, evidence, legal=legal)
            probs = adjusted.probabilities
            adapt_reason = f'{chosen.reason};{adjusted.reason}'
        rng = random.Random(rng_seed)
        action_name = rng.choices(legal, weights=[probs[a] for a in legal], k=1)[0]
        action = to_table_action(obs.legal_actions, action_name)
        decision = RoutedDecision(action_name, action, dict(probs), selected,
                                  tuple(reasons), adapt_reason, rng_seed)
        self.last_decision = decision
        return decision


def uniform_legal(obs: InformationState) -> dict[str,float]:
    legal = [a for a,mask in zip(ACTION_NAMES,obs.legal_actions.mask) if mask]
    return {a:1/len(legal) for a in legal}


def from_e03_checkpoint(path: str, *, device: str='cpu') -> PolicySource:
    """Phase3 E3 trust-checked neural checkpoint adapter, no hidden critic state."""
    from holdem_ai.expand.rl.trainer import load_trained_actor
    actor = load_trained_actor(path, device=device)
    def policy(obs: InformationState):
        probabilities,_ = actor.predict(encode_observation(obs))
        return {a:float(p) for a,p in zip(ACTION_NAMES,probabilities)
                if obs.legal_actions.allows(a)}
    return policy
