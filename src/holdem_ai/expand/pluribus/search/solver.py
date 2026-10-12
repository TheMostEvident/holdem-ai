"""Bounded belief-conditioned external-sampling MCCFR with rollout leaf values.

This is a proof-of-interface SEARCH prototype, not Pluribus' complete nested
imperfect-information search: no exact leaf continuation subgame, no re-solving
from the beginning of the public betting round, no exploitability guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import time
from typing import Protocol

from ..blueprint.mccfr import InfosetStore, SearchBudgetExceeded
from .continuations import CONTINUATIONS, continuation_policy, normalize


class RootBelief(Protocol):
    hero: int
    fingerprint: str
    def sample(self,rng:random.Random): ...


@dataclass(frozen=True)
class SearchBudget:
    max_time_ms: int = 500
    max_nodes: int = 2500
    max_memory_mb: int = 64
    max_iterations: int = 80
    max_infosets: int = 1000
    depth: int = 2
    max_rollout_steps: int = 200
    blueprint_mix: float = 0.15

    def __post_init__(self):
        for name in ('max_time_ms','max_nodes','max_memory_mb','max_iterations','max_infosets',
                     'depth','max_rollout_steps'):
            value=getattr(self,name)
            if type(value) is not int or value<1: raise ValueError(f'{name} must be positive integer')
        if not (math.isfinite(self.blueprint_mix) and 0<=self.blueprint_mix<=1):
            raise ValueError('blueprint_mix must be [0,1]')


@dataclass(frozen=True)
class SearchResult:
    action_probabilities: dict[str,float]
    status: str
    iterations: int
    nodes: int
    elapsed_ms: float
    search_root_id: str
    abstraction_id: str
    continuation_samples: dict[str,int] = field(default_factory=dict)
    estimated_table_mb: float = 0.0
    reason: str = ''


class BlueprintPolicy:
    """Read-only Phase4 store, safe for off-tree action injection.

    Missing action entries have zero blueprint mass unless *all* actions are
    unrecognized, in which case they get uniform fallback. This is a heuristic
    action translation, not exact Pluribus off-tree interpolation.
    """
    def __init__(self,store:InfosetStore | None=None,abstraction_id:str='unknown'):
        self.store=store or InfosetStore()
        self.abstraction_id=abstraction_id

    def policy(self,key:str,actions:tuple[str,...])->dict[str,float]:
        if not actions:raise ValueError('empty legal set')
        entry=self.store.entries.get(key)
        if entry is not None:
            src=dict(zip(entry['actions'],self.store.policy(key,tuple(entry['actions']),average=True)))
            mass={a:src.get(a,0.) for a in actions}
            if sum(mass.values())>0:return normalize(actions,mass)
        return normalize(actions,(1.,)*len(actions))


class SubgameSolver:
    def __init__(self,blueprint:BlueprintPolicy,budget:SearchBudget | None=None,*,seed:int=500000005):
        if type(seed) is not int or seed<0: raise ValueError('seed must be nonnegative int')
        self.blueprint=blueprint
        self.budget=budget or SearchBudget()
        self.seed=seed

    def search(self,belief:RootBelief)->SearchResult:
        start=time.monotonic()
        budget=self.budget
        rid=hashlib.sha256(str(belief.fingerprint).encode()).hexdigest()[:24]
        rng=random.Random(int.from_bytes(hashlib.sha256(f'{self.seed}:{rid}'.encode()).digest()[:16],'big'))
        root=belief.sample(rng)
        if root.is_terminal() or root.current_player!=belief.hero:
            raise ValueError('root must be a nonterminal hero decision')
        root_actions=tuple(root.legal_actions())
        root_key=root.infoset(belief.hero)
        fallback=self.blueprint.policy(root_key,root_actions)
        store=InfosetStore()
        total_nodes=0
        iterations=0
        status='solved'
        reason=''
        sampled={k:0 for k in CONTINUATIONS}
        player_ids=getattr(root,'player_ids',None)
        if player_ids is None:
            n=getattr(root,'num_players',None)
            if n is None: n=len(root.terminal_payoffs()) if root.is_terminal() else None
            if n is None: raise ValueError('cannot determine player IDs')
            player_ids=tuple(range(n))
        if belief.hero not in player_ids: raise ValueError('hero seat not in game')

        def check_budget(pending=0):
            if (time.monotonic()-start)*1000>=budget.max_time_ms: raise TimeoutError('time budget')
            if total_nodes>budget.max_nodes: raise SearchBudgetExceeded('node budget')
            # Conservative upper estimate, NOT a measured RSS/peak-memory cap.
            estimate_bytes=(len(store.entries)+pending)*4096
            if estimate_bytes>budget.max_memory_mb*1024*1024:
                raise SearchBudgetExceeded('estimated table memory budget')
            if len(store.entries)+pending>budget.max_infosets:
                raise SearchBudgetExceeded('infoset budget')

        def choices(actions,probabilities):
            return rng.choices(actions,weights=[probabilities[a] for a in actions],k=1)[0]

        def rollout(state,choices_by_infoset):
            """One unbiased-deal continuation rollout with infoset-consistent variant draws."""
            steps=0
            while not state.is_terminal():
                check_budget()
                if steps>=budget.max_rollout_steps:
                    raise SearchBudgetExceeded('continuation rollout steps exceeded')
                a=state.legal_actions()
                k=state.infoset(state.current_player)
                selector=(state.current_player,k)
                if selector not in choices_by_infoset:
                    variant=rng.choice(CONTINUATIONS)
                    choices_by_infoset[selector]=variant
                    sampled[variant]+=1
                policy=continuation_policy(a,self.blueprint.policy(k,a),choices_by_infoset[selector])
                state=state.next_state(choices(a,policy))
                steps+=1
            payoffs=state.terminal_payoffs()
            if not all(math.isfinite(v) for v in payoffs):raise ArithmeticError('nonfinite rollout')
            return payoffs

        def visit(state,traverser,reach,depth,action_sets,reg,avg,leaf_selector):
            nonlocal total_nodes
            total_nodes+=1
            check_budget(len(action_sets))
            if state.is_terminal():return state.terminal_payoffs()[traverser]
            if depth>=budget.depth:
                return rollout(state,leaf_selector)[traverser]
            who=state.current_player
            if who not in player_ids:raise AssertionError('invalid player')
            actions=tuple(state.legal_actions())
            key=state.infoset(who)
            if key in action_sets and action_sets[key]!=actions:
                raise AssertionError('inconsistent infoset action set')
            action_sets[key]=actions
            prior=self.blueprint.policy(key,actions)
            local=store.policy(key,actions)
            mix={a:(1-budget.blueprint_mix)*local[i]+budget.blueprint_mix*prior[a]
                 for i,a in enumerate(actions)}
            policy=normalize(actions,mix)
            if who!=traverser:
                return visit(state.next_state(choices(actions,policy)),traverser,reach,depth+1,
                             action_sets,reg,avg,leaf_selector)
            av=avg.setdefault(key,[0.]*len(actions))
            for j,a in enumerate(actions):av[j]+=reach*policy[a]
            vals=[visit(state.next_state(a),traverser,reach*policy[a],depth+1,
                        action_sets,reg,avg,leaf_selector) for a in actions]
            mean=sum(policy[a]*v for a,v in zip(actions,vals))
            delta=reg.setdefault(key,[0.]*len(actions))
            for j,v in enumerate(vals):delta[j]+=v-mean
            return mean

        for t in range(budget.max_iterations):
            sampled_before=sampled.copy()
            if (time.monotonic()-start)*1000>=budget.max_time_ms:
                status='timeout';reason='time budget reached before iteration';break
            action_sets={}; reg={};avg={};selector={}
            try:
                check_budget()
                state=belief.sample(rng)
                if state.current_player!=belief.hero or state.is_terminal():
                    raise ValueError('belief sampled an invalid hero root')
                if tuple(state.legal_actions())!=root_actions or state.infoset(belief.hero)!=root_key:
                    raise ValueError('belief root information mismatch; no safe solve possible')
                traverser=player_ids[t%len(player_ids)]
                visit(state,traverser,1.,0,action_sets,reg,avg,selector)
                store.update(action_sets,reg,avg,budget.max_infosets)
                iterations+=1
            except TimeoutError as exc:
                sampled=sampled_before
                status='timeout';reason=str(exc);break
            except SearchBudgetExceeded as exc:
                sampled=sampled_before
                status='resource_guard';reason=str(exc);break
        else:
            status='solved'
        if iterations:
            # Average at root is populated only on traverser=hero iterations.
            entry=store.entries.get(root_key)
            if entry is not None and sum(entry['strategy_sum'])>0:
                probs=normalize(root_actions,store.policy(root_key,root_actions,average=True))
            else:
                probs=fallback
        else:
            probs=fallback
            if status=='solved':status='fallback';reason='no completed search iterations'
        return SearchResult(probs,status,iterations,total_nodes,
                            round((time.monotonic()-start)*1000,3),rid,
                            self.blueprint.abstraction_id,sampled,
                            round(len(store.entries)*4096/2**20,5),reason)
