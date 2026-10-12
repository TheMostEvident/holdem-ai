"""Multi-player external-sampling MCCFR with linear weights and atomic iterations.

On every traversal: chance is sampled by game.start; non-traverser actions
are sampled from their current policy; traverser actions are fully branched.
We add the traverser's own-reach-weighted strategy to the average when its
infosets are traversed. This estimator samples *opponent/chance reach* and
explicitly multiplies *own reach*, avoiding opponent-hole-card exposure.

Regret deltas and average deltas are only committed after traversal succeeds;
node/infoset cap failures cannot leave half an MCCFR iteration applied.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import math
import random
from typing import Protocol


class SearchBudgetExceeded(RuntimeError):
    pass


class StateProtocol(Protocol):
    @property
    def current_player(self) -> int: ...
    def is_terminal(self) -> bool: ...
    def legal_actions(self) -> tuple[str, ...]: ...
    def infoset(self, player: int) -> str: ...
    def next_state(self, action: str): ...
    def terminal_payoffs(self) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class MCCFRConfig:
    seed: int = 400_000_004
    max_nodes_per_iteration: int = 20_000
    max_infosets: int = 200_000
    max_depth: int = 256
    linear_weighting: bool = True
    # Negative regret pruning intentionally not enabled in Phase 4 baseline.
    # Validate baseline convergence before developing biased optional pruning.
    algorithm: str = 'external_sampling_linear_mccfr_v1'

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be nonnegative integer')
        for name in ('max_nodes_per_iteration','max_infosets','max_depth'):
            if type(getattr(self,name)) is not int or getattr(self,name) <= 0:
                raise ValueError(f'{name} must be positive integer')
        if type(self.linear_weighting) is not bool:
            raise ValueError('linear_weighting must be bool')
        if self.algorithm != 'external_sampling_linear_mccfr_v1':
            raise ValueError('unsupported algorithm')


class InfosetStore:
    def __init__(self):
        self.entries: dict[str, dict] = {}

    def policy(self, key: str, actions: tuple[str, ...], *, average: bool = False) -> tuple[float, ...]:
        if not actions:
            raise ValueError('empty action set')
        entry = self.entries.get(key)
        if entry is not None and tuple(entry['actions']) != actions:
            raise AssertionError('one infoset mapped to inconsistent legal action sets')
        if entry is None:
            return tuple(1/len(actions) for _ in actions)
        values = entry['strategy_sum'] if average else [max(v,0.) for v in entry['regret_sum']]
        z = sum(values)
        return tuple(v/z for v in values) if z > 0 else tuple(1/len(actions) for _ in actions)

    def update(self, action_sets: dict, regrets: dict, averages: dict, max_infosets: int):
        if len(self.entries) + sum(k not in self.entries for k in action_sets) > max_infosets:
            raise SearchBudgetExceeded(f'max_infosets={max_infosets} exceeded')
        # All checks before mutation (iteration-level atomicity).
        for key, actions in action_sets.items():
            old = self.entries.get(key)
            if old is not None and tuple(old['actions']) != tuple(actions):
                raise AssertionError('infoset legal actions changed')
        for key, actions in action_sets.items():
            if key not in self.entries:
                self.entries[key] = {'actions':list(actions), 'regret_sum':[0.]*len(actions),
                                     'strategy_sum':[0.]*len(actions)}
        for data,field in ((regrets,'regret_sum'),(averages,'strategy_sum')):
            for key, values in data.items():
                target = self.entries[key][field]
                for i,v in enumerate(values):
                    if not math.isfinite(v):
                        raise ArithmeticError('nonfinite regret/strategy increment')
                    target[i] += v
                    if not math.isfinite(target[i]):
                        raise ArithmeticError('nonfinite cumulative table value')

    def as_dict(self):
        return {k:self.entries[k] for k in sorted(self.entries)}

    @classmethod
    def from_dict(cls, source):
        result = cls()
        for key, entry in source.items():
            acts = entry['actions']
            if not acts or len(set(acts)) != len(acts):
                raise ValueError('invalid checkpoint actions')
            regrets = entry['regret_sum']
            average = entry['strategy_sum']
            if len(acts) != len(regrets) or len(acts) != len(average):
                raise ValueError('checkpoint array lengths mismatch')
            if any(not math.isfinite(v) for v in regrets + average) or any(v < 0 for v in average):
                raise ValueError('invalid checkpoint coefficients')
            result.entries[key] = {'actions':list(acts), 'regret_sum':list(regrets),
                                   'strategy_sum':list(average)}
        return result


class MCCFRSolver:
    def __init__(self, game, config: MCCFRConfig | None = None, store: InfosetStore | None = None,
                 completed_iterations: int = 0):
        self.game = game
        self.config = config or MCCFRConfig()
        self.store = store or InfosetStore()
        self.completed_iterations = completed_iterations
        if type(completed_iterations) is not int or completed_iterations < 0:
            raise ValueError('invalid iteration')
        if not 2 <= game.num_players <= 6:
            raise ValueError('2..6 players required')

    def _iteration_rng(self, i: int) -> random.Random:
        digest = hashlib.sha256(f'e04:{self.config.seed}:{i}'.encode()).digest()
        return random.Random(int.from_bytes(digest[:16], 'big'))

    def iterate(self) -> dict:
        t = self.completed_iterations
        rng = self._iteration_rng(t)
        root = self.game.start(rng)
        player_ids = getattr(root, "player_ids", tuple(range(self.game.num_players)))
        if len(player_ids) != self.game.num_players or len(set(player_ids)) != len(player_ids):
            raise AssertionError("invalid active player IDs")
        traverser = player_ids[t % self.game.num_players]
        weight = (t // self.game.num_players + 1) if self.config.linear_weighting else 1
        action_sets = {}
        regret_deltas = {}
        average_deltas = {}
        nodes = 0
        max_depth = 0

        def vector_for(store, key, n):
            if key not in store:
                store[key] = [0.0] * n
            return store[key]

        def visit(state, own_reach: float, depth: int) -> float:
            nonlocal nodes, max_depth
            nodes += 1
            if depth > self.config.max_depth:
                raise SearchBudgetExceeded('MCCFR recursion depth guard reached; no update committed')
            max_depth = max(max_depth,depth)
            if nodes > self.config.max_nodes_per_iteration:
                raise SearchBudgetExceeded('MCCFR node cap reached; NO partial update committed')
            if state.is_terminal():
                payoff = state.terminal_payoffs()
                if len(payoff) <= max(player_ids):
                    raise AssertionError('invalid terminal payoff vector length')
                if not all(math.isfinite(p) for p in payoff) or abs(sum(payoff)) > 1e-6:
                    raise AssertionError('terminal payoff must be finite and zero-sum')
                return float(payoff[traverser])
            acting = state.current_player
            if acting not in player_ids:
                raise AssertionError('invalid actor')
            actions = state.legal_actions()
            if not actions or len(set(actions)) != len(actions):
                raise AssertionError('duplicate/empty legal actions')
            key = state.infoset(acting)
            if not isinstance(key,str) or not key:
                raise AssertionError('invalid information-set key')
            if key in action_sets and action_sets[key] != actions:
                raise AssertionError('within-iteration inconsistent infoset actions')
            action_sets[key] = actions
            if len(action_sets) + len(self.store.entries) > self.config.max_infosets * 2:
                raise SearchBudgetExceeded('too many pending infosets')
            policy = self.store.policy(key, actions)
            if acting != traverser:
                chosen = rng.choices(actions,weights=policy,k=1)[0]
                return visit(state.next_state(chosen), own_reach, depth+1)
            # Updating average at traverser infosets: visited w.r.t. sampled
            # opponent/chance reach; multiply traverser's own reach explicitly.
            avg = vector_for(average_deltas,key,len(actions))
            for a in range(len(actions)):
                avg[a] += weight * own_reach * policy[a]
            vals = [visit(state.next_state(action),own_reach*policy[j],depth+1)
                    for j,action in enumerate(actions)]
            mean = sum(p*v for p,v in zip(policy,vals))
            reg = vector_for(regret_deltas,key,len(actions))
            for j in range(len(actions)):
                reg[j] += weight*(vals[j]-mean)
            return mean

        traverser_value = visit(root,1.,0)
        self.store.update(action_sets,regret_deltas,average_deltas,self.config.max_infosets)
        self.completed_iterations += 1
        return {'iteration':self.completed_iterations,'traverser':traverser,
                'nodes':nodes,'max_depth':max_depth,'infosets':len(self.store.entries),
                'sampled_value':round(traverser_value,6), 'weight':weight}

    def train(self, additional_iterations: int, on_iteration=None):
        if type(additional_iterations) is not int or additional_iterations <= 0:
            raise ValueError('additional_iterations must be positive')
        stats = []
        for _ in range(additional_iterations):
            result = self.iterate()
            if on_iteration is not None:
                on_iteration(result)
            stats.append(result)
        return stats

    def act(self, state, rng, *, average: bool = True) -> str:
        actor = state.current_player
        key = state.infoset(actor)
        actions = state.legal_actions()
        probs = self.store.policy(key,actions,average=average)
        return rng.choices(actions,weights=probs,k=1)[0]
