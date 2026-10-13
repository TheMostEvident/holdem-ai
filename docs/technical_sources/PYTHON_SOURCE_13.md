# Complete Python source volume 13/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## src/holdem_ai/expand/pluribus/abstraction/cards.py

Original blob `586220dbb5a22e32dd8bbe88cfed78841621f513` (2718 bytes).

<a name="py-145"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/abstraction/cards.py" sha="586220dbb5a22e32dd8bbe88cfed78841621f513" -->

````````python
"""Reproducible (deliberately *untrained*) toy 6-max hold'em abstraction.

Preflop hero cards are EXACT. Postflop hash buckets are an infrastructure
prototype, NOT Pluribus' equity-based EMD/learned card buckets. The public
board and betting history are always exact and opponent private cards excluded.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class AbstractionConfig:
    postflop_buckets: int = 16
    allow_half_pot: bool = False
    schema: str = 'e04-abstraction-v1-hash-prototype'

    def __post_init__(self):
        if type(self.postflop_buckets) is not int or not 2 <= self.postflop_buckets <= 4096:
            raise ValueError('postflop_buckets must be 2..4096')
        if type(self.allow_half_pot) is not bool:
            raise TypeError('allow_half_pot must be bool')

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(encoded).hexdigest()[:24]


def holdem_infoset_key(raw: dict, actions: tuple[str, ...], config: AbstractionConfig) -> str:
    """Only use information supplied by engine.observe(current_actor).

    The engine observation is assumed actor-safe; do not inspect the engine's
    .hole_cards, ._deck, or ._rng. Strict allowlist in the key generator.
    """
    hero = raw['player_id']
    hole = tuple(raw['hand'])
    board = tuple(raw['public_cards'])
    if len(hole) != 2 or not 0 <= hero < 6:
        raise ValueError('invalid acting seat / private hand')
    if len(set((*hole, *board))) != 2 + len(board):
        raise ValueError('duplicate visible cards')
    street = raw['street']
    if street == 'preflop':
        if board:
            raise ValueError('preflop cannot expose board')
        private = ['exact', *sorted(hole)]
    elif street in ('flop', 'turn', 'river'):
        digest = hashlib.sha256('|'.join(sorted(hole) + list(board)).encode()).digest()
        bucket = int.from_bytes(digest[:8], 'big') % config.postflop_buckets
        private = ['prototype_bucket', bucket]
    else:
        raise ValueError('unknown street')
    # Full public information is represented, but no private rival cards.
    public = [
        'e04-holdem-v1', config.fingerprint, hero, street, private, list(board),
        raw['button'], list(raw['stacks']), list(raw['street_bets']),
        list(raw['total_contributions']), list(raw['status']),
        raw['pot'], raw['to_call'], raw['min_raise_to'],
        list(raw['action_history']), list(raw['action_raise_to']), list(actions),
    ]
    return json.dumps(public, separators=(',', ':'), ensure_ascii=True)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/abstraction/cards.py" -->

## src/holdem_ai/expand/pluribus/blueprint/__init__.py

Original blob `88683de2746b43859b6c3aae7ce0ec88bc8d8e60` (380 bytes).

<a name="py-146"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/__init__.py" sha="88683de2746b43859b6c3aae7ce0ec88bc8d8e60" -->

````````python
from .mccfr import MCCFRSolver, MCCFRConfig, InfosetStore, SearchBudgetExceeded
from .toy_games import KuhnGame, LeducGame
from .holdem import HoldemGame
from .checkpoint import save_checkpoint, load_checkpoint
__all__ = ['MCCFRSolver', 'MCCFRConfig', 'InfosetStore', 'SearchBudgetExceeded',
           'KuhnGame', 'LeducGame', 'HoldemGame', 'save_checkpoint', 'load_checkpoint']

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/__init__.py" -->

## src/holdem_ai/expand/pluribus/blueprint/checkpoint.py

Original blob `5d93c88573c56f5e3be21194713b9dbbd8eea8f0` (3419 bytes).

<a name="py-147"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/checkpoint.py" sha="5d93c88573c56f5e3be21194713b9dbbd8eea8f0" -->

````````python
"""Atomic gzipped JSON MCCFR checkpoints; no pickle or torch.load."""
from __future__ import annotations
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
from .mccfr import MCCFRSolver, MCCFRConfig, InfosetStore
from .toy_games import KuhnGame, LeducGame
from .holdem import HoldemGame
from ..abstraction import AbstractionConfig


def describe_game(game):
    data = {'name':game.name, 'players':game.num_players}
    if game.name == 'holdem':
        data.update({'big_blind':game.big_blind,'stack_bb':game.stack_bb,
                     'abstraction':asdict(game.abstraction),
                     'abstraction_id':game.abstraction.fingerprint})
    return data


def build_game(spec, engine_factory=None):
    if spec['name'] == 'kuhn':
        return KuhnGame(spec['players'])
    if spec['name'] == 'leduc':
        return LeducGame(spec['players'])
    if spec['name'] == 'holdem':
        abstraction = AbstractionConfig(**spec['abstraction'])
        if spec['abstraction_id'] != abstraction.fingerprint:
            raise ValueError('card/action abstraction fingerprint mismatch')
        return HoldemGame(spec['players'],big_blind=spec['big_blind'],
                          stack_bb=spec['stack_bb'],abstraction=abstraction,
                          engine_factory=engine_factory)
    raise ValueError('unsupported game in checkpoint')


def save_checkpoint(path, solver: MCCFRSolver):
    file = Path(path)
    file.parent.mkdir(parents=True,exist_ok=True)
    payload = {'phase':'expand04','schema':1,
               'game':describe_game(solver.game),
               'config':asdict(solver.config),
               'completed_iterations':solver.completed_iterations,
               'entries':solver.store.as_dict()}
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    outer = {'sha256':hashlib.sha256(raw).hexdigest(),'payload':payload}
    temp = file.with_name(file.name + '.tmp')
    with temp.open('wb') as handle:
        with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0) as archive:
            archive.write(json.dumps(outer,sort_keys=True,separators=(',', ':'),allow_nan=False).encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp,file)
    return {'path':str(file),'sha256':hashlib.sha256(raw).hexdigest(),
            'completed_iterations':solver.completed_iterations,'infosets':len(solver.store.entries)}


def load_checkpoint(path, engine_factory=None):
    with gzip.open(path,'rt',encoding='utf-8') as handle:
        outer=json.load(handle)
    payload=outer['payload']
    raw=json.dumps(payload,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest()!=outer['sha256']:
        raise ValueError('checksum mismatch: checkpoint tampered or corrupt')
    if payload.get('phase')!='expand04' or payload.get('schema') != 1:
        raise ValueError('incompatible checkpoint schema')
    game=build_game(payload['game'], engine_factory=engine_factory)
    cfg=MCCFRConfig(**payload['config'])
    table=InfosetStore.from_dict(payload['entries'])
    if len(table.entries)>cfg.max_infosets:
        raise ValueError('checkpoint exceeds memory/infoset guard')
    return MCCFRSolver(game,config=cfg,store=table,
                       completed_iterations=payload['completed_iterations'])

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/checkpoint.py" -->

## src/holdem_ai/expand/pluribus/blueprint/evaluate.py

Original blob `f67ea0836c1bba88f15c0cdafdbef8a255c0cafd` (2020 bytes).

<a name="py-148"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/evaluate.py" sha="f67ea0836c1bba88f15c0cdafdbef8a255c0cafd" -->

````````python
"""Separate deterministic evaluation seeds; no trainer updates in evaluation."""
from __future__ import annotations
import hashlib
import math
import random
from .mccfr import MCCFRSolver


def evaluate_vs_uniform(solver: MCCFRSolver, *, hands: int = 100, seed: int = 440_000_000,
                        mode: str = 'average') -> dict:
    if type(hands) is not int or hands < 2:
        raise ValueError('at least two evaluation hands required')
    if mode not in ('average','current'):
        raise ValueError('policy mode must be average/current')
    scores = []
    rng = random.Random(seed)
    for _ in range(hands):
        hand_seed = rng.getrandbits(120)
        state = solver.game.start(random.Random(hand_seed))
        ids = getattr(state,'player_ids',tuple(range(solver.game.num_players)))
        hero = ids[rng.randrange(len(ids))]
        actor_rng = random.Random(rng.getrandbits(120))
        steps = 0
        while not state.is_terminal():
            steps += 1
            if steps > 256:
                raise RuntimeError('game not terminating; check game engine')
            actions = state.legal_actions()
            if state.current_player == hero:
                key=state.infoset(hero)
                p=solver.store.policy(key,actions,average=(mode=='average'))
                action=actor_rng.choices(actions,weights=p,k=1)[0]
            else:
                action=actor_rng.choice(actions)
            state=state.next_state(action)
        scores.append(float(state.terminal_payoffs()[hero]))
    mean=sum(scores)/len(scores)
    var=sum((x-mean)**2 for x in scores)/(len(scores)-1)
    return {'mode':mode,'hands':hands,'seed':seed,'mean_reward':round(mean,6),
            'ci95_lower':round(mean-1.96*math.sqrt(var/len(scores)),6),
            'ci95_upper':round(mean+1.96*math.sqrt(var/len(scores)),6),
            'unit':'BB/hand' if solver.game.name=='holdem' else 'toy_chips/hand',
            'note':'uniform-opponent exploratory evaluation, NOT Nash exploitability'}

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/evaluate.py" -->

## src/holdem_ai/expand/pluribus/blueprint/holdem.py

Original blob `dd4b3bfa01d662fc76bcc402745710b9a310866e` (4709 bytes).

<a name="py-149"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/holdem.py" sha="dd4b3bfa01d662fc76bcc402745710b9a310866e" -->

````````python
"""Small-resource 2--6 seat Hold'em MCCFR adapter using the Phase3 engine.

No hidden-card access while forming infoset. Complete chance deals occur in
engine constructor/reset; trees branch via copy.deepcopy on legal actions.
This is a bounded research prototype; entire 6-max full tree is intractable.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
from typing import Callable
from ..abstraction.cards import AbstractionConfig, holdem_infoset_key


@dataclass
class HoldemState:
    engine: object
    abstraction: AbstractionConfig
    big_blind: int

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(i for i,stack in enumerate(self.engine.initial_stacks) if stack > 0)

    @property
    def current_player(self) -> int:
        return self.engine.current_player

    def is_terminal(self):
        return self.engine.is_over()

    def _obs(self):
        if self.is_terminal():
            raise RuntimeError('terminal hand')
        return self.engine.observe(self.current_player)

    def _mapping(self):
        obs = self._obs()
        ids = obs['legal_actions']
        raw = obs['raw_obs']
        items = []
        if 0 in ids:
            items.append(('fold',0))
        if 1 in ids:
            items.append(('check' if raw['to_call'] == 0 else 'call',1))
        if self.abstraction.allow_half_pot and 2 in ids:
            items.append(('raise_half_pot',2))
        if 3 in ids:
            items.append(('raise_pot',3))
        if 4 in ids:
            # Avoid multiple abstract actions that produce the same actual amount.
            hero = self.current_player
            maxbet = raw['street_bets'][hero] + raw['stacks'][hero]
            tocall = raw['to_call']
            pot_after_call = raw['pot'] + tocall
            target_pot = raw['street_bets'][hero] + tocall + pot_after_call
            target_half = raw['street_bets'][hero] + tocall + pot_after_call//2
            other_targets = ({target_half} if 2 in ids and self.abstraction.allow_half_pot else set())
            if 3 in ids:
                other_targets.add(target_pot)
            if maxbet not in other_targets:
                items.append(('all_in',4))
        return dict(items)

    def legal_actions(self):
        return tuple(self._mapping())

    def infoset(self, player: int):
        if player != self.current_player:
            raise ValueError('not your information state')
        return holdem_infoset_key(self._obs()['raw_obs'],self.legal_actions(),self.abstraction)

    def next_state(self,action: str):
        mapping = self._mapping()
        if action not in mapping:
            raise ValueError('illegal abstract action')
        engine = copy.deepcopy(self.engine)
        engine.step(mapping[action])
        return HoldemState(engine,self.abstraction,self.big_blind)

    def terminal_payoffs(self):
        if not self.is_terminal():
            raise RuntimeError('not terminal')
        values = self.engine.get_result().payoffs_chips
        if len(values) != 6:
            raise AssertionError('must use six physical seats')
        return tuple(v/self.big_blind for v in values)


class HoldemGame:
    def __init__(self, num_players: int = 2, *, big_blind: int = 2, stack_bb: int = 30,
                 abstraction: AbstractionConfig | None = None,
                 engine_factory: Callable | None = None):
        if type(num_players) is not int or not 2 <= num_players <= 6:
            raise ValueError('num_players must be 2..6')
        if type(big_blind) is not int or big_blind < 2:
            raise ValueError('big_blind must be >=2')
        if type(stack_bb) is not int or not 2 <= stack_bb <= 300:
            raise ValueError('stack_bb must be 2..300')
        self.name = 'holdem'
        self.num_players = num_players
        self.big_blind = big_blind
        self.stack_bb = stack_bb
        self.abstraction = abstraction or AbstractionConfig()
        self.engine_factory = engine_factory

    def start(self,rng) -> HoldemState:
        factory = self.engine_factory
        if factory is None:
            from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine
            factory = MultiPlayerHandEngine
        participants = sorted(rng.sample(range(6), self.num_players))
        stacks = [self.stack_bb*self.big_blind if i in participants else 0 for i in range(6)]
        button = rng.choice(participants)
        engine = factory(stacks=stacks,button=button,small_blind=self.big_blind//2,
                         big_blind=self.big_blind,seed=rng.randrange(2**63))
        engine.reset()
        return HoldemState(engine,self.abstraction,self.big_blind)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/holdem.py" -->

## src/holdem_ai/expand/pluribus/blueprint/kuhn_reference.py

Original blob `e74109b06dc74e54a5190cb862afa4db1599c349` (2042 bytes).

<a name="py-150"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/kuhn_reference.py" sha="e74109b06dc74e54a5190cb862afa4db1599c349" -->

````````python
"""Exact 2-player Kuhn exploitability (unabstracted 3-card game only).

For each player, enumerate deterministic response mappings on that player's
information sets; expectation over all six uniformly likely 3P2 deals and
opponent's averaged behavioral strategy. Small toy game only; NOT a 6-max
Nash/exploitability metric.
"""
from __future__ import annotations
from itertools import permutations, product
from .toy_games import ToyState


def _root(hole):
    return ToyState('kuhn',tuple(hole),None,0,(1,1),(0,0),(True,True),(0,1),False,())


def _value(state, player, store, mapping):
    if state.is_terminal():
        return state.terminal_payoffs()[player]
    acting=state.current_player
    actions=state.legal_actions()
    key=state.infoset(acting)
    if acting==player:
        return _value(state.next_state(mapping[key]),player,store,mapping)
    probs=store.policy(key,actions,average=True)
    return sum(p*_value(state.next_state(a),player,store,mapping)
               for p,a in zip(probs,actions))


def _collect(state, player, choices):
    if state.is_terminal():
        return
    if state.current_player==player:
        choices[state.infoset(player)]=state.legal_actions()
    for action in state.legal_actions():
        _collect(state.next_state(action),player,choices)


def kuhn_nash_conv(store):
    """Return max_resp(player0) + max_resp(player1); NashConv >= 0."""
    deals=tuple(permutations(range(3),2))
    values=[]
    for player in (0,1):
        choices={}
        for deal in deals:
            _collect(_root(deal),player,choices)
        keys=sorted(choices)
        best=float('-inf')
        for mapping_actions in product(*(choices[key] for key in keys)):
            mapping=dict(zip(keys,mapping_actions))
            value=sum(_value(_root(deal),player,store,mapping) for deal in deals)/len(deals)
            best=max(best,value)
        values.append(best)
    return {'nash_conv':sum(values),'best_response_values':values,
            'note':'Kuhn-only exact, not 6-max exploitability'}

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/kuhn_reference.py" -->

## src/holdem_ai/expand/pluribus/blueprint/mccfr.py

Original blob `e2a901bbb981a355290561ce785f92cc6fd9be69` (10626 bytes).

<a name="py-151"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/mccfr.py" sha="e2a901bbb981a355290561ce785f92cc6fd9be69" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/mccfr.py" -->

## src/holdem_ai/expand/pluribus/blueprint/toy_games.py

Original blob `df2d91eeef7b1b60e052a7625f50bda6839be058` (6048 bytes).

<a name="py-152"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/toy_games.py" sha="df2d91eeef7b1b60e052a7625f50bda6839be058" -->

````````python
"""Tiny 2/3-player toy poker games for reference testing MCCFR.

Kuhn: one betting round, 1-chip bet, no raises, 3-card deck for 2p,
4-card deck for 3p. Leduc: two streets, paired ranks J/Q/K, fixed 1/2
chip bet, max one bet per street, one shared card. No peek at rival cards.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ToyState:
    game: Literal['kuhn', 'leduc']
    hole: tuple[int, ...]  # Entire deck is engine-owned, NOT returned in infoset.
    board: int | None  # Chance outcome pre-sampled; hidden in preflop infoset.
    street: int
    contributions: tuple[int, ...]
    invested: tuple[int, ...]
    active: tuple[bool, ...]
    pending: tuple[int, ...]
    raised: bool
    history: tuple[tuple[int, int, str], ...]

    @property
    def num_players(self):
        return len(self.hole)

    @property
    def current_player(self) -> int:
        return self.pending[0] if self.pending else -1

    def is_terminal(self) -> bool:
        return not self.pending

    def legal_actions(self) -> tuple[str, ...]:
        if self.is_terminal():
            return ()
        i = self.current_player
        if self.invested[i] < max(self.invested):
            return ('fold', 'call')
        if self.raised:
            # can only reach here if no outstanding action; next street/terminal.
            raise AssertionError('betting round should be closed')
        return ('check', 'bet')

    def infoset(self, player: int) -> str:
        if player != self.current_player:
            raise ValueError('only acting player may view private observation')
        import json
        card = self.hole[player] if self.game == 'kuhn' else self.hole[player] // 2
        visible_board = self.board // 2 if self.game == 'leduc' and self.street == 1 else None
        return json.dumps([self.game, self.num_players, player, card, visible_board,
                           list(self.active), list(self.contributions),
                           list(self.history), list(self.legal_actions())], separators=(',', ':'))

    def next_state(self, action: str) -> 'ToyState':
        if action not in self.legal_actions():
            raise ValueError('illegal toy action')
        i = self.current_player
        active = list(self.active)
        contrib = list(self.contributions)
        invested = list(self.invested)
        pending = list(self.pending[1:])
        raised = self.raised
        if action == 'fold':
            active[i] = False
        elif action == 'call':
            difference = max(invested) - invested[i]
            invested[i] += difference
            contrib[i] += difference
        elif action == 'bet':
            stake = (2 if self.game == 'leduc' and self.street == 1 else 1)
            invested[i] += stake
            contrib[i] += stake
            raised = True
            # Previous checkers must now respond to the outstanding bet.
            pending = [p for p in range(self.num_players) if p != i and active[p]]
            pending.sort(key=lambda p: (p - i) % self.num_players)
        newhist = (*self.history, (self.street, i, action))
        if sum(active) == 1:
            return ToyState(self.game,self.hole,self.board,self.street,tuple(contrib),
                            tuple(invested),tuple(active),(),raised,newhist)
        if pending:
            return ToyState(self.game,self.hole,self.board,self.street,tuple(contrib),
                            tuple(invested),tuple(active),tuple(pending),raised,newhist)
        if self.game == 'leduc' and self.street == 0:
            return ToyState(self.game,self.hole,self.board,1,tuple(contrib),
                            tuple(0 for _ in contrib),tuple(active),
                            tuple(p for p in range(self.num_players) if active[p]),False,newhist)
        return ToyState(self.game,self.hole,self.board,self.street,tuple(contrib),
                        tuple(invested),tuple(active),(),raised,newhist)

    def terminal_payoffs(self) -> tuple[float, ...]:
        if not self.is_terminal():
            raise RuntimeError('no terminal payoff before end')
        active = [p for p, yes in enumerate(self.active) if yes]
        if len(active) == 1:
            winners = active
        elif self.game == 'kuhn':
            best = max(self.hole[i] for i in active)
            winners = [i for i in active if self.hole[i] == best]
        else:
            assert self.board is not None
            boardrank = self.board // 2
            ranks = {i: (int(self.hole[i] // 2 == boardrank), self.hole[i] // 2) for i in active}
            best = max(ranks.values())
            winners = [i for i in active if ranks[i] == best]
        pot = sum(self.contributions)
        result = [-float(c) for c in self.contributions]
        for winner in winners:
            result[winner] += pot / len(winners)
        if abs(sum(result)) > 1e-9:
            raise AssertionError('toy poker zero-sum violated')
        return tuple(result)


class KuhnGame:
    def __init__(self, num_players: int = 2):
        if num_players not in (2,3):
            raise ValueError('KuhnGame supports 2 or 3 players')
        self.num_players = num_players
        self.name = 'kuhn'

    def start(self, rng) -> ToyState:
        cards = rng.sample(range(self.num_players + 1), self.num_players)
        n = self.num_players
        return ToyState('kuhn',tuple(cards),None,0,(1,)*n,(0,)*n,(True,)*n,
                        tuple(range(n)),False,())


class LeducGame:
    def __init__(self, num_players: int = 2):
        if num_players not in (2,3):
            raise ValueError('LeducGame supports 2 or 3 players')
        self.num_players = num_players
        self.name = 'leduc'

    def start(self, rng) -> ToyState:
        cards = rng.sample(range(6),self.num_players+1)
        n = self.num_players
        return ToyState('leduc',tuple(cards[:n]),cards[-1],0,(1,)*n,(0,)*n,
                        (True,)*n,tuple(range(n)),False,())

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/blueprint/toy_games.py" -->

## src/holdem_ai/expand/pluribus/search/__init__.py

Original blob `9c8bae37c7c80e19da874217fcc1d6f84fe89327` (558 bytes).

<a name="py-153"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/search/__init__.py" sha="9c8bae37c7c80e19da874217fcc1d6f84fe89327" -->

````````python
"""Expand Phase5 bounded imperfect-information search prototype."""
from .solver import BlueprintPolicy, SearchBudget, SearchResult, SubgameSolver
from .belief import ToyRootBelief, HoldemObservationBelief, InconsistentPublicHistory
from .continuations import CONTINUATIONS,continuation_policy
from .nested import NestedSearchSession
__all__=['BlueprintPolicy','SearchBudget','SearchResult','SubgameSolver',
         'ToyRootBelief','HoldemObservationBelief','InconsistentPublicHistory',
         'CONTINUATIONS','continuation_policy','NestedSearchSession']

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/search/__init__.py" -->

## src/holdem_ai/expand/pluribus/search/belief.py

Original blob `6c1bc570c2ff08d18a26cae8ed58e47d39525ceb` (8095 bytes).

<a name="py-154"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/search/belief.py" sha="6c1bc570c2ff08d18a26cae8ed58e47d39525ceb" -->

````````python
"""Public-knowledge-only roots for toy poker and collision-free Hold'em sampling."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import random


class InconsistentPublicHistory(ValueError):
    pass


@dataclass(frozen=True)
class ToyRootBelief:
    """Reconstruct a toy game *from public actions + one hero private card*.

    No access to the observed world's rival cards. Opponent actions are replayed
    as public facts. For this MVP, unobserved hands are sampled uniformly given
    public cards (action likelihood updates are not automatic).
    """
    game: object
    hero: int
    hero_card: int
    history: tuple[str,...] = ()
    board_rank: int | None = None
    max_attempts: int = 10000

    def sample(self,rng:random.Random):
        for _ in range(self.max_attempts):
            state=self.game.start(rng)
            own=state.hole[self.hero] if state.game=='kuhn' else state.hole[self.hero]//2
            if own!=self.hero_card: continue
            try:
                for a in self.history:
                    state=state.next_state(a)
            except (ValueError,AssertionError):
                continue
            visible = state.board//2 if state.game=='leduc' and state.street==1 else None
            if self.board_rank is not None and visible!=self.board_rank: continue
            if state.is_terminal(): continue
            return state
        raise InconsistentPublicHistory('cannot sample root satisfying toy public constraints')

    @property
    def fingerprint(self):
        return repr((self.game.name,self.game.num_players,self.hero,self.hero_card,self.history,self.board_rank))


SUITS='cdhs'
RANKS='23456789TJQKA'
DECK=frozenset(s+r for s in SUITS for r in RANKS)


def _card(c):
    if not isinstance(c,str) or c not in DECK:
        raise ValueError(f'invalid canonical card {c!r}')
    return c


class HoldemObservationBelief:
    """Sample complete Phase3 engine worlds consistent with public observations.

    Inputs are an acting player's raw_obs (not an engine instance), start-of-
    hand stacks, blind sizes, and action history already visible in raw_obs.
    Hole cards for opponents and unexposed board/burn cards are NEW samples.
    All dealt cards are collision-free. Any replay mismatch is rejected.
    """
    def __init__(self, raw_obs:dict, *, initial_stacks:tuple[int,...],
                 small_blind:int, big_blind:int, engine_factory:Callable | None=None,
                 max_attempts:int=200, opponent_likelihood:Callable | None=None,
                 abstraction=None):
        self.raw = dict(raw_obs)
        self.stacks = tuple(initial_stacks)
        self.sb = small_blind
        self.bb = big_blind
        self.factory = engine_factory
        self.max_attempts = max_attempts
        self.opponent_likelihood = opponent_likelihood
        from ..abstraction import AbstractionConfig
        self.abstraction = abstraction if abstraction is not None else AbstractionConfig()
        if not isinstance(self.abstraction, AbstractionConfig):raise TypeError('abstraction must be AbstractionConfig')
        if len(self.stacks)!=6 or sum(s>0 for s in self.stacks)<2:
            raise ValueError('six physical seats, at least 2 occupied required')
        if any(type(s) is not int or s<0 for s in self.stacks):
            raise ValueError('invalid starting stacks')
        if not (type(small_blind) is type(big_blind) is int and 0<small_blind<big_blind):
            raise ValueError('invalid blinds')
        self.hero=int(self.raw['player_id']); self.button=int(self.raw['button'])
        self.hole=tuple(_card(c) for c in self.raw['hand'])
        self.board=tuple(_card(c) for c in self.raw['public_cards'])
        if len(self.hole)!=2 or len(self.board) not in (0,3,4,5) or len(set(self.hole+self.board))!=len(self.hole+self.board):
            raise ValueError('duplicate or invalid known cards')
        self.occupied=tuple(i for i,s in enumerate(self.stacks) if s>0)
        if self.hero not in self.occupied or self.button not in self.occupied:
            raise ValueError('hero and button must be seated')
        self.actions=tuple(self.raw['action_history'])
        self.raise_to=tuple(self.raw['action_raise_to'])
        if len(self.actions)!=len(self.raise_to): raise ValueError('invalid action trace')
        if type(max_attempts) is not int or max_attempts<1: raise ValueError('max_attempts must be positive')
        if len(self.raw['stacks'])!=6 or len(self.raw['street_bets'])!=6:
            raise ValueError('invalid observation vectors')
        # Publicly visible fingerprint only (hero cards are own permitted info).
        self.fingerprint=repr((self.hero,self.hole,self.board,self.button,self.stacks,
                               self.actions,self.raise_to,self.raw['street'],tuple(self.raw['stacks'])))

    def _random_deck(self,rng):
        order=[]; nextseat=self.button
        for _ in self.occupied:
            nextseat=next(i for offset in range(1,7) if (i:=(nextseat+offset)%6) in self.occupied)
            order.append(nextseat)
        known={order.index(self.hero):self.hole[0],len(order)+order.index(self.hero):self.hole[1]}
        n=len(order)
        if len(self.board)>=3:
            for j,c in enumerate(self.board[:3]):known[2*n+1+j]=c
        if len(self.board)>=4:known[2*n+5]=self.board[3]
        if len(self.board)==5:known[2*n+7]=self.board[4]
        # 2n hole, burn(2n), flop(2n+1..3), burn(2n+4), turn(2n+5), burn(2n+6), river(2n+7)
        remaining=list(DECK-set(known.values()))
        remaining.sort(); rng.shuffle(remaining)
        it=iter(remaining)
        return [known.get(j) if j in known else next(it) for j in range(52)]

    def sample(self,rng:random.Random):
        factory=self.factory
        if factory is None:
            from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine
            factory=MultiPlayerHandEngine
        for _ in range(self.max_attempts):
            engine=factory(stacks=self.stacks,button=self.button,
                           small_blind=self.sb,big_blind=self.bb,deck=self._random_deck(rng))
            engine.reset()
            weight=1.0
            try:
                for (seat,action),amount in zip(self.actions,self.raise_to):
                    if engine.is_over() or engine.current_player!=seat:
                        raise InconsistentPublicHistory('action seat mismatch')
                    if self.opponent_likelihood is not None and seat!=self.hero:
                        p=float(self.opponent_likelihood(engine.observe(seat),action,amount))
                        if not math.isfinite(p) or p<0 or p>1:
                            raise ValueError('opponent likelihood outside [0,1]')
                        weight*=p
                    if action==5:
                        engine.step_raise_to(amount)
                    else:
                        engine.step(action)
                if engine.is_over() or engine.current_player!=self.hero:
                    raise InconsistentPublicHistory('hero turn mismatch')
                out=engine.observe(self.hero)['raw_obs']
                for key in ('hand','public_cards','street','button','player_id','stacks',
                            'street_bets','total_contributions','status','pot','to_call',
                            'action_history','action_raise_to'):
                    if out[key]!=self.raw[key]:
                        raise InconsistentPublicHistory(f'public replay mismatch: {key}')
                # Reject likelihood weighted particles without changing their hidden states.
                if self.opponent_likelihood is not None and rng.random()>=weight:
                    continue
                from ..blueprint.holdem import HoldemState
                return HoldemState(engine,self.abstraction,self.bb)
            except (InconsistentPublicHistory,ValueError,AssertionError):
                continue
        raise InconsistentPublicHistory('could not reconstruct public observation')

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/search/belief.py" -->

## src/holdem_ai/expand/pluribus/search/continuations.py

Original blob `f7ab3eae453f8228bf67909f0a20c3a0b5a72230` (1934 bytes).

<a name="py-155"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/search/continuations.py" sha="f7ab3eae453f8228bf67909f0a20c3a0b5a72230" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/search/continuations.py" -->

## src/holdem_ai/expand/pluribus/search/nested.py

Original blob `a26758b034671f99b132e51533c90d9333d561cd` (1862 bytes).

<a name="py-156"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/search/nested.py" sha="a26758b034671f99b132e51533c90d9333d561cd" -->

````````python
"""Public-trace session for re-searching at each decision / off-tree action.

Phase5 does not solve the entire previous street from its beginning: this
replays factual public history, then re-solves the *current* information set.
Own committed actions are immutable once registered.
"""
from __future__ import annotations
from dataclasses import dataclass,field


@dataclass
class NestedSearchSession:
    hero: int
    street: str='preflop'
    history: list[tuple[int,str,int|None]]=field(default_factory=list)
    own_commitments: dict[tuple[str,str],str]=field(default_factory=dict)
    off_tree_count: int=0
    street_resets: int=0

    def observe_action(self,*,actor:int,action:str,street:str,infoset_key:str|None=None,
                       raise_to:int|None=None,abstract_actions:tuple[str,...]=()):
        if not isinstance(action,str) or not action:raise ValueError('action required')
        if action=='raise_to' and (type(raise_to) is not int or raise_to<=0):
            raise ValueError('off-tree raise_to requires positive integer target')
        if street!=self.street:
            self.street=street;self.street_resets+=1
        commitment=(street,infoset_key or '')
        if actor==self.hero:
            if not infoset_key:raise ValueError('hero commitment requires infoset')
            if commitment in self.own_commitments:
                if self.own_commitments[commitment]!=action:
                    raise ValueError('cannot revise committed own action')
                raise ValueError('duplicate hero public action')
            self.own_commitments[commitment]=action
        if action=='raise_to' and action not in abstract_actions:
            self.off_tree_count+=1
        self.history.append((actor,action,raise_to))

    def public_trace(self)->tuple[tuple[int,str,int|None],...]:
        return tuple(self.history)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/search/nested.py" -->
