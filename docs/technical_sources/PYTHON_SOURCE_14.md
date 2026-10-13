# Complete Python source volume 14/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## src/holdem_ai/expand/pluribus/search/solver.py

Original blob `b1065e420f128a699e9e60bc04b90dd7b9dfeafa` (9865 bytes).

<a name="py-157"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/pluribus/search/solver.py" sha="b1065e420f128a699e9e60bc04b90dd7b9dfeafa" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/pluribus/search/solver.py" -->

## src/holdem_ai/expand/rl/__init__.py

Original blob `b3b852ff22834154779e3ce70017712dadbdd4e3` (782 bytes).

<a name="py-158"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/rl/__init__.py" sha="b3b852ff22834154779e3ce70017712dadbdd4e3" -->

````````python
"""Expand Phase 3: multiagent PPO on a public-information six-seat game."""
from .model import ActorCritic, ValueV2
from .collector import MultiAgentCollector, RolloutConfig, RolloutResult
from .returns import MultiAgentTransition, PreparedSample, prepare_gae
from .population import OpponentMix, scripted_action
from .ppo import PPOConfig, optimize_ppo

__all__ = ["ActorCritic", "ValueV2", "MultiAgentCollector", "RolloutConfig", "RolloutResult",
           "MultiAgentTransition", "PreparedSample", "prepare_gae", "OpponentMix", "scripted_action",
           "PPOConfig", "optimize_ppo"]
from .trainer import TrainConfig, build_model, train, evaluate_checkpoint, load_trained_actor
__all__.extend(['TrainConfig','build_model','train','evaluate_checkpoint','load_trained_actor'])

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/rl/__init__.py" -->

## src/holdem_ai/expand/rl/collector.py

Original blob `ad20c8894db0db2c17aac828c511ef3003681117` (10450 bytes).

<a name="py-159"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/rl/collector.py" sha="ad20c8894db0db2c17aac828c511ef3003681117" -->

````````python
"""Real-hand multiagent on-policy self-play on E1 table + E2 allowlisted adapter.

One frozen version of the trainable policy collects an entire update before PPO
mutates its weights. No private deck or opponent hole cards reach the model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Callable

from holdem_ai.expand.environment import DynamicTable, TableConfig
from holdem_ai.expand.information import Phase2TableAdapter
from holdem_ai.expand.information.encoder import encode_observation
from holdem_ai.expand.information.schema import ACTION_NAMES

from .model import ActorCritic
from .population import OpponentMix, choose_role, scripted_action
from .returns import MultiAgentTransition


@dataclass(frozen=True)
class RolloutConfig:
    hands_per_update: int = 64
    num_players: tuple[int,...] = (2,3,4,5,6)
    starting_stack_bb: int = 100
    random_stack_range_bb: tuple[int,int] | None = None
    max_decisions: int = 350
    max_warmup_hands: int = 100
    mix: OpponentMix = OpponentMix()

    def __post_init__(self):
        if type(self.hands_per_update) is not int or self.hands_per_update < 1:
            raise ValueError('hands_per_update must be positive')
        if not self.num_players or any(type(n) is not int or n not in (2,3,4,5,6) for n in self.num_players):
            raise ValueError('num_players choices must be 2..6')
        if type(self.starting_stack_bb) is not int or not 20 <= self.starting_stack_bb <= 300:
            raise ValueError('starting_stack_bb must be 20..300')
        if self.random_stack_range_bb is not None:
            lo,hi=self.random_stack_range_bb
            if any(type(v) is not int for v in (lo,hi)) or not 20 <= lo <= hi <= 300:
                raise ValueError('random stack range must be 20..300')
        if self.max_decisions < 1 or self.max_warmup_hands < 1:
            raise ValueError('positive rollout guards required')


@dataclass(frozen=True)
class RolloutResult:
    transitions: tuple[MultiAgentTransition,...]
    total_hands: int
    hands_by_n: dict[int,int]
    learner_payoff_bb: float
    illegal_actions: int
    warmup_hands: int
    hand_rewards: tuple[tuple[float,...], ...]


class MultiAgentCollector:
    def __init__(self, model: ActorCritic, config: RolloutConfig, *,
                 engine_factory: Callable | None = None,
                 table_config: TableConfig | None = None):
        self.model=model
        self.config=config
        self.engine_factory=engine_factory
        self.table_config=table_config or TableConfig()
        if self.table_config.mode != 'cash' or self.table_config.rake_bps:
            raise ValueError('Expand Phase3 supports rake-free cash games only')

    def _fresh_table(self, n: int, rng: random.Random, run_tag: str):
        table=DynamicTable(self.table_config,session_id=run_tag,engine_factory=self.engine_factory)
        adapter=Phase2TableAdapter(table)
        for seat in range(n):
            amount=self.config.starting_stack_bb
            if self.config.random_stack_range_bb is not None:
                amount=rng.randint(*self.config.random_stack_range_bb)
            table.request_join('player_'+str(seat),amount*self.table_config.big_blind,preferred_seat=seat)
        return adapter

    def _act(self, adapter: Phase2TableAdapter, rng: random.Random, *,
             roles: dict[int,str], snapshots: tuple[ActorCritic,...],
             version: str, collect_learning: bool,
             out: list[MultiAgentTransition]) -> None:
        decisions=0
        local=[]
        while adapter.table.current_player_id is not None:
            if decisions >= self.config.max_decisions:
                raise RuntimeError('E3 exceeded per-hand decision limit')
            st=adapter.observe()
            role=roles.get(st.hero_seat,'scripted:tight')
            if role == 'trainable':
                encoded=encode_observation(st)
                probabilities,value=self.model.predict(encoded)
                action=_sample_action(probabilities,rng)
                if collect_learning:
                    local.append(MultiAgentTransition(
                        session_id=adapter.table.session_id,hand_id=adapter._hand_id,
                        player_id=adapter.table.current_player_id,seat_id=st.hero_seat,
                        observation=st,encoded=encoded,action=ACTION_NAMES[action],
                        action_logprob=math.log(max(probabilities[action],1e-38)),
                        value=value,reward_bb=0.0,terminal_hand=False,terminal_session=False,
                        policy_version=version,num_players=st.num_dealt_in))
                action_name=ACTION_NAMES[action]
            elif role.startswith('snapshot:'):
                j=int(role.split(':',1)[1])
                probabilities,_=snapshots[j].predict(encode_observation(st))
                action_name=ACTION_NAMES[_sample_action(probabilities,rng)]
            elif role.startswith('scripted:'):
                action_name=scripted_action(role.split(':',1)[1],st,rng)
            else:
                raise RuntimeError('unknown role')
            if not st.legal_actions.allows(action_name):
                raise AssertionError(f'illegal action selected: {action_name}')
            adapter.act_abstract(action_name)
            decisions+=1
        settled=adapter.settle_hand()
        if sum(settled.payoffs_chips)!=0:
            raise AssertionError('nonzero sum E1 result')
        # One terminal reward per (hand, seat), even if decision streams of
        # different players are interleaved and seats have distinct timelines.
        last_by_seat={}
        for i,t in enumerate(local): last_by_seat[t.seat_id]=i
        for i,t in enumerate(local):
            is_last=(last_by_seat[t.seat_id]==i)
            local[i]=replace(t,terminal_hand=is_last,
                             reward_bb=float(settled.payoffs_bb[t.seat_id]) if is_last else 0.0)
        out.extend(local)
        return settled

    def collect(self, *, seed: int, policy_version: str,
                snapshots: tuple[ActorCritic,...] = (),
                num_hands: int | None = None,
                force_n: int | None = None,
                force_opponent: str | None = None,
                record_training: bool = True) -> RolloutResult:
        if type(seed) is not int or seed < 0:
            raise ValueError('nonnegative seed required')
        total=self.config.hands_per_update if num_hands is None else num_hands
        if type(total) is not int or total < 1:
            raise ValueError('positive total hands required')
        if force_n is not None and force_n not in range(2,7):
            raise ValueError('force_n must be 2..6')
        if force_opponent is not None and force_opponent not in ('random','calling_station','tight','pot_pressure'):
            raise ValueError('unknown heldout opponent')
        # Independent streams: policy sampling never changes the number-of-
        # players schedule, planned buyin amounts or the per-hand deck RNG.
        rng_schedule=random.Random(seed ^ 0x31C31)
        rng_setup=random.Random(seed ^ 0x7EAB1)
        rng_deck=random.Random(seed ^ 0xDECAFE)
        rng_roles=random.Random(seed ^ 0x11A7)
        rng_actions=random.Random(seed ^ 0x51EAD)
        self.model.eval()
        for snapshot in snapshots: snapshot.eval()
        adapters={}
        restart={}
        hands_by_n={n:0 for n in range(2,7)}
        out=[]
        rewards=[]
        warmup=0
        payoff=0.0
        while sum(hands_by_n.values())<total:
            n=force_n if force_n is not None else rng_schedule.choice(self.config.num_players)
            if n not in adapters:
                restart[n]=0
                adapters[n]=self._fresh_table(n,rng_setup,f'e03-v{policy_version}-n{n}-r0')
            adapter=adapters[n]
            hand=adapter.start_hand(rng_deck.getrandbits(60))
            if hand is None:
                raise RuntimeError('table cannot start a hand with two funded seats')
            if len(hand.participants)!=n:
                warmup+=1
                if warmup>self.config.max_warmup_hands * (len(self.config.num_players)+1):
                    raise RuntimeError('BB admissions did not converge before guard')
                # Keep the queue of joining players intact, but do not train on
                # partial roster hands. The warmup game is fully settled.
                self._act(adapter,rng_actions,roles={},snapshots=(),version=policy_version,
                          collect_learning=False,out=[])
                continue
            roles={0:'trainable'}
            for seat,_ in hand.participants:
                if seat==0: continue
                if force_opponent is not None:
                    roles[seat]='scripted:'+force_opponent
                else:
                    picked=choose_role(rng_roles,self.config.mix,has_snapshots=bool(snapshots))
                    if picked=='snapshot':
                        picked='snapshot:'+str(rng_roles.randrange(len(snapshots)))
                    roles[seat]=picked
            prior=len(out)
            settled=self._act(adapter,rng_actions,roles=roles,snapshots=snapshots,
                              version=policy_version,collect_learning=record_training,out=out)
            hands_by_n[n]+=1
            rewards.append(tuple(float(x) for x in settled.payoffs_bb))
            payoff+=sum(settled.payoffs_bb[s] for s,v in roles.items() if v=='trainable')
            if record_training and not any(t.seat_id==0 for t in out[prior:]):
                # It is possible for a stack to be all-in from forced blinds.
                # This case is valid; its payoff is not attributed to a decision.
                pass
            snap=adapter.table.snapshot()
            seated=sum(s.player_id is not None and s.stack_chips>0 for s in snap.seats[:n])
            if seated<n:
                restart[n]+=1
                adapters[n]=self._fresh_table(n,rng_setup,f'e03-v{policy_version}-n{n}-r{restart[n]}')
        return RolloutResult(tuple(out),total,hands_by_n,payoff,0,warmup,tuple(rewards))


def _sample_action(probs: list[float],rng: random.Random) -> int:
    u=rng.random()
    cumul=0.0
    for j,p in enumerate(probs):
        cumul+=p
        if u<cumul: return j
    return next(i for i in range(len(probs)-1,-1,-1) if probs[i]>0)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/rl/collector.py" -->

## src/holdem_ai/expand/rl/model.py

Original blob `44b46e2be61aa7c2c62760b9e2a4d36b0c749a85` (2440 bytes).

<a name="py-160"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/rl/model.py" sha="44b46e2be61aa7c2c62760b9e2a4d36b0c749a85" -->

````````python
"""Expand 3: strictly actor-visible shared multi-seat Actor + Value networks."""
from __future__ import annotations

import torch
from torch import nn

from holdem_ai.expand.information.encoder import EncodedObservation, GLOBAL_DIM, SEAT_DIM, HISTORY_DIM
from holdem_ai.expand.information.policy_v2 import PolicyV2


class ValueV2(nn.Module):
    """Unprivileged value baseline: receives only EncodedObservation."""
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GLOBAL_DIM + SEAT_DIM + HISTORY_DIM, width),
            nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, observations: list[EncodedObservation]) -> torch.Tensor:
        if not observations or any(not isinstance(o, EncodedObservation) for o in observations):
            raise TypeError('ValueV2 requires list[EncodedObservation]')
        device = next(self.parameters()).device
        g = torch.tensor([o.global_features for o in observations], dtype=torch.float32, device=device)
        seats = torch.tensor([o.seat_features for o in observations], dtype=torch.float32, device=device)
        sm = torch.tensor([o.seat_mask for o in observations], dtype=torch.float32, device=device)
        histories = torch.tensor([o.history_features for o in observations], dtype=torch.float32, device=device)
        hm = torch.tensor([o.history_mask for o in observations], dtype=torch.float32, device=device)
        seat_pool = (seats * sm.unsqueeze(-1)).sum(1) / sm.sum(1, keepdim=True).clamp(min=1)
        history_pool = (histories * hm.unsqueeze(-1)).sum(1) / hm.sum(1, keepdim=True).clamp(min=1)
        return self.net(torch.cat((g, seat_pool, history_pool), dim=-1)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, *, width: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.width, self.heads, self.layers = width, heads, layers
        self.actor = PolicyV2(width=width, heads=heads, layers=layers)
        self.value = ValueV2(width=width)

    def forward(self, observations: list[EncodedObservation]):
        return self.actor(observations), self.value(observations)

    @torch.no_grad()
    def predict(self, observation: EncodedObservation):
        self.eval()
        logits, values = self([observation])
        return logits[0].softmax(-1).detach().cpu().tolist(), float(values[0].item())

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/rl/model.py" -->

## src/holdem_ai/expand/rl/population.py

Original blob `d6dac794a39a593a780ed66c3638d1135ab32f5c` (2015 bytes).

<a name="py-161"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/rl/population.py" sha="d6dac794a39a593a780ed66c3638d1135ab32f5c" -->

````````python
"""Training-time opponent selection, separate from final evaluation opponents."""
from __future__ import annotations

import random
from dataclasses import dataclass

from holdem_ai.expand.information.schema import InformationState, ACTION_NAMES


@dataclass(frozen=True)
class OpponentMix:
    shared_policy_fraction: float = 0.25
    snapshot_fraction: float = 0.25
    scripted_names: tuple[str, ...] = ('random', 'calling_station', 'tight', 'pot_pressure')

    def __post_init__(self):
        if not (0 <= self.shared_policy_fraction <= 1 and 0 <= self.snapshot_fraction <= 1
                and self.shared_policy_fraction + self.snapshot_fraction <= 1):
            raise ValueError('fractions must be nonnegative with sum <=1')
        if not self.scripted_names or any(x not in ('random','calling_station','tight','pot_pressure')
                                           for x in self.scripted_names):
            raise ValueError('unsupported scripted policy')


def scripted_action(name: str, state: InformationState, rng: random.Random) -> str:
    legal = [a for a, available in zip(ACTION_NAMES, state.legal_actions.mask) if available]
    if not legal:
        raise ValueError('no legal action')
    if name == 'random':
        return rng.choice(legal)
    if name == 'calling_station':
        order = ('check', 'call', 'fold')
    elif name == 'tight':
        order = ('check', 'fold', 'call')
    elif name == 'pot_pressure':
        order = ('raise_pot', 'raise_double_pot', 'raise_half_pot', 'all_in', 'call', 'check', 'fold')
    else:
        raise ValueError('unknown scripted opponent')
    return next((a for a in order if a in legal), legal[0])


def choose_role(rng: random.Random, mix: OpponentMix, *, has_snapshots: bool) -> str:
    u = rng.random()
    if u < mix.shared_policy_fraction:
        return 'trainable'
    if u < mix.shared_policy_fraction + mix.snapshot_fraction and has_snapshots:
        return 'snapshot'
    return 'scripted:' + rng.choice(mix.scripted_names)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/rl/population.py" -->

## src/holdem_ai/expand/rl/ppo.py

Original blob `c7db716af6555a88edb2017fd7722de33a8e369c` (3780 bytes).

<a name="py-162"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/rl/ppo.py" sha="c7db716af6555a88edb2017fd7722de33a8e369c" -->

````````python
"""Clipped PPO on frozen collected 2-6 player trajectories."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch.distributions import Categorical

from .returns import PreparedSample
from .model import ActorCritic
from holdem_ai.expand.information.schema import ACTION_NAMES


@dataclass(frozen=True)
class PPOConfig:
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 0.0003
    epochs: int = 2
    minibatch_size: int = 64
    max_grad_norm: float = 1.0
    target_kl: float = 0.20

    def __post_init__(self):
        if not (0 < self.clip_epsilon < 1 and self.value_coef >= 0 and self.entropy_coef >= 0
                and self.learning_rate > 0 and self.epochs > 0 and self.minibatch_size > 0
                and self.max_grad_norm > 0 and self.target_kl > 0):
            raise ValueError('invalid PPO hyperparameters')


def optimize_ppo(model: ActorCritic, optimizer: torch.optim.Optimizer,
                 samples: tuple[PreparedSample, ...], config: PPOConfig) -> dict:
    if not samples:
        raise ValueError('cannot optimize empty batch')
    model.train()
    device = next(model.parameters()).device
    advantages = torch.tensor([s.advantage for s in samples], dtype=torch.float32,device=device)
    if len(samples) > 1:
        advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    returns = torch.tensor([s.return_bb for s in samples], dtype=torch.float32,device=device)
    old_lp = torch.tensor([s.transition.action_logprob for s in samples],dtype=torch.float32,device=device)
    actions = torch.tensor([ACTION_NAMES.index(s.transition.action) for s in samples],dtype=torch.int64,device=device)
    stats=[]
    early_stop=False
    for epoch in range(config.epochs):
        for indices in torch.randperm(len(samples),device='cpu').split(config.minibatch_size):
            indices = indices.to(device)
            obs = [samples[int(i)].transition.encoded for i in indices.cpu().tolist()]
            logits, v = model(obs)
            dist=Categorical(logits=logits)
            new_lp=dist.log_prob(actions[indices])
            log_ratio=new_lp-old_lp[indices]
            ratio=log_ratio.exp()
            adv=advantages[indices]
            loss_pi=-torch.minimum(ratio*adv, ratio.clamp(1-config.clip_epsilon,1+config.clip_epsilon)*adv).mean()
            loss_v=torch.square(v-returns[indices]).mean()
            entropy=dist.entropy().mean()
            loss=loss_pi+config.value_coef*loss_v-config.entropy_coef*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite PPO loss: checkpoint should not be saved')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError('nonfinite PPO gradient: optimizer step aborted')
            optimizer.step()
            # Not an unbiased KL estimator, but a common PPO proxy.
            approx_kl=float(((ratio-1)-log_ratio).mean().detach().item())
            stats.append((float(loss_pi.detach()), float(loss_v.detach()),float(entropy.detach()),approx_kl))
            if approx_kl > config.target_kl:
                early_stop=True
                break
        if early_stop:
            break
    def avg(i): return sum(x[i] for x in stats)/len(stats)
    return {'actor_loss':round(avg(0),6), 'value_loss':round(avg(1),6),
            'entropy':round(avg(2),6),'approx_kl':round(avg(3),6),
            'optimizer_steps':len(stats), 'early_kl_stop':early_stop,
            'samples':len(samples)}

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/rl/ppo.py" -->

## src/holdem_ai/expand/rl/returns.py

Original blob `f7b7a098691bcd9e8afe4103160187784efeb8e9` (2513 bytes).

<a name="py-163"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/rl/returns.py" sha="f7b7a098691bcd9e8afe4103160187784efeb8e9" -->

````````python
"""Per-hand, per-seat GAE: decision streams are never conflated across players."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from holdem_ai.expand.information.schema import InformationState
from holdem_ai.expand.information.encoder import EncodedObservation


@dataclass(frozen=True)
class MultiAgentTransition:
    session_id: str
    hand_id: str
    player_id: str
    seat_id: int
    observation: InformationState
    encoded: EncodedObservation
    action: str
    action_logprob: float
    value: float
    reward_bb: float
    terminal_hand: bool
    terminal_session: bool
    policy_version: str
    num_players: int


@dataclass(frozen=True)
class PreparedSample:
    transition: MultiAgentTransition
    advantage: float
    return_bb: float


def prepare_gae(transitions: Sequence[MultiAgentTransition], *, gamma: float = 1.0,
                lam: float = 0.95) -> tuple[PreparedSample, ...]:
    if not (0 <= gamma <= 1 and 0 <= lam <= 1):
        raise ValueError('gamma and lambda must be in [0,1]')
    if not transitions:
        raise ValueError('empty transition batch')
    groups: dict[tuple[str, str, int], list[int]] = {}
    for k, t in enumerate(transitions):
        key = t.session_id, t.hand_id, t.seat_id
        groups.setdefault(key, []).append(k)
    out: list[PreparedSample | None] = [None]*len(transitions)
    for key, positions in groups.items():
        if not transitions[positions[-1]].terminal_hand:
            raise ValueError(f'unterminated player trajectory {key}')
        if sum(t.terminal_hand for t in (transitions[k] for k in positions)) != 1:
            raise ValueError('player trajectory has inconsistent terminal flag')
        if any(transitions[k].terminal_hand for k in positions[:-1]):
            raise ValueError('terminal action occurs before last player decision')
        gae = 0.0
        for offset in range(len(positions)-1, -1, -1):
            k=positions[offset]
            t=transitions[k]
            nxt = 0.0 if offset == len(positions)-1 else transitions[positions[offset+1]].value
            # The hand has no intermediate chip reward; the last decision
            # receives the entire seat's signed settled BB payoff.
            delta = t.reward_bb + gamma * nxt * (not t.terminal_hand) - t.value
            gae = delta + gamma * lam * (not t.terminal_hand) * gae
            out[k] = PreparedSample(t, gae, gae+t.value)
    return tuple(out)  # type: ignore[return-value]

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/rl/returns.py" -->

## src/holdem_ai/expand/rl/trainer.py

Original blob `5cf2a43277fc964448ae67b840dd450fe473f6e8` (16291 bytes).

<a name="py-164"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/expand/rl/trainer.py" sha="5cf2a43277fc964448ae67b840dd450fe473f6e8" -->

````````python
"""Reproducible E3 PPO training, snapshots, safe resume and validation gates.

All training/evaluation card seeds are separated. Never report E3 as a Pluribus
blueprint or as evidence of converged multi-player equilibrium strategies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Callable

import torch

from .collector import MultiAgentCollector, RolloutConfig
from .model import ActorCritic
from .population import OpponentMix
from .ppo import PPOConfig, optimize_ppo
from .returns import prepare_gae


@dataclass(frozen=True)
class TrainConfig:
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    width: int = 64
    heads: int = 4
    layers: int = 2
    seed: int = 303000
    rollout_seed: int = 320000000
    validation_seed: int = 330000000
    eval_every: int = 2
    eval_hands: int = 8
    snapshot_every: int = 2
    max_snapshots: int = 3
    gamma: float = 1.0
    gae_lambda: float = 0.95

    def __post_init__(self):
        if self.width < 16 or self.heads < 1 or self.width % self.heads or self.layers < 1:
            raise ValueError('invalid actor network configuration')
        if any(type(x) is not int or x < 0 for x in (self.seed,self.rollout_seed,self.validation_seed)):
            raise ValueError('nonnegative seeds required')
        if self.eval_every <= 0 or self.eval_hands < 1 or self.snapshot_every <= 0 or self.max_snapshots < 0:
            raise ValueError('invalid eval/snapshot cadence')
        if not (0<=self.gamma<=1 and 0<=self.gae_lambda<=1):
            raise ValueError('invalid GAE coefficients')
        # Validation bank and training bank must not overlap for up to 1M updates.
        if abs(self.rollout_seed-self.validation_seed) < 1_000_000:
            raise ValueError('rollout and validation seed domains too close')


def config_from_dict(data: dict) -> TrainConfig:
    rd=dict(data['rollout'])
    rd['mix']=OpponentMix(**{**rd['mix'],'scripted_names':tuple(rd['mix']['scripted_names'])})
    rd['num_players']=tuple(rd['num_players'])
    if rd['random_stack_range_bb'] is not None:
        rd['random_stack_range_bb']=tuple(rd['random_stack_range_bb'])
    return TrainConfig(**{**data,'rollout':RolloutConfig(**rd),'ppo':PPOConfig(**data['ppo'])})


def _atomic_json(path: Path, record: Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp,path)


def _atomic_pt(path: Path, data: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp,path)


def _hash(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):
            h.update(x)
    return h.hexdigest()


def _log(path: Path, data: dict):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def build_model(cfg: TrainConfig, device: str) -> ActorCritic:
    if device not in ('cpu','cuda'):
        raise ValueError('device cpu or cuda required')
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    return ActorCritic(width=cfg.width,heads=cfg.heads,layers=cfg.layers).to(device)


def _checkpoint(model,optimizer,cfg,update, snapshots,best_score, best_update):
    return {'format':'expand-e03-v1','model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'config':asdict(cfg),
            'completed_updates':update,'snapshots':list(snapshots),
            'best_validation_score':best_score,'best_update':best_update,
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def load_trained_actor(path: str | Path, *, device='cpu') -> ActorCritic:
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format') not in ('expand-e03-v1','expand-e03-snapshot-v1'):
        raise ValueError('not an Expand Phase3 trusted checkpoint')
    cfg=config_from_dict(payload['config'])
    model=build_model(cfg,device)
    model.load_state_dict(payload['model'])
    return model.eval()


def _evaluate(model: ActorCritic, cfg: TrainConfig, *, device: str,
              num_hands: int | None = None, seed: int | None = None,
              engine_factory: Callable | None = None) -> dict:
    """Frozen random/calling_station anchor; no training transitions recorded."""
    scores={}
    for j,name in enumerate(('random','calling_station')):
        collector=MultiAgentCollector(model,cfg.rollout,engine_factory=engine_factory)
        result=collector.collect(seed=(cfg.validation_seed+j*100000 if seed is None else seed+j*100000),
                                  policy_version='eval',num_hands=(num_hands or cfg.eval_hands),
                                  force_opponent=name,record_training=False)
        scores[name]=round(result.learner_payoff_bb/result.total_hands*100,5)
    return scores


def _train_impl(root: str | Path, *, config: TrainConfig | None = None, resume: bool = False,
          additional_updates: int = 3, device: str = 'cpu',
          engine_factory: Callable | None = None, emit: Callable[[dict],None] | None = None) -> dict:
    if type(additional_updates) is not int or additional_updates < 1:
        raise ValueError('additional_updates >=1 required')
    folder=Path(root)
    folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'run.json'
    latest=folder/'latest.pt'
    best=folder/'best.pt'
    state_file=folder/'state.json'
    logfile=folder/'metrics.jsonl'
    if resume:
        if not all(x.exists() for x in (manifest,latest,state_file,logfile,best)):
            raise FileNotFoundError('incomplete run; no resume possible')
        record=json.loads(manifest.read_text(encoding='utf-8'))
        cfg=config_from_dict(record['config'])
        if config is not None and asdict(cfg)!=asdict(config):
            raise ValueError('resume config differs from run.json')
        blob=torch.load(latest,map_location='cpu',weights_only=True)
        if blob.get('format')!='expand-e03-v1' or blob['config']!=asdict(cfg):
            raise ValueError('checkpoint manifest mismatch')
        # Verify validation-selected best does not change silently.
        saved_state=json.loads(state_file.read_text(encoding='utf-8'))
        if saved_state['best_sha256']!=_hash(best):
            raise ValueError('best checkpoint hash differs from stored state')
    else:
        if any(p.name != '.expand03.lock' for p in folder.iterdir()):
            raise FileExistsError('fresh training root must be empty; use --resume or a new folder')
        cfg=config or TrainConfig()
        torch.manual_seed(cfg.seed)
        if device=='cuda':torch.cuda.manual_seed_all(cfg.seed)
        _atomic_json(manifest,{'phase':'expand-03','format':'expand-e03-v1','config':asdict(cfg)})
        blob=None
    model=build_model(cfg,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.ppo.learning_rate)
    if resume:
        model.load_state_dict(blob['model'])
        optimizer.load_state_dict(blob['optimizer'])
        # Map optimizer tensors to actual device (cpu checkpoint -> CUDA resume).
        for group in optimizer.state.values():
            for key,val in group.items():
                if torch.is_tensor(val):group[key]=val.to(device)
        torch.set_rng_state(blob['torch_rng'])
        if device=='cuda' and blob.get('cuda_rng') and len(blob['cuda_rng'])==torch.cuda.device_count():
            torch.cuda.set_rng_state_all(blob['cuda_rng'])
        updates=int(blob['completed_updates'])
        snapshots=list(blob['snapshots'])
        best_score=float(blob['best_validation_score'])
        best_update=int(blob['best_update'])
    else:
        updates=0
        snapshots=[]
        baseline=_evaluate(model,cfg,device=device,engine_factory=engine_factory)
        best_score=0.0
        best_update=0
        _atomic_json(folder/'validation_anchor.json',baseline)
        initial=_checkpoint(model,optimizer,cfg,updates,snapshots,best_score,best_update)
        _atomic_pt(latest,initial)
        _atomic_pt(best,initial)
        _atomic_json(state_file,{'completed_updates':0,'best_update':0,
                                 'best_score_bb100':0.0,'best_sha256':_hash(best)})
        _log(logfile,{'kind':'baseline','update':0,'vs_opponent_bb100':baseline})
    baseline=json.loads((folder/'validation_anchor.json').read_text(encoding='utf-8'))
    since=time.perf_counter()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    run_history=[]
    for _ in range(additional_updates):
        next_u=updates+1
        frozen=[]
        for p in snapshots:
            path=folder/p
            if not path.is_file():raise FileNotFoundError('missing frozen snapshot '+p)
            frozen.append(load_trained_actor(path,device=device))
        collector=MultiAgentCollector(model,cfg.rollout,engine_factory=engine_factory)
        data=collector.collect(seed=cfg.rollout_seed+next_u,policy_version=f'update-{updates}',
                               snapshots=tuple(frozen))
        if not data.transitions:
            raise RuntimeError('no trainable decisions from collected hands')
        samples=prepare_gae(data.transitions,gamma=cfg.gamma,lam=cfg.gae_lambda)
        stats=optimize_ppo(model,optimizer,samples,cfg.ppo)
        updates=next_u
        validated=None
        if updates % cfg.eval_every == 0:
            current=_evaluate(model,cfg,device=device,engine_factory=engine_factory)
            gain=sum(current[k]-baseline[k] for k in ('random','calling_station'))/2
            validated={'current':current,'score_delta_bb100':round(gain,5)}
            if gain>best_score:
                best_score=gain
                best_update=updates
                best_blob=_checkpoint(model,optimizer,cfg,updates,snapshots,best_score,best_update)
                _atomic_pt(best,best_blob)
        if cfg.max_snapshots and updates%cfg.snapshot_every==0:
            snap_dir=folder/'snapshots'
            snap_dir.mkdir(exist_ok=True)
            rel=f'snapshots/update_{updates:07d}.pt'
            _atomic_pt(folder/rel,{'format':'expand-e03-snapshot-v1','config':asdict(cfg),
                                   'model':model.state_dict(),'completed_updates':updates})
            snapshots.append(rel)
            if len(snapshots)>cfg.max_snapshots:
                # Old snapshots intentionally retained on disk for audit. They
                # are only removed from active opponent sampling population.
                snapshots=snapshots[-cfg.max_snapshots:]
        blob=_checkpoint(model,optimizer,cfg,updates,snapshots,best_score,best_update)
        _atomic_pt(latest,blob)
        _atomic_json(state_file,{'completed_updates':updates,'best_update':best_update,
                                 'best_score_bb100':round(best_score,5),'best_sha256':_hash(best)})
        row={'kind':'update','update':updates,'hands':data.total_hands,
             'num_players_histogram':data.hands_by_n,'warmup_hands':data.warmup_hands,
             'learner_payoff_bb':round(data.learner_payoff_bb,5),
             'illegal_actions':data.illegal_actions,**stats}
        if validated is not None:row['validation']=validated
        _log(logfile,row)
        run_history.append(row)
        if emit:emit(row)
    summary={'phase':'expand-03','completed_updates':updates,
             'updates_this_call':additional_updates,'best_update':best_update,
             'best_score_delta_bb100':round(best_score,5),
             'latest_checkpoint':str(latest),'best_checkpoint':str(best),
             'duration_seconds':round(time.perf_counter()-since,3),
             'cuda_peak_allocated_mb':round(torch.cuda.max_memory_allocated()/2**20,3) if device=='cuda' else None,
             'last_update':run_history[-1]}
    _atomic_json(folder/'summary.json',summary)
    return summary


def evaluate_checkpoint(checkpoint: str | Path, *, device: str='cpu',
                        hands: int=100, seed: int=340000000,
                        opponents: tuple[str,...]=('random','calling_station','tight','pot_pressure'),
                        num_players: tuple[int,...]=(2,3,4,5,6),
                        engine_factory: Callable | None = None) -> dict:
    if type(hands) is not int or hands < 1 or type(seed) is not int or seed < 0:
        raise ValueError('invalid evaluation settings')
    actor=load_trained_actor(checkpoint,device=device)
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=config_from_dict(raw['config'])
    results={}
    for n in num_players:
        for j,name in enumerate(opponents):
            if name not in ('random','calling_station','tight','pot_pressure'):
                raise ValueError('unknown eval opponent')
            collector=MultiAgentCollector(actor,cfg.rollout,engine_factory=engine_factory)
            outcome=collector.collect(seed=seed+int(n)*1000000+j*100000,
                          policy_version='heldout',num_hands=hands,force_n=int(n),
                          force_opponent=name,record_training=False)
            scores=[r[0] for r in outcome.hand_rewards]
            mean=100*sum(scores)/len(scores)
            if len(scores)>1:
                variance=sum((x-sum(scores)/len(scores))**2 for x in scores)/(len(scores)-1)
                se=100*math.sqrt(variance/len(scores))
            else:se=None
            results[f'{n}p_{name}']={'hands':len(scores),'mean_bb100':round(mean,5),
                 'ci95_normal':None if se is None else [round(mean-1.96*se,5),round(mean+1.96*se,5)]}
    return {'phase':'expand-03','checkpoint':str(checkpoint),'seed':seed,
            'hands_per_cell':hands,'results':results,
            'caution':'Hand-level normal CI is exploratory, not exploitability or multi-seed proof.'}


@contextmanager
def _exclusive_lock(root: Path):
    """OS-level advisory lock for one training run; .expand03.lock may persist."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.expand03.lock').open('a+b') as handle:
        handle.seek(0)
        handle.write(b'\0')
        handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Expand3 run directory is already locked') from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Expand3 run directory is already locked') from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def train(root: str | Path, *, config: TrainConfig | None = None, resume: bool = False,
          additional_updates: int = 3, device: str = 'cpu',
          engine_factory: Callable | None = None, emit: Callable[[dict],None] | None = None) -> dict:
    """Train one root with serialized writes and immutable resume settings."""
    with _exclusive_lock(Path(root)):
        return _train_impl(root, config=config, resume=resume,
                           additional_updates=additional_updates, device=device,
                           engine_factory=engine_factory, emit=emit)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/expand/rl/trainer.py" -->

## src/holdem_ai/experiments/__init__.py

Original blob `c45dbcdd525a3fadb2abb368e5264f09bb51e773` (39 bytes).

<a name="py-165"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/experiments/__init__.py" sha="c45dbcdd525a3fadb2abb368e5264f09bb51e773" -->

````````python
"""Consolidated functional package."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/experiments/__init__.py" -->

## src/holdem_ai/experiments/batch_scaling.py

Original blob `b96c28cb5aa54c5ffece2b871b509f489da70ad8` (17194 bytes).

<a name="py-166"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/experiments/batch_scaling.py" sha="b96c28cb5aa54c5ffece2b871b509f489da70ad8" -->

````````python
"""Phase 13 training-batch experiments, preserving Phase 9/10 checkpoints.

New experiments may warm-start from Phase 9/10/11/13 weights + Adam state,
then explicitly override rollout/optimization batch size.  Resume of a Phase13
experiment uses the checkpoint's SAVED PPO configuration without silent edits.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.training.ppo import PPOConfig, freeze_old_policy_data, optimize_ppo
from holdem_ai.training.ppo_session import create_session, load_checkpoint, save_checkpoint
from holdem_ai.training.qboost import QBoostConfig, freeze_qboost_data, optimize_qboost
from holdem_ai.training.qboost_session import create_qboost_session, load_qboost_checkpoint, save_qboost_checkpoint, from_phase9_checkpoint
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.evaluation.seed_blocked import EvaluationConfig, evaluate_seed_blocks
from holdem_ai.evaluation.action_diagnostics import action_diagnostics
from holdem_ai.evaluation.reward_diagnostics import summarize_rollout_rewards


@dataclass(frozen=True)
class BatchExperimentConfig:
    algorithm: str = 'qboost'
    device: str = 'cpu'
    duration_seconds: float = 15 * 60
    max_updates: int | None = 200
    hands_per_update: int | None = None
    minibatch_size: int | None = None
    epochs: int | None = None
    rollout_seed: int | None = None
    env_batch_size: int = 32
    eval_every: int = 25
    eval_groups: int = 40
    eval_mode: str = 'sampled'
    skip_eval: bool = False
    small_network: bool = False
    beta: float = 1.0
    checkpoint: str = 'checkpoints/phase13_qboost.pt'
    log_dir: str = 'runs/phase13_qboost'
    from_checkpoint: str | None = None
    resume: str | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in ('ppo', 'qboost') or self.device not in ('cpu', 'cuda'):
            raise ValueError('invalid algorithm/device')
        if isinstance(self.duration_seconds, bool) or not isinstance(self.duration_seconds, (int, float)) or not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError('duration_seconds must be finite and positive')
        for name in ('hands_per_update', 'minibatch_size', 'epochs', 'rollout_seed'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < (0 if name == 'rollout_seed' else 1)):
                raise ValueError(f'{name} must be a nonnegative/positive int or None')
        for name in ('env_batch_size', 'eval_every'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive integer')
        if type(self.eval_groups) is not int or self.eval_groups < 2:
            raise ValueError('eval_groups must be >=2')
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates < 1):
            raise ValueError('max_updates must be positive int or None')
        if self.eval_mode not in ('sampled', 'greedy'):
            raise ValueError('eval_mode must be sampled/greedy')
        if not isinstance(self.beta, (int, float)) or isinstance(self.beta, bool) or not 0 <= self.beta <= 1:
            raise ValueError('beta must be [0,1]')
        if self.resume and self.from_checkpoint:
            raise ValueError('--resume and --from-checkpoint are mutually exclusive')
        if self.resume and any(getattr(self, field) is not None for field in
                               ('hands_per_update', 'minibatch_size', 'epochs', 'rollout_seed')):
            raise ValueError('resume uses saved batch parameters; start a NEW stage using --from-checkpoint to reconfigure')
        if self.from_checkpoint and Path(self.from_checkpoint).resolve() == Path(self.checkpoint).resolve():
            raise ValueError('warm-start must not overwrite its source checkpoint')
        if not self.checkpoint or not self.log_dir:
            raise ValueError('checkpoint and log_dir required')


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _event(path: Path, record: dict, emit: Callable[[dict], None] | None) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    if emit:
        emit(record)


def _base(session, algorithm):
    return session if algorithm == 'ppo' else session.base


def _load_from_path(path: str, algorithm: str, device: str, beta: float = 1.0):
    metadata = torch.load(path, weights_only=True, map_location='cpu')
    phase = metadata.get('phase')
    if algorithm == 'ppo':
        if phase != 9:
            raise ValueError(f'ppo needs phase=9 checkpoint, got {phase}')
        return load_checkpoint(path, device=device)
    if phase == 9:
        return from_phase9_checkpoint(path, device=device, qconfig=QBoostConfig(beta=beta))
    if phase == 10:
        return load_qboost_checkpoint(path, device=device)
    raise ValueError(f'qboost needs phase=9/10 checkpoint, got {phase}')


def _initialize(config: BatchExperimentConfig):
    if config.resume:
        return _load_from_path(config.resume, config.algorithm, config.device, beta=config.beta)
    if config.from_checkpoint:
        session = _load_from_path(config.from_checkpoint, config.algorithm, config.device, beta=config.beta)
    else:
        net = (NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                             ff_dim=64, hidden_dim=48, max_tokens=256)
               if config.small_network else NetworkConfig())
        default_cfg = PPOConfig(seed=42)
        session = (create_session(device=config.device, seed=42,
                                  network_config=net, ppo_config=default_cfg)
                   if config.algorithm == 'ppo' else
                   create_qboost_session(device=config.device, seed=42,
                                         network_config=net, ppo_config=default_cfg,
                                         qconfig=QBoostConfig(beta=config.beta)))
    base = _base(session, config.algorithm)
    if config.algorithm == 'qboost' and not math.isclose(session.qconfig.beta, config.beta):
        raise ValueError('qboost beta must match checkpoint; do not silently alter objective')
    old = base.ppo_config
    # New warm-start experiment gets a distinct seed namespace from its source.
    seed = (config.rollout_seed if config.rollout_seed is not None else
            old.seed + 1_000_003 if config.from_checkpoint else old.seed)
    new_cfg = replace(old,
                      hands_per_update=config.hands_per_update or 64,
                      minibatch_size=config.minibatch_size or 64,
                      epochs=config.epochs or 2,
                      seed=seed)
    base.ppo_config = new_cfg
    return session


def _save(session, algorithm, path):
    if algorithm == 'ppo':
        save_checkpoint(session, path)
    else:
        save_qboost_checkpoint(session, path)


def _make_anchor(base, path: Path, resume: bool):
    model = TransformerActor(base.network_config).to(base.device).eval().requires_grad_(False)
    if resume:
        source = torch.load(path, weights_only=True, map_location='cpu')
        if source['network_config'] != asdict(base.network_config):
            raise ValueError('anchor architecture mismatch')
        model.load_state_dict(source['actor_state'])
    else:
        model.load_state_dict(base.actor.state_dict())
        tmp = path.with_name(path.name + '.tmp')
        torch.save({'network_config': asdict(base.network_config),
                    'actor_state': {k: t.detach().cpu().clone()
                                    for k, t in model.state_dict().items()}}, tmp)
        os.replace(tmp, path)
    return model


def _one_update(session, config: BatchExperimentConfig) -> dict:
    base = _base(session, config.algorithm)
    iteration = base.completed_updates
    start = time.perf_counter()
    rows = collect_batched_self_play(base.actor, base.ppo_config,
                                     iteration=iteration, device=base.device,
                                     env_batch_size=config.env_batch_size)
    diagnostics = action_diagnostics(rows, base.actor, device=base.device,
                                     chunk_size=base.ppo_config.inference_batch_size)
    reward_diagnostics = summarize_rollout_rewards(rows)
    if config.algorithm == 'ppo':
        frozen = freeze_old_policy_data(base.actor, base.critic, rows,
                                        device=base.device,
                                        chunk_size=base.ppo_config.inference_batch_size)
        stats = optimize_ppo(base.actor, base.critic,
                             base.actor_optimizer, base.critic_optimizer,
                             frozen, base.ppo_config, device=base.device,
                             shuffle_seed=base.ppo_config.seed + 100003 * iteration)
    else:
        frozen = freeze_qboost_data(base.actor, base.critic, rows,
                                    device=base.device,
                                    chunk_size=base.ppo_config.inference_batch_size)
        stats = optimize_qboost(base.actor, base.critic,
                                base.actor_optimizer, base.critic_optimizer,
                                frozen, base.ppo_config, session.qconfig,
                                device=base.device,
                                shuffle_seed=base.ppo_config.seed + 100003 * iteration)
    if base.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    base.completed_updates += 1
    return {'kind': 'update', 'algorithm': config.algorithm,
            'update': base.completed_updates, 'hands': rows.hands,
            'elapsed_seconds': round(elapsed, 5),
            'hands_per_second_training': round(rows.hands / max(elapsed, 1e-9), 3),
            'env_batch_size': config.env_batch_size,
            'minibatch_size': base.ppo_config.minibatch_size,
            'ppo_epochs': base.ppo_config.epochs,
            'action_diagnostics': diagnostics,
            'reward_diagnostics': reward_diagnostics, **stats}


def run_batch_experiment(config: BatchExperimentConfig, *,
                         emit: Callable[[dict], None] | None = None) -> dict:
    """Atomic checkpoint each update.  Resume uses same log dir and checkpoint path.

    Duration is an invocation budget (not a global wall clock across restarts).
    The only change to PPO/QBoost math is the number of sampled hands/optimizer
    minibatches; the loss implementations are imported unchanged from Phase9/10.
    """
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    root = Path(config.log_dir)
    checkpoint = Path(config.checkpoint)
    meta_path, anchor_path = root / 'run.json', root / 'anchor.pt'
    events_path, summary_path = root / 'metrics.jsonl', root / 'summary.json'
    resuming = config.resume is not None
    if resuming:
        if not all(p.is_file() for p in (meta_path, events_path, anchor_path, checkpoint)):
            raise FileNotFoundError('resume requires run.json/metrics.jsonl/anchor.pt/checkpoint')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta.get('phase') != 13 or meta.get('algorithm') != config.algorithm
                or Path(meta['checkpoint']).resolve() != checkpoint.resolve()
                or checkpoint.resolve() != Path(config.resume).resolve()):
            raise ValueError('resume metadata/checkpoint path mismatch')
    else:
        if checkpoint.exists() or (root.exists() and any(root.iterdir())):
            raise FileExistsError('fresh experiment cannot overwrite existing checkpoint/logs')
        root.mkdir(parents=True, exist_ok=True)
    session = _initialize(config)
    base = _base(session, config.algorithm)
    if resuming:
        if (asdict(base.ppo_config) != meta['ppo_config'] or
                asdict(base.network_config) != meta['network_config']):
            raise ValueError('checkpoint and run.json configuration disagree')
    else:
        meta = {'phase': 13, 'algorithm': config.algorithm,
                'checkpoint': str(checkpoint.resolve()),
                'initial_updates': base.completed_updates,
                'source_checkpoint': str(Path(config.from_checkpoint).resolve()) if config.from_checkpoint else None,
                'checkpoint_format_phase': 9 if config.algorithm == 'ppo' else 10,
                'network_config': asdict(base.network_config),
                'ppo_config': asdict(base.ppo_config),
                'beta': session.qconfig.beta if config.algorithm == 'qboost' else 0.0,
                'config': asdict(config)}
        _atomic_json(meta_path, meta)
    anchor = _make_anchor(base, anchor_path, resume=resuming)
    if not resuming:
        _save(session, config.algorithm, checkpoint)
    if config.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    end_time = started + config.duration_seconds
    completed = 0
    prev_seconds = 0.0
    evaluations = 0
    reason = 'duration_limit'
    _event(events_path, {'kind': 'resume' if resuming else 'start',
                         'completed_updates': base.completed_updates,
                         'duration_seconds': config.duration_seconds,
                         'env_batch_size': config.env_batch_size,
                         'hands_per_update': base.ppo_config.hands_per_update}, emit)

    def eval_now():
        nonlocal evaluations
        cfg = EvaluationConfig(groups=config.eval_groups, seed=12_000_000,
                               mode=config.eval_mode, device=config.device,
                               big_blind=base.ppo_config.big_blind)
        for opponent, name in ((None, 'random'), (anchor, 'frozen_anchor')):
            summary = evaluate_seed_blocks(base.actor, opponent=opponent,
                                           config=cfg).summary()
            _event(events_path, {'kind': 'evaluation', 'update': base.completed_updates,
                                 'opponent': name, **summary}, emit)
            evaluations += 1

    try:
        if not config.skip_eval and not resuming and time.perf_counter() < end_time:
            eval_now()
        while True:
            if config.max_updates is not None and completed >= config.max_updates:
                reason = 'update_limit'
                break
            if time.perf_counter() + max(prev_seconds * 1.1, 0.02) >= end_time:
                break
            report = _one_update(session, config)
            # Always save the optimizer state before declaring an update committed.
            _save(session, config.algorithm, checkpoint)
            _event(events_path, report, emit)
            completed += 1
            prev_seconds = report['elapsed_seconds']
            if (not config.skip_eval and base.completed_updates % config.eval_every == 0
                    and time.perf_counter() < end_time):
                eval_now()
        if (not config.skip_eval and completed > 0 and
                base.completed_updates % config.eval_every != 0 and
                time.perf_counter() < end_time):
            eval_now()
    except KeyboardInterrupt:
        reason = 'keyboard_interrupt'
        _event(events_path, {'kind': 'interrupt', 'update_on_disk':
                             int(torch.load(checkpoint, weights_only=True,
                                            map_location='cpu')['completed_updates'])}, emit)
    elapsed = time.perf_counter() - started
    summary = {'phase': 13, 'algorithm': config.algorithm, 'device': config.device,
               'stop_reason': reason, 'updates_this_run': completed,
               'completed_updates': int(torch.load(checkpoint, weights_only=True,
                                                   map_location='cpu')['completed_updates']),
               'elapsed_seconds': round(elapsed, 5),
               'evaluations': evaluations,
               'hands_per_update': base.ppo_config.hands_per_update,
               'env_batch_size': config.env_batch_size,
               'checkpoint': str(checkpoint), 'log_dir': str(root),
               'cuda_peak_allocated_mb': (round(torch.cuda.max_memory_allocated() / 2**20, 3)
                                          if config.device == 'cuda' else None)}
    _atomic_json(summary_path, summary)
    _event(events_path, {'kind': 'end', **summary}, emit)
    return summary

````````

<!-- END_PY_SOURCE path="src/holdem_ai/experiments/batch_scaling.py" -->

## src/holdem_ai/experiments/controlled_study.py

Original blob `0d35bf8ef140b3f268c0a2716b30580d3b6773c7` (21714 bytes).

<a name="py-167"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/experiments/controlled_study.py" sha="0d35bf8ef140b3f268c0a2716b30580d3b6773c7" -->

````````python
"""Matched-start PPO/QBoost study with independent held-out test evaluation.

Both arms start from a trusted Phase9 checkpoint (identical actor/critic AND
Adam states). Within each training seed, on-policy rollouts have identical RNG
seed schedules but can diverge after the networks change. A validation-only
best checkpoint is saved; test seeds are untouched until all arms complete.

Phase13 supplies the unchanged PPO/QBoost loss and batched self-play step.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from holdem_ai.utils.progress import TrainingProgress
from holdem_ai.evaluation.seed_blocked import EvaluationConfig, compare_results, evaluate_seed_blocks
from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.evaluation.seed_blocked import model_fingerprint
from holdem_ai.experiments.batch_scaling import BatchExperimentConfig, _base, _load_from_path, _one_update, _save
from holdem_ai.evaluation.seed_statistics import seed_level_summary


@dataclass(frozen=True)
class ComparisonConfig:
    source_checkpoint: str
    root: str = 'runs/phase14_study'
    seeds: tuple[int, ...] = (41, 42, 43)
    updates: int = 100
    hands_per_update: int = 64
    env_batch_size: int = 32
    minibatch_size: int = 64
    epochs: int = 2
    beta: float = 1.0
    eval_every: int = 25
    validation_groups: int = 40
    test_groups: int = 200
    eval_mode: str = 'sampled'
    device: str = 'cpu'
    arm_duration_minutes: float = 30.0
    validation_seed: int = 12_000_000
    test_seed: int = 22_000_000
    entropy_warning_threshold: float = 0.6
    resume: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if not self.source_checkpoint or not self.root:
            raise ValueError('source_checkpoint and root are required')
        if self.device not in ('cpu', 'cuda') or self.eval_mode not in ('sampled', 'greedy'):
            raise ValueError('invalid device/evaluation mode')
        if not self.seeds or any(type(s) is not int or not 0 <= s < 100_000 for s in self.seeds):
            raise ValueError('seeds must be distinct integers in [0,100000)')
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError('training seeds must be unique')
        for key in ('updates', 'hands_per_update', 'env_batch_size', 'minibatch_size',
                    'epochs', 'eval_every', 'validation_groups', 'test_groups'):
            val = getattr(self, key)
            if type(val) is not int or val < (2 if key.endswith('groups') else 1):
                raise ValueError(f'{key} must be positive integer (groups >= 2)')
        if self.env_batch_size > self.hands_per_update:
            raise ValueError('env_batch_size must be <= hands_per_update')
        for name in ('validation_seed', 'test_seed'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative integer')
        if self.updates * self.hands_per_update >= 1_000_000:
            raise ValueError('rollout seed namespace supports fewer than 1,000,000 hands per arm')
        for start, groups in ((self.validation_seed, self.validation_groups),
                              (self.test_seed, self.test_groups)):
            # Rollout cards occupy the 100-million+ namespace (disjoint by seed).
            for train_seed in self.seeds:
                lo = 100_000_000 + train_seed * 1_000_000
                hi = lo + 1_000_000
                if start < hi and start + groups > lo:
                    raise ValueError('evaluation seed range overlaps a training rollout seed range')
        v = set(range(self.validation_seed, self.validation_seed+self.validation_groups))
        t = set(range(self.test_seed, self.test_seed+self.test_groups))
        if v & t:
            raise ValueError('validation and test deck seeds must not overlap')
        if not 0 <= self.beta <= 1 or not math.isfinite(self.beta):
            raise ValueError('beta must be finite in [0,1]')
        if not (math.isfinite(self.arm_duration_minutes) and self.arm_duration_minutes > 0):
            raise ValueError('arm_duration_minutes must be finite positive')
        if not (math.isfinite(self.entropy_warning_threshold) and self.entropy_warning_threshold >= 0):
            raise ValueError('entropy warning threshold must be >= 0')


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _event(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_and_metadata(cfg: ComparisonConfig) -> tuple[Path, dict, dict]:
    source = Path(cfg.source_checkpoint).resolve(strict=True)
    payload = torch.load(source, map_location='cpu', weights_only=True)
    if payload.get('phase') != 9 or not all(k in payload for k in
            ('actor_optimizer', 'critic_optimizer', 'actor_state', 'critic_state')):
        raise ValueError('matched PPO/QBoost study requires a trusted Phase9 full checkpoint')
    digest = _sha256(source)
    frozen_cfg = asdict(cfg)
    # Only `resume` and console presentation may differ across invocations.
    frozen_cfg.pop('resume')
    frozen_cfg.pop('progress')
    frozen_cfg.pop('arm_duration_minutes')  # new wall-clock budget per resumed invocation
    frozen_cfg['source_checkpoint'] = str(source)
    frozen_cfg['seeds'] = list(cfg.seeds)
    frozen_cfg['source_sha256'] = digest
    return source, payload, frozen_cfg


def _case_paths(root: Path, seed: int, algorithm: str) -> dict[str, Path]:
    case = root / f'seed_{seed}' / algorithm
    return {'root': case, 'latest': case / 'latest.pt', 'best': case / 'best.pt',
            'state': case / 'state.json', 'events': case / 'metrics.jsonl',
            'meta': case / 'run.json'}


def _rollout_seed(seed: int) -> int:
    return 100_000_000 + seed * 1_000_000


def _make_batch_config(cfg: ComparisonConfig, path: Path, seed: int, algorithm: str) -> BatchExperimentConfig:
    return BatchExperimentConfig(
        algorithm=algorithm, device=cfg.device, duration_seconds=cfg.arm_duration_minutes*60,
        max_updates=cfg.updates, hands_per_update=cfg.hands_per_update,
        minibatch_size=cfg.minibatch_size, epochs=cfg.epochs,
        rollout_seed=_rollout_seed(seed), env_batch_size=cfg.env_batch_size,
        eval_every=cfg.eval_every, eval_groups=cfg.validation_groups,
        eval_mode=cfg.eval_mode, beta=cfg.beta,
        checkpoint=str(path), log_dir=str(path.parent), skip_eval=True,
    )


def _eval(actor, opponent, cfg: ComparisonConfig, *, partition: str):
    seed, groups = ((cfg.validation_seed, cfg.validation_groups)
                    if partition == 'validation' else (cfg.test_seed, cfg.test_groups))
    settings = EvaluationConfig(groups=groups, seed=seed, mode=cfg.eval_mode,
                                device=cfg.device)
    return evaluate_seed_blocks(actor, opponent=opponent, config=settings)


def _new_arm(cfg: ComparisonConfig, source: Path, seed: int, algorithm: str,
             paths: dict[str, Path], initial_fingerprint: str) -> None:
    if paths['root'].exists() and any(paths['root'].iterdir()):
        raise FileExistsError(f'new study cannot overwrite {paths["root"]}')
    paths['root'].mkdir(parents=True, exist_ok=True)
    session = _load_from_path(str(source), algorithm, cfg.device, beta=cfg.beta)
    base = _base(session, algorithm)
    base.ppo_config = replace(base.ppo_config, seed=_rollout_seed(seed),
                              hands_per_update=cfg.hands_per_update,
                              minibatch_size=cfg.minibatch_size, epochs=cfg.epochs)
    if model_fingerprint(base.actor) != initial_fingerprint:
        raise AssertionError('matched-start initial actor fingerprint mismatch')
    _save(session, algorithm, paths['latest'])
    _write_json(paths['meta'], {
        'phase': 14, 'algorithm': algorithm, 'seed': seed,
        'initial_update': base.completed_updates,
        'source_fingerprint': initial_fingerprint,
        'ppo_config': asdict(base.ppo_config),
        'network_config': asdict(base.network_config),
    })
    _write_json(paths['state'], {
        'best_score_bb100': None, 'best_update': None,
        'last_validation_update': None, 'entropy_warnings': 0,
    })
    _event(paths['events'], {'kind': 'start', 'algorithm': algorithm,
                             'completed_updates': base.completed_updates,
                             'duration_seconds': cfg.arm_duration_minutes*60,
                             'env_batch_size': cfg.env_batch_size})


def _load_arm(paths: dict[str, Path], cfg: ComparisonConfig, seed: int,
              algorithm: str, fingerprint: str):
    meta = json.loads(paths['meta'].read_text(encoding='utf-8'))
    state = json.loads(paths['state'].read_text(encoding='utf-8'))
    session = _load_from_path(str(paths['latest']), algorithm, cfg.device, beta=cfg.beta)
    base = _base(session, algorithm)
    if (meta['source_fingerprint'] != fingerprint or meta['seed'] != seed or
            meta['algorithm'] != algorithm or meta['ppo_config'] != asdict(base.ppo_config)
            or meta['network_config'] != asdict(base.network_config)):
        raise ValueError('arm checkpoint and study metadata diverged; refusing resume')
    if base.completed_updates < meta['initial_update'] or base.completed_updates > meta['initial_update'] + cfg.updates:
        raise ValueError('checkpoint counter outside pre-registered update budget')
    return session, meta, state


def _validation(cfg: ComparisonConfig, base, anchor, paths: dict[str, Path],
                state: dict, label: str) -> dict:
    result = _eval(base.actor, anchor, cfg, partition='validation').summary()
    score = float(result['mean_bb100'])
    update = base.completed_updates
    event = {'kind': 'validation', 'update': update, 'algorithm': label,
             'opponent': 'frozen_initial', **result}
    _event(paths['events'], event)
    if state['best_score_bb100'] is None or score > state['best_score_bb100']:
        # The saved file contains precisely the policy just evaluated.
        tmp = paths['best'].with_name('best.pt.tmp')
        shutil.copyfile(paths['latest'], tmp)
        os.replace(tmp, paths['best'])
        state['best_score_bb100'] = score
        state['best_update'] = update
    state['last_validation_update'] = update
    _write_json(paths['state'], state)
    return event


def _run_arm(cfg: ComparisonConfig, source: Path, seed: int, algorithm: str,
             root: Path, anchor, fingerprint: str,
             emit: Callable[[dict], None] | None = None) -> dict:
    paths = _case_paths(root, seed, algorithm)
    if not paths['meta'].is_file():
        if cfg.resume and paths['root'].exists() and any(paths['root'].iterdir()):
            raise FileNotFoundError(f'incomplete arm metadata: {paths["meta"]}')
        _new_arm(cfg, source, seed, algorithm, paths, fingerprint)
    session, meta, state = _load_arm(paths, cfg, seed, algorithm, fingerprint)
    base = _base(session, algorithm)
    first = meta['initial_update']
    target = first + cfg.updates
    # The loss and collector use Phase13 without changing the policy-gradient math.
    step_cfg = _make_batch_config(cfg, paths['latest'], seed, algorithm)
    deadline = time.perf_counter() + cfg.arm_duration_minutes*60
    updates_this_call = 0
    with _Phase14Progress(duration_seconds=cfg.arm_duration_minutes*60,
                          max_updates=max(1, target-base.completed_updates),
                          threaded=cfg.progress, prefer_tqdm=cfg.progress) if cfg.progress else _SilentProgress() as bar:
        bar.on_event({'kind': 'start', 'completed_updates': base.completed_updates})
        if state['last_validation_update'] is None:
            row = _validation(cfg, base, anchor, paths, state, algorithm)
            if emit:
                emit(row)
        while base.completed_updates < target:
            if time.perf_counter() + 0.05 >= deadline:
                break
            try:
                item = _one_update(session, step_cfg)
            except KeyboardInterrupt:
                _event(paths['events'], {'kind': 'interrupt', 'last_saved_update': base.completed_updates})
                break
            _save(session, algorithm, paths['latest'])
            _event(paths['events'], item)
            bar.on_event(item)
            updates_this_call += 1
            if item.get('entropy', 999) < cfg.entropy_warning_threshold:
                state['entropy_warnings'] += 1
                _event(paths['events'], {'kind':'entropy_warning', 'update':base.completed_updates,
                                         'entropy':item['entropy']})
                _write_json(paths['state'], state)
            if base.completed_updates % cfg.eval_every == 0 or base.completed_updates == target:
                row = _validation(cfg, base, anchor, paths, state, algorithm)
                if emit:
                    emit(row)
        bar.on_event({'kind': 'end', 'stop_reason': 'update_limit' if base.completed_updates == target
                      else 'duration_limit'})
    if not paths['best'].exists():
        raise AssertionError('no best validation checkpoint generated')
    return {'algorithm': algorithm, 'seed': seed,
            'training_diagnostics': _summarize_training(paths['events']),
            'updates_this_call': updates_this_call,
            'completed_updates': base.completed_updates,
            'target_updates': target, 'finished': base.completed_updates == target,
            'best_validation_update': state['best_update'],
            'best_validation_bb100': state['best_score_bb100'],
            'entropy_warnings': state['entropy_warnings'],
            'best_checkpoint': str(paths['best']), 'latest_checkpoint': str(paths['latest'])}


def _summarize_training(path: Path) -> dict:
    updates = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
               if line.strip()]
    updates = [x for x in updates if x.get('kind') == 'update']
    count = len(updates)
    def average(name):
        values = [float(u[name]) for u in updates if name in u]
        return round(math.fsum(values)/len(values), 6) if values else None
    hands = sum(int(u['hands']) for u in updates)
    seconds = math.fsum(float(u['elapsed_seconds']) for u in updates)
    metrics = {'updates_logged': count, 'training_hands_logged': hands,
               'training_seconds_logged': round(seconds, 4),
               'full_training_hands_per_second': round(hands/seconds, 3) if seconds else None,
               'average_entropy': average('entropy'),
               'average_critic_loss': average('critic_loss'),
               'average_approx_kl': average('approx_kl'),
               'average_logit_variance_ratio': average('logit_variance_ratio')}
    for a in ('fold','check','call','raise_double_pot','all_in'):
        opts = [u['action_diagnostics'] for u in updates if 'action_diagnostics' in u]
        legal = sum(o['legal_opportunities'].get(a,0) for o in opts)
        used = sum(o['actions'].get(a,0) for o in opts)
        metrics[a+'_when_legal'] = round(used/legal,6) if legal else None
    return metrics


class _Phase14Progress(TrainingProgress):
    def _open_display(self):
        super()._open_display()
        if self._bar is not None:
            self._bar.set_description('Phase 14')


class _SilentProgress:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def on_event(self, event): pass


def _final_test(cfg: ComparisonConfig, root: Path, seed: int, anchor) -> dict:
    # Identical test deck seeds, initial anchor and evaluation mode across both arms.
    output = root / f'seed_{seed}' / 'test_results.json'
    p_paths, q_paths = (_case_paths(root, seed, x) for x in ('ppo','qboost'))
    if output.exists():
        cached = json.loads(output.read_text(encoding='utf-8'))
        if (cached.get('best_sha256_ppo') != _sha256(p_paths['best'])
                or cached.get('best_sha256_qboost') != _sha256(q_paths['best'])):
            raise ValueError('best checkpoint changed after test evaluation')
        return cached
    p_model = load_actor(p_paths['best'], device=cfg.device)
    q_model = load_actor(q_paths['best'], device=cfg.device)
    p_anchor = _eval(p_model, anchor, cfg, partition='test')
    q_anchor = _eval(q_model, anchor, cfg, partition='test')
    paired = compare_results(q_anchor, p_anchor)
    p_random = _eval(p_model, None, cfg, partition='test')
    q_random = _eval(q_model, None, cfg, partition='test')
    paired_random = compare_results(q_random, p_random)
    head_to_head = _eval(q_model, p_model, cfg, partition='test')
    result = {'seed': seed, 'phase': 14, 'partition': 'test',
              'best_sha256_ppo': _sha256(p_paths['best']),
              'best_sha256_qboost': _sha256(q_paths['best']),
              'best_checkpoint_ppo': str(p_paths['best']),
              'best_checkpoint_qboost': str(q_paths['best']),
              'ppo_vs_initial': p_anchor.summary(), 'qboost_vs_initial': q_anchor.summary(),
              'qboost_minus_ppo_vs_initial': paired,
              'ppo_vs_random': p_random.summary(), 'qboost_vs_random': q_random.summary(),
              'qboost_minus_ppo_vs_random': paired_random,
              'qboost_vs_ppo_headtohead': head_to_head.summary()}
    _write_json(output, result)
    return result


def run_study(cfg: ComparisonConfig, *, emit: Callable[[dict], None] | None = None) -> dict:
    """Run all arms; save partial results if a per-arm deadline is reached.

    `--resume` must be used to continue the SAME preregistered study. Test
    results are produced only after every PPO and QBoost arm has completed.
    """
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    source, _payload, prereg = _source_and_metadata(cfg)
    root = Path(cfg.root)
    study_file = root / 'study.json'
    if cfg.resume:
        if not study_file.is_file():
            raise FileNotFoundError('no Phase14 study.json to resume')
        original = json.loads(study_file.read_text(encoding='utf-8'))
        if original != prereg:
            raise ValueError('study configuration/source bytes differ; refusing resume')
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('fresh Phase14 study cannot overwrite prior experiment')
        if root.resolve() == source.parent or source.is_relative_to(root.resolve()):
            raise ValueError('study directory must not contain the source checkpoint')
        _write_json(study_file, prereg)
    anchor = load_actor(source, device=cfg.device)
    fingerprint = model_fingerprint(anchor)
    outputs = []
    for seed in cfg.seeds:
        for algorithm in ('ppo','qboost'):
            item = _run_arm(cfg, source, seed, algorithm, root, anchor, fingerprint, emit=emit)
            outputs.append(item)
            if emit:
                emit({'kind':'arm_end', **item})
            if not item['finished']:
                partial = {'phase':14, 'status':'partial', 'root':str(root),
                           'arms': outputs, 'note':'Use --resume; test partition remains untouched.'}
                _write_json(root/'summary.json', partial)
                return partial
    test_results = [_final_test(cfg, root, seed, anchor) for seed in cfg.seeds]
    diffs = [r['qboost_minus_ppo_vs_initial']['delta_mean_bb100'] for r in test_results]
    diffs_random = [r['qboost_minus_ppo_vs_random']['delta_mean_bb100'] for r in test_results]
    h2h = [r['qboost_vs_ppo_headtohead']['mean_bb100'] for r in test_results]
    result = {
        'phase': 14, 'status': 'complete', 'root': str(root),
        'source_sha256': prereg['source_sha256'], 'training_seeds': list(cfg.seeds),
        'total_training_hands': 2*len(cfg.seeds)*cfg.updates*cfg.hands_per_update,
        'arms': outputs,
        'primary_qboost_minus_ppo_vs_initial': seed_level_summary(diffs),
        'secondary_qboost_minus_ppo_vs_random': seed_level_summary(diffs_random),
        'secondary_direct_headtohead_qboost_bb100': seed_level_summary(h2h),
        'seed_results': test_results,
        'notes': [
            'Preselected checkpoints by validation vs the same frozen source Actor.',
            'Test seeds are disjoint and first accessed only after all training finishes.',
            'Training-seed t intervals are conditional on the shared held-out deck bank.',
            'Policy trajectories diverge after updates; common RNG seeds are not identical trajectories.',
            'This evaluation is not GTO exploitability or proof of VRPO-paper reproduction.',
        ],
    }
    _write_json(root/'summary.json', result)
    return result

````````

<!-- END_PY_SOURCE path="src/holdem_ai/experiments/controlled_study.py" -->

## src/holdem_ai/experiments/longrun.py

Original blob `50e3f7afacf2e97651df73e3ca1525e73fdbae3a` (26085 bytes).

<a name="py-168"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/experiments/longrun.py" sha="50e3f7afacf2e97651df73e3ca1525e73fdbae3a" -->

````````python
"""Phase 15: safe long-run HUNL training, validation and sealed final audit.

Reuses Phase13's on-policy collector / PPO and QBoost optimizers unchanged.
A fresh run MUST have a new directory and an independent rollout seed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from holdem_ai.utils.progress import TrainingProgress
from holdem_ai.experiments.ppo_safety import safety_assessment
from holdem_ai.evaluation.seed_blocked import EvaluationConfig, evaluate_seed_blocks
from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.experiments.batch_scaling import BatchExperimentConfig, _base, _load_from_path, _one_update, _save


@contextmanager
def _exclusive_lock(root: Path):
    """Non-blocking OS file lock; released automatically after process death.

    A persistent .phase15.lock is expected and is NOT a partial checkpoint.
    """
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.phase15.lock').open('a+b') as handle:
        handle.seek(0)
        handle.write(b'\0')
        handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('another Phase15 process holds this run directory') from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('another Phase15 process holds this run directory') from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


DEFAULTS = {
    'hands_per_update': 64, 'env_batch_size': 32, 'minibatch_size': 64,
    'epochs': 2, 'rollout_seed': 400_000_000, 'eval_every': 100,
    'eval_groups': 100, 'eval_mode': 'sampled', 'validation_seed': 15_000_000,
    'final_groups': 400, 'final_seed': 25_000_000,
    'entropy_alert_threshold': 0.45, 'entropy_stop_patience': 12,
    'max_approx_kl': 0.10, 'beta': 1.0,
    'entropy_stop_enabled': False,
}


@dataclass(frozen=True)
class LongRunConfig:
    root: str = 'runs/phase15_ppo'
    source_checkpoint: str | None = None
    algorithm: str | None = None
    device: str = 'cpu'
    duration_minutes: float = 300.0
    max_updates: int | None = None  # ADDITIONAL updates THIS invocation
    resume: bool = False
    progress: bool = True
    hands_per_update: int | None = None
    env_batch_size: int | None = None
    minibatch_size: int | None = None
    epochs: int | None = None
    rollout_seed: int | None = None
    eval_every: int | None = None
    eval_groups: int | None = None
    eval_mode: str | None = None
    validation_seed: int | None = None
    final_groups: int | None = None
    final_seed: int | None = None
    entropy_alert_threshold: float | None = None
    entropy_stop_patience: int | None = None
    entropy_stop_enabled: bool | None = None
    max_approx_kl: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        if not self.root or self.device not in ('cpu', 'cuda'):
            raise ValueError('nonempty root and cpu/cuda device required')
        if self.algorithm is not None and self.algorithm not in ('ppo','qboost'):
            raise ValueError('algorithm must be ppo/qboost')
        if not self.resume and (not self.source_checkpoint or not self.algorithm):
            raise ValueError('fresh run requires --source-checkpoint and --algorithm')
        if not isinstance(self.duration_minutes, (int,float)) or isinstance(self.duration_minutes, bool) or not math.isfinite(self.duration_minutes) or self.duration_minutes <= 0:
            raise ValueError('duration_minutes must be finite positive')
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates < 1):
            raise ValueError('max_updates must be positive or omitted')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _event(path: Path, row: dict, emit: Callable[[dict],None] | None = None) -> None:
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if emit:
        emit(row)


def _paths(root: Path) -> dict[str, Path]:
    return {k: root / v for k, v in {
        'manifest': 'run.json', 'latest': 'latest.pt', 'best': 'best.pt',
        'state': 'state.json', 'events': 'metrics.jsonl',
        'summary': 'summary.json', 'final': 'final_report.json',
    }.items()}


def _spec(cfg: LongRunConfig) -> dict:
    spec = {name: (DEFAULTS[name] if getattr(cfg,name) is None else getattr(cfg,name)) for name in DEFAULTS}
    for name in ('hands_per_update','env_batch_size','minibatch_size','epochs','eval_every',
                 'eval_groups','final_groups','entropy_stop_patience'):
        value = spec[name]
        if type(value) is not int or value < (2 if name in ('eval_groups','final_groups') else 1):
            raise ValueError(f'{name} must be a positive integer (groups >=2)')
    for name in ('rollout_seed','validation_seed','final_seed'):
        if type(spec[name]) is not int or spec[name] < 0:
            raise ValueError(f'{name} must be nonnegative integer')
    if spec['env_batch_size'] > spec['hands_per_update']:
        raise ValueError('env_batch_size cannot exceed hands_per_update')
    if spec['eval_mode'] not in ('sampled','greedy'):
        raise ValueError('eval_mode must be sampled/greedy')
    if type(spec['entropy_stop_enabled']) is not bool:
        raise ValueError('entropy_stop_enabled must be boolean')
    for name in ('entropy_alert_threshold','max_approx_kl','beta'):
        value = spec[name]
        if (not isinstance(value, (float,int)) or isinstance(value,bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite nonnegative')
    if spec['beta'] > 1 or spec['max_approx_kl'] <= 0:
        raise ValueError('beta <=1 and max_approx_kl >0 required')
    # Each population of evaluation decks must be held out of all training RNGs.
    intervals = [(spec['validation_seed'],spec['validation_seed']+spec['eval_groups']),
                 (spec['final_seed'],spec['final_seed']+spec['final_groups'])]
    if max(intervals[0][0],intervals[1][0]) < min(intervals[0][1],intervals[1][1]):
        raise ValueError('validation and final card seed ranges overlap')
    # Reserve a 20M-seed training block (up to 20M / hands_per_update updates).
    train_range = (spec['rollout_seed'],spec['rollout_seed']+20_000_000)
    for a,b in intervals:
        if max(a,train_range[0]) < min(b,train_range[1]):
            raise ValueError('held-out card seeds overlap reserved rollout seed namespace')
    return spec


def _read(root: Path) -> tuple[dict,dict,dict[str,Path]]:
    paths = _paths(root)
    if not all(paths[k].is_file() for k in ('manifest','state','latest','best','events')):
        raise FileNotFoundError('incomplete Phase15 run: missing manifest/state/latest/best/metrics')
    manifest=json.loads(paths['manifest'].read_text(encoding='utf-8'))
    state=json.loads(paths['state'].read_text(encoding='utf-8'))
    if state.get('best_sha256') != _sha256(paths['best']):
        raise ValueError('selected best.pt changed or best/state files are out of sync')
    return manifest,state,paths


def _match_resume(cfg: LongRunConfig, manifest: dict) -> dict:
    spec = manifest['settings']
    if cfg.source_checkpoint is not None and Path(cfg.source_checkpoint).resolve() != Path(manifest['source']).resolve():
        raise ValueError('resume source checkpoint differs')
    if cfg.algorithm is not None and cfg.algorithm != manifest['algorithm']:
        raise ValueError('resume algorithm differs')
    for name in DEFAULTS:
        value = getattr(cfg, name)
        if value is not None and value != spec.get(name, True if name == 'entropy_stop_enabled' else DEFAULTS[name]):
            raise ValueError(f'resume configuration mismatch: {name}')
    if _sha256(Path(manifest['source'])) != manifest['source_sha256']:
        raise ValueError('source checkpoint bytes changed since experiment initialization')
    return spec


def _evaluation(actor, opponent, spec: dict, device: str, *, final: bool = False, big_blind: int = 100) -> dict:
    settings = EvaluationConfig(
        groups=spec['final_groups'] if final else spec['eval_groups'],
        seed=spec['final_seed'] if final else spec['validation_seed'],
        mode=spec['eval_mode'], device=device, big_blind=big_blind,
    )
    return evaluate_seed_blocks(actor, opponent=opponent, config=settings).summary()


def _validation(base, anchor, spec: dict, device: str, baseline: dict) -> dict:
    current_anchor = _evaluation(base.actor, anchor, spec, device, big_blind=base.ppo_config.big_blind)
    current_random = _evaluation(base.actor, None, spec, device, big_blind=base.ppo_config.big_blind)
    # Equal-weighted CHANGE from initial policy against two fixed opponents.
    anchor_gain = current_anchor['mean_bb100'] - baseline['anchor']['mean_bb100']
    random_gain = current_random['mean_bb100'] - baseline['random']['mean_bb100']
    score = (anchor_gain + random_gain) / 2
    return {'kind': 'validation', 'update': base.completed_updates,
            'score_bb100': round(score,4), 'gain_vs_anchor_bb100': round(anchor_gain,4),
            'gain_vs_random_bb100': round(random_gain,4),
            'vs_anchor': current_anchor, 'vs_random': current_random}


class _Progress(TrainingProgress):
    def _open_display(self):
        super()._open_display()
        if self._bar is not None:
            self._bar.set_description('Phase 15')


def _emit_or_skip(event: dict, emit: Callable[[dict],None] | None, bar) -> None:
    bar.on_event(event)
    if emit:
        emit(event)


def _summary(root: Path, manifest: dict, state: dict, *, stop_reason: str,
             updates_this_run: int, elapsed: float, peak: float | None) -> dict:
    return {
        'phase': 15, 'algorithm': manifest['algorithm'], 'stop_reason': stop_reason,
        'updates_this_run': updates_this_run, 'completed_updates': state['completed_updates'],
        'total_updates_since_start': state['completed_updates']-manifest['initial_update'],
        'total_training_hands': (state['completed_updates']-manifest['initial_update'])*manifest['settings']['hands_per_update'],
        'elapsed_seconds_this_call': round(elapsed,3),
        'best_validation_update': state['best_update'],
        'best_validation_score_bb100': state['best_score_bb100'],
        'entropy_warnings': state['entropy_warnings'],
        'kl_warnings': state['kl_warnings'], 'latest_checkpoint': str(root/'latest.pt'),
        'best_checkpoint': str(root/'best.pt'), 'cuda_peak_allocated_mb': peak,
        'sealed': (root/'final_report.json').is_file(),
    }


def _run_longrun_impl(cfg: LongRunConfig, *, emit: Callable[[dict],None] | None = None) -> dict:
    """Train against a per-invocation wall-clock budget, with safe resume.

    A completed update is persisted atomically BEFORE logging. Unexpected power
    failure can lose the last metrics row but never silently reuse old weights.
    """
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    root = Path(cfg.root)
    paths = _paths(root)
    if cfg.resume:
        manifest,state,paths = _read(root)
        spec = _match_resume(cfg, manifest)
        # Historical manifests precede the warning-only entropy flag.
        # Keep old behavior rather than silently changing resumed experiments.
        if 'entropy_stop_enabled' not in spec:
            spec = dict(spec, entropy_stop_enabled=True)
        if paths['final'].exists():
            raise RuntimeError('run is sealed by final evaluation; cannot resume training')
        algorithm = manifest['algorithm']
        session = _load_from_path(str(paths['latest']), algorithm, cfg.device, beta=spec['beta'])
        base = _base(session,algorithm)
        if asdict(base.ppo_config) != manifest['ppo_config'] or asdict(base.network_config) != manifest['network_config']:
            raise ValueError('checkpoint config differs from manifest')
        if base.completed_updates < state['completed_updates']:
            raise ValueError('latest checkpoint is older than committed state')
        if base.completed_updates > state['completed_updates']:
            # Latest save preceded event/state write when the previous process failed.
            _event(paths['events'], {'kind':'recovered_update','update':base.completed_updates})
            state['completed_updates'] = base.completed_updates
            _atomic_json(paths['state'],state)
    else:
        if root.exists() and any(p.name != '.phase15.lock' for p in root.iterdir()):
            raise FileExistsError('fresh Phase15 root is not empty; use --resume or a new directory')
        spec = _spec(cfg)
        algorithm = cfg.algorithm
        source = Path(cfg.source_checkpoint).resolve(strict=True)
        if source == paths['latest'].resolve() or source.is_relative_to(root.resolve()):
            raise ValueError('run directory cannot contain the source checkpoint')
        session = _load_from_path(str(source), algorithm, cfg.device, beta=spec['beta'])
        base = _base(session,algorithm)
        if algorithm == 'qboost' and not math.isclose(session.qconfig.beta, spec['beta']):
            raise ValueError('QBoost beta must match source checkpoint')
        base.ppo_config = replace(base.ppo_config,
                                  hands_per_update=spec['hands_per_update'],
                                  minibatch_size=spec['minibatch_size'],
                                  epochs=spec['epochs'],seed=spec['rollout_seed'])
        # No Future-deck or opponent-hole-card observation reaches Actor.
        manifest = {'phase': 15, 'algorithm': algorithm, 'source': str(source),
                    'source_sha256': _sha256(source),
                    'initial_update': base.completed_updates,
                    'ppo_config': asdict(base.ppo_config),
                    'network_config': asdict(base.network_config),
                    'settings': spec}
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(paths['manifest'], manifest)
        _save(session,algorithm,paths['latest'])
        shutil.copyfile(paths['latest'],paths['best'])
        anchor = load_actor(source,device=cfg.device)
        base_anchor = _evaluation(base.actor,anchor,spec,cfg.device,big_blind=base.ppo_config.big_blind)
        base_random = _evaluation(base.actor,None,spec,cfg.device,big_blind=base.ppo_config.big_blind)
        baseline = {'anchor':base_anchor,'random':base_random}
        state = {'completed_updates':base.completed_updates,
                 'best_sha256':_sha256(paths['best']),
                 'best_update':base.completed_updates, 'best_score_bb100':0.0,
                 'baseline':baseline, 'last_validation_update':base.completed_updates,
                 'entropy_warnings':0, 'kl_warnings':0, 'consecutive_safety_warnings':0,
                 'validation_count':1}
        _atomic_json(paths['state'],state)
        _event(paths['events'], {'kind':'validation','update':base.completed_updates,
                                 'score_bb100':0.0,'vs_anchor':base_anchor,
                                 'vs_random':base_random})
    # The source snapshot never mutates during this run (checked every resume).
    anchor = load_actor(manifest['source'],device=cfg.device)
    if cfg.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    deadline = started + cfg.duration_minutes*60
    updates_this_run = 0
    last_update_seconds = 0.0
    stop_reason = 'duration_limit'
    step_cfg = BatchExperimentConfig(
        algorithm=algorithm,device=cfg.device,duration_seconds=cfg.duration_minutes*60,
        max_updates=cfg.max_updates,env_batch_size=spec['env_batch_size'],
        checkpoint=str(paths['latest']),log_dir=str(root),eval_every=spec['eval_every'],
        eval_groups=spec['eval_groups'],skip_eval=True,beta=spec['beta'],
    )
    bar = (_Progress(duration_seconds=cfg.duration_minutes*60,max_updates=cfg.max_updates)
           if cfg.progress else None)
    progress_ctx = bar if bar is not None else nullcontext()
    with progress_ctx:
        event = {'kind':'resume' if cfg.resume else 'start',
                 'completed_updates':base.completed_updates,
                 'duration_seconds':cfg.duration_minutes*60,
                 'hands_per_update':spec['hands_per_update'],
                 'env_batch_size':spec['env_batch_size']}
        _event(paths['events'],event)
        if bar: bar.on_event(event)
        if emit: emit(event)
        try:
            while True:
                if cfg.max_updates is not None and updates_this_run >= cfg.max_updates:
                    stop_reason='update_limit'
                    break
                if time.perf_counter()+max(0.1, 1.1*last_update_seconds) >= deadline:
                    break
                if (base.completed_updates + 1)*spec['hands_per_update'] >= 20_000_000:
                    stop_reason='rollout_seed_namespace_limit'
                    break
                item = _one_update(session,step_cfg)
                _save(session,algorithm,paths['latest'])
                _event(paths['events'],item)
                state['completed_updates']=base.completed_updates
                updates_this_run+=1
                last_update_seconds = float(item['elapsed_seconds'])
                if bar: bar.on_event(item)
                safety = safety_assessment(
                    entropy=item.get('entropy', 100.0),
                    approx_kl=item.get('approx_kl', 0.0),
                    entropy_threshold=spec['entropy_alert_threshold'],
                    max_approx_kl=spec['max_approx_kl'],
                    entropy_stop_enabled=spec['entropy_stop_enabled'],
                    consecutive=state['consecutive_safety_warnings'],
                )
                if safety['low_entropy']:
                    state['entropy_warnings'] += 1
                    _event(paths['events'], {'kind':'entropy_warning',
                                           'update':base.completed_updates,
                                           'entropy':item['entropy']})
                if safety['high_kl']:
                    state['kl_warnings'] += 1
                    _event(paths['events'], {'kind':'kl_warning',
                                           'update':base.completed_updates,
                                           'approx_kl':item['approx_kl']})
                state['consecutive_safety_warnings'] = safety['consecutive_safety_warnings']
                _atomic_json(paths['state'],state)
                if base.completed_updates % spec['eval_every'] == 0:
                    if time.perf_counter() < deadline:
                        v = _validation(base,anchor,spec,cfg.device,state['baseline'])
                        if v['score_bb100'] > state['best_score_bb100']:
                            temp=paths['best'].with_name('best.pt.tmp')
                            shutil.copyfile(paths['latest'],temp)
                            os.replace(temp,paths['best'])
                            state['best_score_bb100']=v['score_bb100']
                            state['best_update']=base.completed_updates
                            state['best_sha256']=_sha256(paths['best'])
                        state['last_validation_update']=base.completed_updates
                        state['validation_count']+=1
                        _atomic_json(paths['state'],state)
                        _event(paths['events'],v,emit)
                if state['consecutive_safety_warnings'] >= spec['entropy_stop_patience']:
                    stop_reason='safety_stop'
                    break
            # Evaluate last saved policy, even if the invocation stopped between periodic checks.
            if (updates_this_run and state['last_validation_update'] != base.completed_updates
                    and time.perf_counter() < deadline):
                v=_validation(base,anchor,spec,cfg.device,state['baseline'])
                if v['score_bb100'] > state['best_score_bb100']:
                    tmp=paths['best'].with_name('best.pt.tmp')
                    shutil.copyfile(paths['latest'],tmp)
                    os.replace(tmp,paths['best'])
                    state['best_score_bb100']=v['score_bb100']
                    state['best_update']=base.completed_updates
                    state['best_sha256']=_sha256(paths['best'])
                state['last_validation_update']=base.completed_updates
                state['validation_count']+=1
                _atomic_json(paths['state'],state)
                _event(paths['events'],v,emit)
        except KeyboardInterrupt:
            stop_reason='keyboard_interrupt'
            # A mid-update interruption never overwrites the previous atomic checkpoint.
            state['completed_updates']=int(torch.load(paths['latest'],weights_only=True,map_location='cpu')['completed_updates'])
            _atomic_json(paths['state'],state)
        if bar: bar.on_event({'kind':'end','stop_reason':stop_reason})
    elapsed=time.perf_counter()-started
    peak = (round(torch.cuda.max_memory_allocated()/2**20,3) if cfg.device=='cuda' else None)
    summary=_summary(root,manifest,state,stop_reason=stop_reason,
                     updates_this_run=updates_this_run,elapsed=elapsed,peak=peak)
    _atomic_json(paths['summary'],summary)
    _event(paths['events'],{'kind':'end',**summary},emit)
    return summary


def _export_curve(root: Path) -> None:
    events=root/'metrics.jsonl'
    rows=[json.loads(line) for line in events.read_text(encoding='utf-8').splitlines() if line.strip()]
    for kind,cols,filename in (
        ('update',('update','hands','elapsed_seconds','hands_per_second_training','entropy',
                   'critic_loss','approx_kl','logit_variance_ratio'), 'training_curve.csv'),
        ('validation',('update','score_bb100','gain_vs_anchor_bb100','gain_vs_random_bb100'),
         'validation_curve.csv')):
        with (root/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=cols)
            writer.writeheader()
            for row in rows:
                if row.get('kind')==kind:
                    writer.writerow({key:row.get(key) for key in cols})


def _finalize_longrun_impl(root: str | Path, *, device: str = 'cpu') -> dict:
    """ONE held-out evaluation of best.pt; seals the run against further resume.

    Calling again verifies checkpoint SHA and returns the stored report.
    """
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    folder=Path(root)
    manifest,state,paths=_read(folder)
    _match_resume(LongRunConfig(root=str(folder),resume=True),manifest)
    best_sha=_sha256(paths['best'])
    if paths['final'].exists():
        cached=json.loads(paths['final'].read_text(encoding='utf-8'))
        if cached['best_sha256'] != best_sha:
            raise ValueError('best checkpoint has changed since final evaluation')
        return cached
    model=load_actor(paths['best'],device=device)
    anchor=load_actor(manifest['source'],device=device)
    spec=manifest['settings']
    big_blind=manifest['ppo_config']['big_blind']
    against_anchor=_evaluation(model,anchor,spec,device,final=True,big_blind=big_blind)
    against_random=_evaluation(model,None,spec,device,final=True,big_blind=big_blind)
    result={'phase':15,'partition':'final_held_out','algorithm':manifest['algorithm'],
            'best_checkpoint':str(paths['best']),'best_sha256':best_sha,
            'best_update':state['best_update'], 'initial_update':manifest['initial_update'],
            'validation_score_bb100':state['best_score_bb100'],
            'vs_frozen_initial':against_anchor,'vs_random':against_random,
            'final_seed':spec['final_seed'],'final_groups':spec['final_groups'],
            'note':'Test used only after checkpoint selection. Not GTO exploitability.'}
    _export_curve(folder)
    _atomic_json(paths['final'],result)  # Commit seal LAST after successful evaluation.
    return result


def run_longrun(cfg: LongRunConfig, *, emit: Callable[[dict],None] | None = None) -> dict:
    """OS-locked entry point for the crash-safe long-running training loop."""
    with _exclusive_lock(Path(cfg.root)):
        return _run_longrun_impl(cfg,emit=emit)


def finalize_longrun(root: str | Path, *, device: str = 'cpu') -> dict:
    """OS-locked held-out evaluation; no training can run concurrently."""
    with _exclusive_lock(Path(root)):
        return _finalize_longrun_impl(root,device=device)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/experiments/longrun.py" -->
