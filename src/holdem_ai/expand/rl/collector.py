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
