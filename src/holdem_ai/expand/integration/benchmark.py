"""E7 reproducible *independent-session* cash and tournament benchmarks.

Cash uses one scored hand per freshly initialized table, not iid sequential
hands. Bootstrap hands used to admit BB-waiting players are reported separately.
Tournament uses one full tournament as one cluster; no hand-iid CI or ICM claim.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable
import math
import random
import statistics
from holdem_ai.expand.environment import DynamicTable, TableConfig
from holdem_ai.expand.information import Phase2TableAdapter
from holdem_ai.expand.information.builder import make_information_state
from holdem_ai.expand.information.schema import PublicAction
from holdem_ai.expand.rl.population import scripted_action
from holdem_ai.expand.opponent.bridge import PublicOpponentTracker
from .agent import IntegratedPokerAgent
from .tournament import SingleTableTournament, TournamentConfig


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str = 'cash'
    groups: int = 4
    seed: int = 700000007
    players: tuple[int,...] = (2,3,4,5,6)
    stacks_bb: tuple[int,...] = (20,100,200)
    opponents: tuple[str,...] = ('random','calling_station','tight','pot_pressure')
    max_actions: int = 400
    max_warmup_hands: int = 90
    max_tournament_hands: int = 120

    def __post_init__(self):
        if self.mode not in ('cash','tournament'):raise ValueError('invalid benchmark mode')
        if type(self.groups) is not int or self.groups<2:raise ValueError('groups >=2 required for CI')
        if type(self.seed) is not int or self.seed<0:raise ValueError('seed must be nonnegative')
        if not self.players or any(type(n) is not int or n<2 or n>6 for n in self.players):
            raise ValueError('players must be 2..6')
        if not self.stacks_bb or any(type(s) is not int or not 20<=s<=300 for s in self.stacks_bb):
            raise ValueError('stack sizes must be 20..300 BB')
        if not self.opponents or any(x not in ('random','calling_station','tight','pot_pressure')
                                      for x in self.opponents):raise ValueError('invalid opponent')
        if any(type(getattr(self,x)) is not int or getattr(self,x)<1
               for x in ('max_actions','max_warmup_hands','max_tournament_hands')):
            raise ValueError('positive episode guards required')


def _stats(values:list[float], factor:float=1.)->dict:
    if len(values)<2:raise ValueError('at least 2 independent blocks required')
    mean=statistics.mean(values)*factor
    se=statistics.stdev(values)*factor/math.sqrt(len(values))
    return {'n_independent_sessions':len(values),'mean':round(mean,6),
            'se':round(se,6),'ci95_normal':[round(mean-1.96*se,6),round(mean+1.96*se,6)]}


def _cash_session(agent:IntegratedPokerAgent,*,n:int,stack_bb:int,style:str,
                  scenario_seed:int,hero_seat:int,engine_factory:Callable|None,
                  max_actions:int,max_warmup_hands:int)->dict:
    rng=random.Random(scenario_seed)
    tc=TableConfig()
    table=DynamicTable(tc,session_id=f'e07-{scenario_seed}',engine_factory=engine_factory)
    adapter=Phase2TableAdapter(table)
    tracker=PublicOpponentTracker()
    tenures={}
    for seat in range(n):
        player='hero' if seat==hero_seat else f'opp_{seat}'
        tenures[player]=table.request_join(player,stack_bb*tc.big_blind,preferred_seat=seat)
    warmups=0
    while True:
        handle=adapter.start_hand(scenario_seed*101+warmups)
        if handle is None:raise RuntimeError('table stalled with insufficient active participants')
        roster=dict(handle.participants)
        tracker.sync_seats(table.snapshot(),tenures)
        ready=(len(roster)==n and roster.get(hero_seat)=='hero')
        actions=0
        while table.current_player_id is not None:
            if actions>=max_actions:raise RuntimeError('cash episode decision guard triggered')
            state=adapter.observe()
            pid=table.current_player_id
            if ready and pid=='hero':
                model=None
                # Opponent evidence is per-opponent & contextual; no oracle.
                # Phase 6 routing falls back if there is insufficient evidence.
                for (key_id,join_id),candidate in tracker.registry.models.items():
                    if key_id!='hero' and candidate.contexts:
                        context=next(reversed(candidate.contexts))
                        model=candidate.evidence(context)
                        break
                chosen=agent.act(state,rng.randrange(2**63),evidence=model).action_name
            else:
                chosen=scripted_action(style,state,rng)
            adapter.act_abstract(chosen)
            tracker.record_accepted_action(state,handle=handle,player_id=pid,
                       player_session_id=tenures[pid],action_index=actions,action=chosen)
            actions+=1
        settled=adapter.settle_hand()
        tracker.close_hand(handle)
        if sum(settled.payoffs_chips)!=0:
            raise AssertionError('cash payoff not zero-sum')
        if ready:
            return {'payoff_bb':settled.payoffs_bb[hero_seat], 'hands':1,
                    'warmup_hands':warmups,'illegal_actions':0,
                    'opponent_events':sum(len(m.seen_events) for m in tracker.registry.models.values())}
        warmups+=1
        if warmups>=max_warmup_hands:
            raise RuntimeError('full roster was not reached before admission guard')
        # A busted player during BB-waiting needs a fresh buy-in before the
        # measured game; give the fresh identity a new tenure.
        snapshot=table.snapshot()
        for seat in range(n):
            if snapshot.seats[seat].player_id is None:
                pid='hero' if seat==hero_seat else f'opp_{seat}'
                tenures[pid]=table.request_join(pid,stack_bb*tc.big_blind,preferred_seat=seat)


def cash_benchmark(agent:IntegratedPokerAgent, config:BenchmarkConfig,
                   *, engine_factory:Callable|None=None)->dict:
    if config.mode!='cash':raise ValueError('requires cash benchmark config')
    rows=[]
    all_values=[]
    total_warmup=0
    for n in config.players:
        for stacks in config.stacks_bb:
            for style in config.opponents:
                values=[]
                for b in range(config.groups):
                    # Shared deterministic scenario seeds for A/B comparisons.
                    scenario=config.seed+n*10000000+stacks*10000+config.opponents.index(style)*1000+b
                    run=_cash_session(agent,n=n,stack_bb=stacks,style=style,scenario_seed=scenario,
                          hero_seat=b%n,engine_factory=engine_factory,
                          max_actions=config.max_actions,max_warmup_hands=config.max_warmup_hands)
                    values.append(run['payoff_bb'])
                    all_values.append(run['payoff_bb'])
                    total_warmup+=run['warmup_hands']
                rows.append({'players':n,'stack_bb':stacks,'opponent':style,
                             'bb100':_stats(values,100.),'block_payoffs_bb':values})
    # Aggregate is descriptive. Independent sessions are the sampling unit,
    # but across non-identical opponents/stack groups it is a mixture estimand.
    return {'mode':'cash','config':asdict(config),'overall_bb100':_stats(all_values,100.),
            'strata':rows,'scored_hands':len(all_values),'admission_warmup_hands':total_warmup,
            'note':'One scored hand per independent table; no training or opponent adaptation advantage claimed.'}


def _tournament_session(agent:IntegratedPokerAgent, *, n:int,style:str,seed:int,hero_seat:int,
                        engine_factory:Callable|None, max_actions:int,max_hands:int):
    rng=random.Random(seed)
    cfg=TournamentConfig(starting_stack=2000,hands_per_level=2,max_hands=max_hands)
    tour=SingleTableTournament(cfg,engine_factory=engine_factory)
    for seat in range(n):
        tour.register('hero' if seat==hero_seat else f'opp_{seat}',seat=seat)
    tour.close_registration()
    public_history=[]
    while not tour.finished:
        hand=tour.start_hand(seed*1000+tour.hands_played)
        if hand is None:
            if tour.finished:break
            raise RuntimeError('unfinishable tournament')
        actions=0
        public_history=[]
        while tour.current_player_id is not None:
            if actions>=max_actions:raise RuntimeError('tournament decision guard triggered')
            raw=tour.observe_current()
            info=make_information_state(raw,big_blind_chips=hand.big_blind,
                seat_presence=tuple(e is not None for e in tour.seats),
                seat_statuses=tuple('seated' if e else 'vacant' for e in tour.seats),
                history=tuple(public_history))
            if tour.current_player_id=='hero':
                chosen=agent.act(info,rng.randrange(2**63)).action_name
            else:
                chosen=scripted_action(style,info,rng)
            from holdem_ai.expand.information.actions import to_table_action
            table_action=to_table_action(info.legal_actions,chosen)
            raise_target=table_action.raise_to_chips
            if chosen=='all_in':
                r=raw['raw_obs']
                raise_target=r['street_bets'][info.hero_seat]+r['stacks'][info.hero_seat]
            tour.act(table_action)
            public_history.append(PublicAction(info.hero_seat,info.street,chosen,
                 raise_target/hand.big_blind if raise_target is not None else None,info.to_call_bb,info.pot_bb))
            actions+=1
        result=tour.settle_hand()
        if sum(result.payoffs_chips)!=0:raise AssertionError('tournament hand nonzero sum')
    report=tour.result()
    hero_prize=sum(e[3] for e in report.entries if e[1]=='hero')
    hero_win=float(any(e[1]=='hero' and e[2]==1 for e in report.entries))
    return hero_prize,hero_win,report.hands_played


def tournament_benchmark(agent:IntegratedPokerAgent, config:BenchmarkConfig,
                         *, engine_factory:Callable|None=None)->dict:
    if config.mode!='tournament':raise ValueError('requires tournament benchmark config')
    rows=[]
    prizes=[]
    for n in config.players:
        for style in config.opponents:
            p=[];wins=[];lengths=[]
            for b in range(config.groups):
                scenario=config.seed+n*10000000+config.opponents.index(style)*1000+b
                value,won,length=_tournament_session(agent,n=n,style=style,seed=scenario,
                   hero_seat=b%n,engine_factory=engine_factory,
                   max_actions=config.max_actions,max_hands=config.max_tournament_hands)
                p.append(value);wins.append(won);lengths.append(length);prizes.append(value)
            rows.append({'players':n,'opponent':style,'prize_ev':_stats(p),
                         'win_rate':sum(wins)/len(wins),
                         'mean_hands':statistics.mean(lengths),'session_prizes':p})
    return {'mode':'tournament','config':asdict(config),'overall_prize_ev':_stats(prizes),
            'strata':rows,'tournaments':len(prizes),
            'note':'Session-clustered prize EV. No ICM estimate, tournament hand bb100, or multi-table support.'}


def paired_cash_difference(left:dict,right:dict)->dict:
    if left['config']!=right['config']:
        raise ValueError('paired comparison requires identical scenario configuration')
    la=left['strata'];ra=right['strata']
    if len(la)!=len(ra):raise ValueError('different stratification')
    diffs=[]
    for a,b in zip(la,ra):
        if (a['players'],a['stack_bb'],a['opponent'])!=(b['players'],b['stack_bb'],b['opponent']):
            raise ValueError('different strata')
        diffs += [x-y for x,y in zip(a['block_payoffs_bb'],b['block_payoffs_bb'])]
    return {'paired_difference_bb100':_stats(diffs,100.), 'direction':'left_minus_right'}
