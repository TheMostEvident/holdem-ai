"""E7 no-training cash/tournament benchmark; never overwrites reports."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from holdem_ai.expand.integration import (IntegratedPokerAgent, RoutingConfig,
                                           from_e03_checkpoint, uniform_legal,
                                           release_manifest, write_new_json)
from holdem_ai.expand.integration.benchmark import BenchmarkConfig, cash_benchmark, tournament_benchmark, paired_cash_difference


def parser():
    p=argparse.ArgumentParser(description='Expand E7 holdout 6-max cash / tournament evaluation only')
    p.add_argument('--mode',choices=('cash','tournament','both'),default='cash')
    p.add_argument('--groups',type=int,default=4)
    p.add_argument('--seed',type=int,default=7070007)
    p.add_argument('--players',default='2,3,4,5,6')
    p.add_argument('--stacks-bb',default='20,100,200')
    p.add_argument('--opponents',default='random,calling_station,tight,pot_pressure')
    p.add_argument('--checkpoint',default=None,help='E3 checkpoint, omit for explicitly labelled uniform baseline')
    p.add_argument('--baseline-checkpoint',default=None,help='optional second E3 checkpoint for paired cash A-B')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--adaptive',action='store_true',help='enable E6 routing only with trained expert checkpoints')
    p.add_argument('--expert-checkpoint',action='append',default=[],metavar='STYLE=PATH')
    p.add_argument('--max-actions',type=int,default=400)
    p.add_argument('--max-warmup-hands',type=int,default=90)
    p.add_argument('--max-tournament-hands',type=int,default=120)
    p.add_argument('--output-json',required=True)
    return p


def run(args):
    if args.device=='cuda' and args.checkpoint is None and args.baseline_checkpoint is None:
        raise ValueError('CUDA model execution requires at least one neural checkpoint')
    cfg=BenchmarkConfig(mode='cash' if args.mode=='both' else args.mode,
              groups=args.groups,seed=args.seed,
              players=tuple(int(x) for x in args.players.split(',')),
              stacks_bb=tuple(int(x) for x in args.stacks_bb.split(',')),
              opponents=tuple(x.strip() for x in args.opponents.split(',')),
              max_actions=args.max_actions,max_warmup_hands=args.max_warmup_hands,
              max_tournament_hands=args.max_tournament_hands)
    experts={}
    for item in args.expert_checkpoint:
        if '=' not in item: raise ValueError('expert format must be STYLE=PATH')
        style,path=item.split('=',1)
        if style not in ('fold_heavy','calling','aggressive') or not path or style in experts:
            raise ValueError('unique valid style and nonempty path required')
        experts[style]=from_e03_checkpoint(path,device=args.device)
    if args.adaptive and not experts: raise ValueError('adaptive mode needs externally trained hero policy experts')
    if experts and not args.adaptive:raise ValueError('expert checkpoints require --adaptive')
    agent=IntegratedPokerAgent(from_e03_checkpoint(args.checkpoint,device=args.device)
          if args.checkpoint else uniform_legal,config=RoutingConfig(use_adaptation=args.adaptive),
          expert_policies=experts)
    result={'phase':'expand07','candidate':'e03_checkpoint' if args.checkpoint else 'uniform_legal_baseline',
            'note':'Evaluation only, no PPO/MCCFR training. E4/E5 search is opt-in source API only; E6 adaptation requires checkpoints.'}
    if args.mode in ('cash','both'):
        result['cash']=cash_benchmark(agent,replace(cfg,mode='cash'))
    if args.mode in ('tournament','both'):
        result['tournament']=tournament_benchmark(agent,replace(cfg,mode='tournament'))
    if args.baseline_checkpoint:
        if 'cash' not in result:
            raise ValueError('paired baseline requires cash mode')
        baseline=IntegratedPokerAgent(from_e03_checkpoint(args.baseline_checkpoint,device=args.device))
        result['baseline_cash']=cash_benchmark(baseline,replace(cfg,mode='cash'))
        result['paired_cash']=paired_cash_difference(result['cash'],result['baseline_cash'])
    root=Path.cwd()
    result['release_manifest']=release_manifest(project_root=root,config=asdict(cfg),seeds=[args.seed],
                   checkpoints={**{k:v for k,v in (('candidate',args.checkpoint),('baseline',args.baseline_checkpoint)) if v},**{f'expert_{k}':item.split('=',1)[1] for item in args.expert_checkpoint for k in [item.split('=',1)[0]]}},
                   abstraction_id='none-neural-only',scripts=['scripts/expand_phase07_benchmark.py'])
    write_new_json(args.output_json,result)
    print(json.dumps({'phase':7,'cash_hands':result.get('cash',{}).get('scored_hands',0),
          'tournaments':result.get('tournament',{}).get('tournaments',0),
          'candidate':result['candidate'],'report':args.output_json},ensure_ascii=False,indent=2))
    return result


def main():run(parser().parse_args())
if __name__=='__main__':main()
