"""Expand Phase4 reference MCCFR. CPU-friendly prototype; no GPU requirement."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from holdem_ai.expand.pluribus.blueprint import (MCCFRSolver,MCCFRConfig,KuhnGame,
                                                   LeducGame,HoldemGame,save_checkpoint,load_checkpoint)
from holdem_ai.expand.pluribus.blueprint.evaluate import evaluate_vs_uniform
from holdem_ai.expand.pluribus.abstraction import AbstractionConfig


def parser():
    p=argparse.ArgumentParser(description='Expand Phase 4: External-Sampling Linear MCCFR (experimental)')
    p.add_argument('--game',choices=('kuhn','leduc','holdem'),default='kuhn')
    p.add_argument('--players',type=int,default=2)
    p.add_argument('--stack-bb',type=int,default=20)
    p.add_argument('--postflop-buckets',type=int,default=16)
    p.add_argument('--allow-half-pot',action='store_true')
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--root',default='runs/expand_phase04_kuhn')
    p.add_argument('--seed',type=int,default=400000004)
    p.add_argument('--node-budget',type=int,default=20000)
    p.add_argument('--max-infosets',type=int,default=200000)
    p.add_argument('--max-depth',type=int,default=256)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--no-linear-weighting',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--eval-hands',type=int,default=0)
    p.add_argument('--eval-seed',type=int,default=440000000)
    return p


def run(args):
    root=Path(args.root)
    target=root/'blueprint.json.gz'
    if args.resume:
        if not target.is_file():
            raise FileNotFoundError(f'resume checkpoint missing: {target}')
        solver=load_checkpoint(target)
        # Resume never changes training hyperparameters; fresh CLI flags which
        # differ from defaults are rejected rather than silently ignored.
        defaults=parser().parse_args([])
        for param in ('game','players','stack_bb','postflop_buckets','allow_half_pot',
                      'seed','node_budget','max_infosets','max_depth','no_linear_weighting'):
            if getattr(args,param)!=getattr(defaults,param):
                raise ValueError(f'resume restores saved configuration; omit --{param.replace("_","-")}')
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('new experiment needs empty/nonexistent directory; use --resume')
        if args.game=='kuhn':game=KuhnGame(args.players)
        elif args.game=='leduc':game=LeducGame(args.players)
        else:
            game=HoldemGame(args.players,stack_bb=args.stack_bb,
                            abstraction=AbstractionConfig(args.postflop_buckets,args.allow_half_pot))
        cfg=MCCFRConfig(seed=args.seed,max_nodes_per_iteration=args.node_budget,
                        max_infosets=args.max_infosets,max_depth=args.max_depth,linear_weighting=not args.no_linear_weighting)
        solver=MCCFRSolver(game,cfg)
    if args.iterations <= 0 or args.save_every <= 0:
        raise ValueError('iterations and save-every must be positive')
    root.mkdir(parents=True,exist_ok=True)
    last=None
    pending_events=[]
    def commit_batch():
        if not pending_events:return
        save_checkpoint(target,solver)
        # Event log contains only updates that are persisted in the checkpoint.
        with (root/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            for event in pending_events:
                stream.write(json.dumps(event,sort_keys=True)+'\n')
        pending_events.clear()
    from holdem_ai.expand.pluribus.blueprint.mccfr import SearchBudgetExceeded
    error=None
    try:
        for k in range(args.iterations):
            last=solver.iterate()
            pending_events.append(last)
            if (k+1)%args.save_every==0:
                commit_batch()
            if (k+1)%100==0:
                print(json.dumps(last),flush=True)
    except SearchBudgetExceeded as exc:
        error=str(exc)
    except KeyboardInterrupt:
        error='keyboard_interrupt'
    # Save all successful iterations, even if stopped by budget/interrupt.
    commit_batch()
    ck=save_checkpoint(target,solver)
    result={'phase':'Expand Phase 4','game':solver.game.name,
            'players':solver.game.num_players,**ck, 'last_iteration':last,
            'stopped_by_budget':error}
    if args.eval_hands:
        result['evaluation']=evaluate_vs_uniform(solver,hands=args.eval_hands,seed=args.eval_seed)
    if solver.game.name=='kuhn' and solver.game.num_players==2:
        from holdem_ai.expand.pluribus.blueprint.kuhn_reference import kuhn_nash_conv
        result['toy_nash_conv']=kuhn_nash_conv(solver.store)
    (root/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


def main():run(parser().parse_args())


if __name__=='__main__':main()
