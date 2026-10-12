"""Phase5 read-only toy / Hold'em search benchmark. NEVER alters a Phase4 checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
from holdem_ai.expand.pluribus.blueprint import KuhnGame,LeducGame,HoldemGame,load_checkpoint
from holdem_ai.expand.pluribus.search import (BlueprintPolicy,SearchBudget,SubgameSolver,
                                               ToyRootBelief,HoldemObservationBelief)


def parser():
    p=argparse.ArgumentParser(description='Expand Phase5 bounded belief-conditioned subgame search')
    p.add_argument('--game',choices=('kuhn','leduc','holdem'),default='kuhn')
    p.add_argument('--players',type=int,default=2)
    p.add_argument('--checkpoint',default=None,help='optional trusted Phase4 blueprint.json.gz')
    p.add_argument('--seed',type=int,default=505)
    p.add_argument('--root-seed',type=int,default=500505)
    p.add_argument('--stack-bb',type=int,default=2)
    p.add_argument('--time-ms',type=int,default=500)
    p.add_argument('--nodes',type=int,default=2500)
    p.add_argument('--memory-mb',type=int,default=64)
    p.add_argument('--iterations',type=int,default=80)
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--output-json',required=True)
    return p


def run(args):
    if args.checkpoint:
        trained=load_checkpoint(args.checkpoint)
        game=trained.game
        if game.name!=args.game or game.num_players!=args.players:
            raise ValueError('checkpoint game/player mismatch')
        table=trained.store
    else:
        table=None
        game={'kuhn':KuhnGame,'leduc':LeducGame,'holdem':HoldemGame}[args.game](args.players)
        if args.game=='holdem':
            game=HoldemGame(args.players,stack_bb=args.stack_bb)
    state=game.start(random.Random(args.root_seed))
    hero=state.current_player
    if game.name=='holdem':
        raw=state.engine.observe(hero)['raw_obs']
        belief=HoldemObservationBelief(raw,initial_stacks=state.engine.initial_stacks,
                                        small_blind=state.engine.small_blind,
                                        big_blind=state.engine.big_blind,
                                        abstraction=game.abstraction)
        abstraction_id=game.abstraction.fingerprint
    else:
        private=state.hole[hero] if game.name=='kuhn' else state.hole[hero]//2
        belief=ToyRootBelief(game,hero,private)
        abstraction_id=game.name+'-exact-v1'
    budget=SearchBudget(max_time_ms=args.time_ms,max_nodes=args.nodes,
                        max_memory_mb=args.memory_mb,max_iterations=args.iterations,
                        depth=args.depth)
    result=SubgameSolver(BlueprintPolicy(table,abstraction_id),budget,seed=args.seed).search(belief)
    out={'phase':'expand05','game':game.name,'players':game.num_players,'hero':hero,
         'status':result.status,'iterations':result.iterations,'nodes':result.nodes,
         'elapsed_ms':result.elapsed_ms,'action_probabilities':result.action_probabilities,
         'continuation_samples':result.continuation_samples,
         'abstraction_id':result.abstraction_id,'search_root_id':result.search_root_id,
         'estimated_table_mb':result.estimated_table_mb,'reason':result.reason,
         'notes':['belief conditioned on known hero and public cards only',
                  'current-decision approximation; NOT full Pluribus nested search',
                  'rollout leaf mixture is NOT the original leaf continuation game']}
    dest=Path(args.output_json)
    if dest.exists(): raise FileExistsError('refusing to overwrite Phase5 report')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return out


def main():run(parser().parse_args())
if __name__=='__main__':main()
