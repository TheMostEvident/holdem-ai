"""Paired-deal toy benchmark: search vs the SAME immutable Phase4 blueprint.

For 2-player Kuhn/Leduc only; a sanity benchmark, not 6-max exploitability.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import statistics
from holdem_ai.expand.pluribus.blueprint import KuhnGame,LeducGame,load_checkpoint
from holdem_ai.expand.pluribus.search import BlueprintPolicy,SearchBudget,SubgameSolver,ToyRootBelief


def parser():
    p=argparse.ArgumentParser(description='Phase5 toy search / blueprint paired-deal comparison')
    p.add_argument('--game',choices=('kuhn','leduc'),default='kuhn')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--hands',type=int,default=30,help='deals per hero seat')
    p.add_argument('--seed',type=int,default=550005)
    p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--time-ms',type=int,default=500)
    p.add_argument('--nodes',type=int,default=2500)
    p.add_argument('--output-json',required=True)
    return p


def _play(game,blueprint,seed,hero,with_search,budget):
    state=game.start(random.Random(seed))
    rng=random.Random(seed+2000+hero)
    fallback=0
    calls=0
    while not state.is_terminal():
        acting=state.current_player
        actions=state.legal_actions()
        if acting==hero and with_search:
            own=state.hole[hero] if game.name=='kuhn' else state.hole[hero]//2
            board=state.board//2 if game.name=='leduc' and state.street==1 else None
            public_actions=tuple(step[2] for step in state.history)
            belief=ToyRootBelief(game,hero,own,history=public_actions,board_rank=board)
            sol=SubgameSolver(blueprint,budget,seed=seed+hero*10000+len(public_actions)).search(belief)
            probs=sol.action_probabilities
            calls+=1
            if sol.status!='solved':fallback+=1
        else:
            probs=blueprint.policy(state.infoset(acting),actions)
        action=rng.choices(list(probs),weights=list(probs.values()),k=1)[0]
        state=state.next_state(action)
    return state.terminal_payoffs()[hero],calls,fallback


def run(args):
    if args.hands<2:raise ValueError('at least 2 deals required')
    trained=load_checkpoint(args.checkpoint)
    if trained.game.name!=args.game or trained.game.num_players!=2:
        raise ValueError('requires a 2-player matching Phase4 checkpoint')
    game=trained.game
    blueprint=BlueprintPolicy(trained.store,args.game+'-exact-v1')
    budget=SearchBudget(max_iterations=args.iterations,max_time_ms=args.time_ms,max_nodes=args.nodes)
    differences=[];search_rewards=[];base_rewards=[];calls=0;fallbacks=0
    for hand in range(args.hands):
        for hero in (0,1):
            seed=args.seed+hand
            a,c,f=_play(game,blueprint,seed,hero,True,budget)
            b,_,_=_play(game,blueprint,seed,hero,False,budget)
            differences.append(a-b);search_rewards.append(a);base_rewards.append(b)
            calls+=c;fallbacks+=f
    mean=statistics.mean(differences)
    se=statistics.stdev(differences)/math.sqrt(len(differences))
    output={'phase':'expand05','game':args.game,'hands':len(differences),
            'paired_mean_difference':mean,'ci95_normal':[mean-1.96*se,mean+1.96*se],
            'search_mean':statistics.mean(search_rewards),
            'blueprint_mean':statistics.mean(base_rewards),
            'search_calls':calls,'non_solved_calls':fallbacks,
            'caveats':['within-deal/seat outcomes are not fully independent',
                       'normal CI exploratory only: correlated seats and small samples',
                       'NOT exact Pluribus search or 6-max exploitability']}
    path=Path(args.output_json)
    if path.exists():raise FileExistsError('output exists; avoid overwriting')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))
    return output


def main():run(parser().parse_args())
if __name__=='__main__':main()
