"""Independent optional E3 heldout benchmark; never changes weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from holdem_ai.expand.rl.trainer import evaluate_checkpoint


def main():
    p=argparse.ArgumentParser(description='Expand Phase3 checkpoint vs scripted 2-6 player opponents')
    p.add_argument('--candidate',required=True)
    p.add_argument('--hands-per-cell',type=int,default=100)
    p.add_argument('--num-players',default='2,3,4,5,6')
    p.add_argument('--opponents',default='random,calling_station,tight,pot_pressure')
    p.add_argument('--seed',type=int,default=340000000)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output-json',default='runs/expand_phase03/eval_heldout.json')
    args=p.parse_args()
    path=Path(args.output_json)
    if path.exists():raise FileExistsError('output exists; use a unique path')
    outcome=evaluate_checkpoint(args.candidate,device=args.device,hands=args.hands_per_cell,
             seed=args.seed,num_players=tuple(int(x) for x in args.num_players.split(',')),
             opponents=tuple(args.opponents.split(',')))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(outcome,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print(json.dumps({'output':str(path),'results':outcome['results']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
