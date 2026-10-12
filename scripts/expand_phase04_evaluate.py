"""Read-only benchmark of a Phase4 blueprint against a uniform legal agent."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from holdem_ai.expand.pluribus.blueprint import load_checkpoint
from holdem_ai.expand.pluribus.blueprint.evaluate import evaluate_vs_uniform


def parser():
    p=argparse.ArgumentParser(description='Evaluate Phase4 frozen strategy without training')
    p.add_argument('--root',required=True)
    p.add_argument('--hands',type=int,default=200)
    p.add_argument('--seed',type=int,default=440000000)
    p.add_argument('--mode',choices=('average','current'),default='average')
    p.add_argument('--output-json',required=True)
    return p


def main():
    args=parser().parse_args()
    destination=Path(args.output_json)
    if destination.exists():
        raise FileExistsError('output exists, choose a new path')
    model=load_checkpoint(Path(args.root)/'blueprint.json.gz')
    result=evaluate_vs_uniform(model,hands=args.hands,seed=args.seed,mode=args.mode)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
