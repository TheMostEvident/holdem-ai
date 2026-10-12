"""Phase 15 HUNL long-run training, from a Phase14 best.pt (or trusted PPO/QBoost checkpoint)."""
from __future__ import annotations

import argparse
import json

from holdem_ai.experiments.longrun import LongRunConfig, run_longrun


def build_parser():
    p=argparse.ArgumentParser(description='Phase15 safe long-run PPO/QBoost self-play training')
    p.add_argument('--source-checkpoint',default=None)
    p.add_argument('--algorithm',choices=('ppo','qboost'),default=None)
    p.add_argument('--root',default='runs/phase15_ppo')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--duration-minutes',type=float,default=300.0)
    p.add_argument('--max-updates',type=int,default=None,help='additional updates this invocation')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-progress',action='store_true')
    for name,kwargs in (
        ('hands-per-update',dict(type=int)), ('env-batch-size',dict(type=int)),
        ('minibatch-size',dict(type=int)), ('epochs',dict(type=int)),
        ('rollout-seed',dict(type=int)),('eval-every',dict(type=int)),
        ('eval-groups',dict(type=int)),('validation-seed',dict(type=int)),
        ('final-groups',dict(type=int)),('final-seed',dict(type=int)),
        ('entropy-alert-threshold',dict(type=float)),
        ('entropy-stop-patience',dict(type=int)),('max-approx-kl',dict(type=float)),
        ('beta',dict(type=float)),
    ):
        p.add_argument('--'+name,default=None,**kwargs)
    p.add_argument('--eval-mode',choices=('greedy','sampled'),default=None)
    return p


def run(args):
    options=vars(args).copy()
    options['progress']=not options.pop('no_progress')
    cfg=LongRunConfig(**options)
    def emit(e):
        if e.get('kind') in ('start','resume','validation','end'):
            print(json.dumps(e,ensure_ascii=False),flush=True)
    result=run_longrun(cfg,emit=emit)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    run(build_parser().parse_args())


if __name__=='__main__':
    main()
