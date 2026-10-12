"""Expand Phase 3: 2-6 player PPO self-play on the existing cash game engine.

Use --resume with only --root, --max-updates and --device. No Phase 15
checkpoint can be loaded here: this is a separate 6-seat observation model.
"""
from __future__ import annotations
import argparse
import json

from holdem_ai.expand.rl import RolloutConfig, PPOConfig, OpponentMix
from holdem_ai.expand.rl.trainer import TrainConfig, train


def build_parser():
    p=argparse.ArgumentParser(description='Expand Phase3 6-max self-play PPO, CPU/CUDA')
    p.add_argument('--root',default='runs/expand_phase03_pilot')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--max-updates',type=int,default=3,help='ADDITIONAL PPO updates in this invocation')
    p.add_argument('--hands-per-update',type=int,default=None)
    p.add_argument('--num-players',default=None,help='comma-separated player counts, e.g. 2,3,4,5,6')
    p.add_argument('--stack-bb',type=int,default=None)
    p.add_argument('--random-stack-range-bb',default=None,help='min,max; overrides fixed stack when enabled')
    p.add_argument('--width',type=int,default=None)
    p.add_argument('--heads',type=int,default=None)
    p.add_argument('--layers',type=int,default=None)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--rollout-seed',type=int,default=None)
    p.add_argument('--validation-seed',type=int,default=None)
    p.add_argument('--eval-every',type=int,default=None)
    p.add_argument('--eval-hands',type=int,default=None)
    p.add_argument('--snapshot-every',type=int,default=None)
    p.add_argument('--max-snapshots',type=int,default=None)
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--minibatch-size',type=int,default=None)
    p.add_argument('--learning-rate',type=float,default=None)
    p.add_argument('--shared-fraction',type=float,default=None)
    p.add_argument('--snapshot-fraction',type=float,default=None)
    return p


def make_config(args):
    kw=vars(args)
    if args.resume:
        provided=[key for key,v in kw.items() if key not in ('root','device','resume','max_updates') and v is not None]
        if provided:
            raise ValueError('resume loads immutable settings; remove flags: '+', '.join(provided))
        return None
    default=TrainConfig()
    values={k:kw[k] if kw[k] is not None else getattr(default,k) for k in
            ('width','heads','layers','seed','rollout_seed','validation_seed','eval_every','eval_hands',
             'snapshot_every','max_snapshots')}
    r=default.rollout
    mix=OpponentMix(shared_policy_fraction=args.shared_fraction if args.shared_fraction is not None else r.mix.shared_policy_fraction,
                    snapshot_fraction=args.snapshot_fraction if args.snapshot_fraction is not None else r.mix.snapshot_fraction)
    np=tuple(int(x) for x in args.num_players.split(',')) if args.num_players is not None else r.num_players
    stack_range=(tuple(int(x) for x in args.random_stack_range_bb.split(','))
                 if args.random_stack_range_bb is not None else None)
    rollout=RolloutConfig(hands_per_update=args.hands_per_update or r.hands_per_update,
                          num_players=np,starting_stack_bb=args.stack_bb or r.starting_stack_bb,
                          random_stack_range_bb=stack_range,mix=mix)
    p=default.ppo
    ppo=PPOConfig(epochs=args.epochs or p.epochs,
                  minibatch_size=args.minibatch_size or p.minibatch_size,
                  learning_rate=args.learning_rate or p.learning_rate)
    return TrainConfig(rollout=rollout,ppo=ppo,**values)


def main():
    args=build_parser().parse_args()
    cfg=make_config(args)
    outcome=train(args.root,config=cfg,resume=args.resume,additional_updates=args.max_updates,
                  device=args.device,emit=lambda item: print(json.dumps(item,ensure_ascii=False),flush=True))
    print(json.dumps(outcome,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
