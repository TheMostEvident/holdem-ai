"""Seed-blocked synthetic change-point diagnostic: NOT poker win-rate testing.

All actions are publicly visible and are used for online one-step-ahead
forecasting. Static-uniform baseline is evaluated on the EXACT SAME draws.
The adaptive gate uses a fixed EXTERNAL proposed strategy, not opponent cards.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import statistics
from holdem_ai.expand.opponent import (OpponentConfig,OpponentRegistry,PublicDecision,
                                      SafeAdaptivePolicy,GuardConfig,StyleExpertRouter,save_registry)

LEGAL=('fold','call','raise_pot')
STYLE={
 'tight':(.76,.20,.04),
 'loose':(.15,.65,.20),
 'aggressive':(.08,.17,.75),
 'balanced':(.33,.34,.33),
}
CTX=('preflop','button','4to10',LEGAL)
BASE={'fold':.3,'call':.4,'raise_pot':.3}
# Diagnostic toy policies only, NOT learned poker counter-strategies.
EXPERTS={
 'fold_heavy':{'fold':.1,'call':.3,'raise_pot':.6},
 'calling':{'fold':.05,'call':.25,'raise_pot':.70},
 'aggressive':{'fold':.5,'call':.4,'raise_pot':.1},
}


def parser():
    p=argparse.ArgumentParser(description='Expand Phase6 public-only synthetic style-shift benchmark')
    p.add_argument('--blocks',type=int,default=16)
    p.add_argument('--warmup',type=int,default=90)
    p.add_argument('--post-shift',type=int,default=90)
    p.add_argument('--initial-style',choices=sorted(STYLE),default='tight')
    p.add_argument('--new-style',choices=sorted(STYLE),default='aggressive')
    p.add_argument('--seed',type=int,default=660006)
    p.add_argument('--output-json',required=True)
    p.add_argument('--checkpoint',default=None,help='optional last-block public-only checkpoint')
    return p


def _sample(style:str,rng:random.Random)->str:
    return rng.choices(LEGAL,weights=STYLE[style],k=1)[0]


def block(args,block_id:int):
    cfg=OpponentConfig()
    registry=OpponentRegistry(cfg)
    registry.bind_seat(3,'opponent','join-tenure-1')
    model=registry.models[('opponent','join-tenure-1')]
    gate=SafeAdaptivePolicy(cfg,GuardConfig())
    router=StyleExpertRouter(cfg)
    # Disjoint deterministic random streams, independent per scenario block.
    rng=random.Random(args.seed+block_id*104729)
    static_loss=[];online_loss=[]
    shifted_fallbacks=0; adapted=0;max_kl=0.;max_tv=0.;shift_detections=[]
    route_styles={'fold_heavy':0,'calling':0,'aggressive':0,'unrouted':0}
    for i in range(args.warmup+args.post_shift):
        style=args.initial_style if i<args.warmup else args.new_style
        # Predict before observing the current public action: no target leakage.
        prior=model.evidence(CTX).posterior
        selected=_sample(style,rng)
        prob=prior[selected]
        static_loss.append(-math.log(1/len(LEGAL)))
        online_loss.append(-math.log(prob))
        evidence=model.evidence(CTX)
        route=router.propose(evidence,EXPERTS,hero_legal=LEGAL)
        route_styles[route.style if route.proposal is not None else 'unrouted']+=1
        adjusted=gate.adjust(BASE,route.proposal,evidence,legal=LEGAL,
                             expected_opponent_context=CTX)
        adapted+=int(adjusted.used_proposal)
        max_kl=max(max_kl,adjusted.kl_from_base)
        max_tv=max(max_tv,adjusted.tv_from_base)
        if i>=args.warmup and adjusted.reason=='change_point_fallback':shifted_fallbacks+=1
        action=PublicDecision('opponent','join-tenure-1',f'hand-{i}',i,3,'preflop','button','4to10',LEGAL,selected)
        out=registry.observe(action)
        model.end_hand(action.hand_id)
        if i>=args.warmup and out.shift_active:
            shift_detections.append(i-args.warmup)
    return {'block':block_id,'n':len(online_loss),'uniform_logloss':statistics.mean(static_loss),
            'online_logloss':statistics.mean(online_loss),
            'logloss_delta_uniform_minus_online':statistics.mean(static_loss)-statistics.mean(online_loss),
            'first_shift_signal_offset':min(shift_detections) if shift_detections else None,
            'shift_fallback_decisions':shifted_fallbacks,'adapted_decisions':adapted,
            'max_kl':max_kl,'max_tv':max_tv,'expert_route_counts':route_styles,'registry':registry}


def run(args):
    if args.blocks<2 or args.warmup<10 or args.post_shift<10 or args.seed<0:
        raise ValueError('blocks>=2, warmup/post-shift>=10, seed>=0 required')
    path=Path(args.output_json)
    if path.exists():raise FileExistsError('output already exists')
    if args.checkpoint is not None and Path(args.checkpoint).exists():
        raise FileExistsError('checkpoint already exists')
    data=[block(args,i) for i in range(args.blocks)]
    last=data[-1].pop('registry')
    for row in data[:-1]:row.pop('registry')
    diff=[r['logloss_delta_uniform_minus_online'] for r in data]
    se=statistics.stdev(diff)/math.sqrt(len(diff))
    out={'phase':'expand06','scope':'synthetic_public_action_forecasting_only',
         'seed':args.seed,'blocks':args.blocks,'warmup':args.warmup,'post_shift':args.post_shift,
         'initial_style':args.initial_style,'new_style':args.new_style,
         'mean_online_logloss':statistics.mean(r['online_logloss'] for r in data),
         'mean_static_logloss':statistics.mean(r['uniform_logloss'] for r in data),
         'mean_block_improvement':statistics.mean(diff),
         'ci95_normal_block_level':[statistics.mean(diff)-1.96*se,statistics.mean(diff)+1.96*se],
         'shift_detected_blocks':sum(r['first_shift_signal_offset'] is not None for r in data),
         'max_observed_kl':max(r['max_kl'] for r in data),
         'max_observed_tv':max(r['max_tv'] for r in data),
         'rows':data,
         'limitations':['NOT actual poker rewards or exploitability',
                        'synthetic independent seeded style actions only',
                        'static uniform baseline is simple and not GTO',
                        'expert proposals are fixed diagnostic toys, not poker counter-strategies',
                        'no opponent private information is collected',
                        'confidence interval assumes scenario seed blocks independent']}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    if args.checkpoint is not None:save_registry(args.checkpoint,last)
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2,ensure_ascii=False))
    return out


def main():run(parser().parse_args())
if __name__=='__main__':main()
