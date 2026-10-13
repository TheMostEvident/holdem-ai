# Complete Python source volume 2/28

Verbatim code from commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`, part of [integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Fenced source content is exact; original SHA provided for verification.

## PY-013 — `scripts/expand_phase02_smoke.py`

Original Git blob `f688e356adf933519d8000191757cae61adfff99`, 4124 bytes.

<a name="py-013"></a>

<!-- BEGIN_PY_SOURCE path="scripts/expand_phase02_smoke.py" sha="f688e356adf933519d8000191757cae61adfff99" -->

````````python
"""Expand E2: real-engine 2..6 person acting observation & action smoke.

This script does NOT train or certify a 6-max poker player. It only checks
that existing Phase 3 rules and E1/E2 actor observations cooperate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from holdem_ai.expand.environment import DynamicTable
from holdem_ai.expand.information import Phase2TableAdapter


def run(*, hands_per_n: int, seed: int, output_json: str) -> dict:
    if type(hands_per_n) is not int or hands_per_n < 1 or type(seed) is not int or seed < 0:
        raise ValueError('positive hands-per-n and nonnegative seed required')
    from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine  # check dependency
    assert MultiPlayerHandEngine is not None
    results = []
    for target_n in range(2,7):
        table = DynamicTable(session_id=f'e02-smoke-n{target_n}')
        env = Phase2TableAdapter(table)
        for i in range(target_n):
            table.request_join(f'p{i}', 10000, preferred_seat=i)
        counted, setup_hands, actions, max_hist, missing = 0, 0, 0, 0, 0
        attempts = 0
        while counted < hands_per_n:
            if attempts > hands_per_n + 100:
                raise RuntimeError(f'failed to admit {target_n} players')
            hand = env.start_hand(seed + target_n*100000 + attempts)
            attempts += 1
            if hand is None:
                raise RuntimeError('not enough players for table')
            live_n = len(hand.participants)
            if live_n != target_n:
                setup_hands += 1
            steps = 0
            while table.current_player_id is not None:
                if steps >= 500:
                    raise RuntimeError('hand exceeded 500 actions')
                state = env.observe()
                x = env.encode()
                assert state.num_dealt_in == live_n
                assert state.hero_seat in dict(hand.participants)
                assert len(x.legal_mask) == 7 and len(x.seat_mask) == 6
                assert sum(x.seat_mask) == target_n
                assert len(x.global_features) > 100
                assert sum(x.legal_mask) >= 1
                max_hist = max(max_hist, len(state.action_history))
                chosen = 'call' if state.to_call_bb > 0 else 'check'
                if not state.legal_actions.allows(chosen):
                    missing += 1
                    raise AssertionError('engine rejects legal check/call')
                env.act_abstract(chosen)
                steps += 1
                actions += 1
            result = env.settle_hand()
            assert sum(result.payoffs_chips) == 0
            assert sum(result.final_stacks) == sum(result.starting_stacks)
            if live_n == target_n:
                counted += 1
        results.append({'players': target_n, 'hands': counted, 'entry_setup_hands': setup_hands,
                        'actor_decisions_including_setup': actions, 'max_history': max_hist,
                        'illegal_or_missing_check_call': missing})
    report = {'phase': 'Expand Phase 2', 'kind':'real_engine_info_state_smoke',
              'total_target_hands': sum(x['hands'] for x in results),
              'total_setup_hands': sum(x['entry_setup_hands'] for x in results),
              'training_updates': 0,'seed':seed,'results':results}
    dest=Path(output_json)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x',encoding='utf-8') as fh:
        json.dump(report,fh,ensure_ascii=False,indent=2)
        fh.write('\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description='Expand Phase 2 2..6 player info-state smoke; no learning')
    p.add_argument('--hands-per-n',type=int,default=10)
    p.add_argument('--seed',type=int,default=180201)
    p.add_argument('--output-json',default='runs/expand_phase02/smoke_10.json')
    args=p.parse_args()
    run(hands_per_n=args.hands_per_n,seed=args.seed,output_json=args.output_json)


if __name__=='__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/expand_phase02_smoke.py" -->

## PY-014 — `scripts/expand_phase03_evaluate.py`

Original Git blob `faec9d885323bfd171b50a01f15be468999272cd`, 1497 bytes.

<a name="py-014"></a>

<!-- BEGIN_PY_SOURCE path="scripts/expand_phase03_evaluate.py" sha="faec9d885323bfd171b50a01f15be468999272cd" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/expand_phase03_evaluate.py" -->

## PY-015 — `scripts/expand_phase03_train.py`

Original Git blob `431bc859a0904230b4e443b291b811b9305285eb`, 4063 bytes.

<a name="py-015"></a>

<!-- BEGIN_PY_SOURCE path="scripts/expand_phase03_train.py" sha="431bc859a0904230b4e443b291b811b9305285eb" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/expand_phase03_train.py" -->

## PY-016 — `scripts/expand_phase04_evaluate.py`

Original Git blob `5b5354690b37121c1a4d76c5879aeb1f836e84b4`, 1280 bytes.

<a name="py-016"></a>

<!-- BEGIN_PY_SOURCE path="scripts/expand_phase04_evaluate.py" sha="5b5354690b37121c1a4d76c5879aeb1f836e84b4" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/expand_phase04_evaluate.py" -->

## PY-017 — `scripts/expand_phase04_mccfr.py`

Original Git blob `a95f2d2b51fc74fab52472a385f11b145b7a8054`, 5062 bytes.

<a name="py-017"></a>

<!-- BEGIN_PY_SOURCE path="scripts/expand_phase04_mccfr.py" sha="a95f2d2b51fc74fab52472a385f11b145b7a8054" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/expand_phase04_mccfr.py" -->

## PY-018 — `scripts/expand_phase05_evaluate.py`

Original Git blob `e0e25f10bf9f204cf0b2211ba8395147d9410eef`, 4121 bytes.

<a name="py-018"></a>

<!-- BEGIN_PY_SOURCE path="scripts/expand_phase05_evaluate.py" sha="e0e25f10bf9f204cf0b2211ba8395147d9410eef" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/expand_phase05_evaluate.py" -->

## PY-019 — `scripts/expand_phase05_search.py`

Original Git blob `3b8b3f9924ac01c42fbefc595fe8609567e331c5`, 3918 bytes.

<a name="py-019"></a>

<!-- BEGIN_PY_SOURCE path="scripts/expand_phase05_search.py" sha="3b8b3f9924ac01c42fbefc595fe8609567e331c5" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/expand_phase05_search.py" -->

## PY-020 — `scripts/expand_phase06_benchmark.py`

Original Git blob `c8c542d15b20b4c022ca10ddebd6f6ad10f12764`, 6571 bytes.

<a name="py-020"></a>

<!-- BEGIN_PY_SOURCE path="scripts/expand_phase06_benchmark.py" sha="c8c542d15b20b4c022ca10ddebd6f6ad10f12764" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/expand_phase06_benchmark.py" -->

## PY-021 — `scripts/expand_phase07_benchmark.py`

Original Git blob `6dd304f84ed92463211fa38ef53dceda6c0bdd2c`, 5102 bytes.

<a name="py-021"></a>

<!-- BEGIN_PY_SOURCE path="scripts/expand_phase07_benchmark.py" sha="6dd304f84ed92463211fa38ef53dceda6c0bdd2c" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/expand_phase07_benchmark.py" -->

## PY-022 — `scripts/gui10_smoke.py`

Original Git blob `05cd34d3902fafa95336ffe078b70e4e74301a0c`, 1820 bytes.

<a name="py-022"></a>

<!-- BEGIN_PY_SOURCE path="scripts/gui10_smoke.py" sha="05cd34d3902fafa95336ffe078b70e4e74301a0c" -->

````````python
"""Transport-free GUI-1.0 HUNL smoke run; human plays check/call by default.

Run from repo root: uv run --extra dev python scripts/gui10_smoke.py --hands 20
"""
from __future__ import annotations

import argparse
import json

from holdem_ai.agents.diagnostic_opponents import CallingStation
from holdem_ai.game_service import HUNLGameSession


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI-1.0 human-agent integration smoke")
    parser.add_argument("--hands", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    if options.hands < 1:
        parser.error("--hands must be positive")
    session = HUNLGameSession(bot_policy=CallingStation(), seed=options.seed)
    for hand in range(options.hands):
        view = session.start_hand()
        for _ in range(300):
            if view["phase"] == "complete":
                break
            legal = {action["label"]: action["index"] for action in view["actions"]}
            action = legal.get("check", legal.get("call"))
            if action is None:
                raise AssertionError("human has neither check nor call")
            view = session.play_human_action(action, expected_revision=view["revision"])
        else:
            raise RuntimeError("hand exceeded action limit")
        if sum(view["hand_result"]["payoffs_chips"]) != 0:
            raise AssertionError("non-zero-sum hand")
        print(f"hand={hand+1:3} button={view['button']} hero_bb={view['hand_result']['payoffs_bb'][0]:+8.2f} board={' '.join(view['board'])}")
    print("PASS:", json.dumps({
        "hands": view["completed_hands"],
        "cumulative_bb": view["cumulative_bb"],
        "bb_per_100": view["bb_per_100"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

````````

<!-- END_PY_SOURCE path="scripts/gui10_smoke.py" -->

## PY-023 — `scripts/gui11_smoke.py`

Original Git blob `d85e3a52faf3baf37588f7b3a6f69fa858bfcb41`, 2129 bytes.

<a name="py-023"></a>

<!-- BEGIN_PY_SOURCE path="scripts/gui11_smoke.py" sha="d85e3a52faf3baf37588f7b3a6f69fa858bfcb41" -->

````````python
"""HTTP smoke exercise without starting an external server.

Run: uv run --extra dev --extra gui python -m scripts.gui11_smoke --hands 10
"""
from __future__ import annotations

import argparse

from fastapi.testclient import TestClient

from backend.main import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="GUI-1.1 local HTTP smoke exercise")
    ap.add_argument("--hands", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.hands < 1:
        ap.error("--hands must be positive")
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/hunl/sessions", json={"bot": "calling_station", "seed": args.seed})
        response.raise_for_status()
        item = response.json()
        sid, state = item["session_id"], item["state"]
        for hand in range(1, args.hands + 1):
            while state["phase"] != "complete":
                assert state["your_turn"]
                assert state["actions"]
                offer = next((a for a in state["actions"] if a["kind"] in ("check", "call")), state["actions"][0])
                res = client.post(
                    f"/api/v1/hunl/sessions/{sid}/actions",
                    json={"action_index": offer["index"], "expected_revision": state["revision"]},
                )
                res.raise_for_status()
                state = res.json()["state"]
            print(f"hand={hand} bb={state['cumulative_bb']:.2f} bb/100={state['bb_per_100']:.2f} "
                  f"showdown={state['hand_result']['showdown']}")
            if hand < args.hands:
                res = client.post(
                    f"/api/v1/hunl/sessions/{sid}/next-hand",
                    json={"expected_revision": state["revision"]},
                )
                res.raise_for_status()
                state = res.json()["state"]
        deleted = client.delete(f"/api/v1/hunl/sessions/{sid}")
        assert deleted.status_code == 204
        print(f"PASS: {args.hands} complete HUNL200 hands over HTTP interface")


if __name__ == "__main__":
    main()

````````

<!-- END_PY_SOURCE path="scripts/gui11_smoke.py" -->

## PY-024 — `scripts/gui14_smoke.py`

Original Git blob `16be506b341abd2ab0d86aab8adede228057b5a7`, 2785 bytes.

<a name="py-024"></a>

<!-- BEGIN_PY_SOURCE path="scripts/gui14_smoke.py" sha="16be506b341abd2ab0d86aab8adede228057b5a7" -->

````````python
"""Smoke a trusted local Phase 15/16 neural checkpoint through GUI FastAPI.

Usage (from repo root):
  uv run --extra gui --extra gpu python -m scripts.gui14_smoke --bot ppo --hands 3

Set HOLDEM_AI_PPO_CHECKPOINT / HOLDEM_AI_QBOOST_CHECKPOINT before starting.
This utility never downloads or auto-discovers a checkpoint.
"""
from __future__ import annotations

import argparse
from fastapi.testclient import TestClient
from backend.main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description='GUI-1.4 trained policy API smoke')
    parser.add_argument('--bot', choices=('ppo','qboost'), required=True)
    parser.add_argument('--hands', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.hands < 1:
        parser.error('--hands must be positive')
    with TestClient(create_app()) as client:
        catalog = client.get('/api/v1/hunl/bots').json()
        selected = next(b for b in catalog['bots'] if b['id'] == args.bot)
        if not selected['available']:
            parser.error(f'{args.bot}: {selected["message"]}')
        response = client.post('/api/v1/hunl/sessions', json={
            'bot': args.bot, 'seed': args.seed, 'big_blind': 100,
            'small_blind': 50, 'stack_bb': 200})
        response.raise_for_status()
        payload = response.json()
        sid, game = payload['session_id'], payload['state']
        for index in range(args.hands):
            for _ in range(200):
                if game['phase'] == 'complete':
                    break
                assert game['your_turn']
                offer = next((o for o in game['actions'] if o['kind'] in ('call','check')), game['actions'][0])
                answer = client.post(f'/api/v1/hunl/sessions/{sid}/actions', json={
                    'action_index': offer['index'], 'expected_revision': game['revision']})
                answer.raise_for_status()
                game = answer.json()['state']
            else:
                raise RuntimeError('max human decisions exceeded')
            result = game['hand_result']
            assert sum(result['payoffs_chips']) == 0
            assert sum(result['final_stacks']) == 40000
            print(f'hand={index + 1} net_bb={result["payoffs_bb"][0]:.2f} '
                  f'cumulative_bb={game["cumulative_bb"]:.2f}')
            if index + 1 < args.hands:
                answer = client.post(f'/api/v1/hunl/sessions/{sid}/next-hand', json={
                    'expected_revision': game['revision']})
                answer.raise_for_status()
                game = answer.json()['state']
        print(f'GUI-1.4 PASS ({args.bot}, {args.hands} hands, mode={catalog["policy_mode"]})')


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/gui14_smoke.py" -->
