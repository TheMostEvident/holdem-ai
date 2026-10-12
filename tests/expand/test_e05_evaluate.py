from argparse import Namespace
import random
import pytest
from holdem_ai.expand.pluribus.blueprint import KuhnGame,MCCFRSolver,MCCFRConfig,save_checkpoint
from scripts.expand_phase05_evaluate import run

def test_evaluation_not_modifying_checkpoint(tmp_path):
    path=tmp_path/'blueprint.json.gz'
    solver=MCCFRSolver(KuhnGame(2),MCCFRConfig(seed=1))
    solver.train(50)
    save_checkpoint(path,solver)
    original=path.read_bytes()
    args=Namespace(game='kuhn',checkpoint=str(path),hands=3,seed=111,iterations=4,
                   time_ms=300,nodes=500,output_json=str(tmp_path/'out.json'))
    out=run(args)
    assert out['hands']==6
    assert out['search_calls']>0
    assert len(out['ci95_normal'])==2
    assert path.read_bytes()==original
