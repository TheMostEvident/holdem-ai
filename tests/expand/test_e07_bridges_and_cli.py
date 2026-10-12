import argparse
import json
from types import SimpleNamespace
import pytest

from holdem_ai.expand.integration import from_e05_solver,from_e04_checkpoint
from test_e07_agent import state
from scripts.expand_phase07_benchmark import run,parser
from e03_fake_engine import FastFakeEngine


def test_e07_search_source_rejects_timeout_without_mask_override():
    obs=state()
    class Solver:
        def search(self,root):
            return SimpleNamespace(status='timeout',action_probabilities={'fold':.5,'check':.5})
    source=from_e05_solver(Solver(),verified_root_belief=lambda state:'belief')
    with pytest.raises(TimeoutError):source(obs)


def test_e07_search_source_rejects_nonmatching_action_support():
    class Solver:
        def search(self,root):
            return SimpleNamespace(status='solved',action_probabilities={'call':1.})
    source=from_e05_solver(Solver(),verified_root_belief=lambda state:'belief')
    with pytest.raises(ValueError):source(state())


def test_e07_missing_checkpoint_refuses_fake_blueprint(tmp_path):
    with pytest.raises(FileNotFoundError):from_e04_checkpoint(str(tmp_path/'missing.json.gz'),exact_infoset_key=lambda s:'x')


def test_e07_cli_validation_adaptive_requires_expert(tmp_path):
    args=parser().parse_args(['--groups','2','--players','2','--adaptive',
                              '--output-json',str(tmp_path/'out.json')])
    with pytest.raises(ValueError,match='requires|needs'):run(args)


def test_e07_cli_completes_fake_cash_with_manifest(tmp_path,monkeypatch):
    import scripts.expand_phase07_benchmark as cli
    old_cash=cli.cash_benchmark
    def fake_cash(agent,cfg):return old_cash(agent,cfg,engine_factory=FastFakeEngine)
    monkeypatch.setattr(cli,'cash_benchmark',fake_cash)
    output=tmp_path/'bench.json'
    args=parser().parse_args(['--groups','2','--players','2','--stacks-bb','20',
        '--opponents','calling_station','--seed','42','--output-json',str(output)])
    report=run(args)
    assert output.exists()
    assert report['cash']['scored_hands']==2
    assert json.loads(output.read_text())['release_manifest']['format']=='expand-e07-manifest-v1'
    with pytest.raises(FileExistsError):run(args)
