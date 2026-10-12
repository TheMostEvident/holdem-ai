import json
import argparse
import pytest
from scripts.expand_phase04_mccfr import parser,run
from scripts.expand_phase04_evaluate import main as evaluate_main


def test_cli_kuhn_smoke_resume(tmp_path):
    root=tmp_path/'first'
    args=parser().parse_args(['--game','kuhn','--iterations','12','--root',str(root),
                              '--eval-hands','8'])
    one=run(args)
    assert one['completed_iterations']==12 and one['evaluation']['hands']==8
    resumed=run(parser().parse_args(['--root',str(root),'--resume','--iterations','8']))
    assert resumed['completed_iterations']==20
    with pytest.raises(FileExistsError):run(args)


def test_cli_leduc_smoke(tmp_path):
    root=tmp_path/'leduc'
    result=run(parser().parse_args(['--game','leduc','--iterations','4','--root',str(root)]))
    assert result['completed_iterations']==4


def test_cli_budget_limit_saves_empty_model(tmp_path):
    result=run(parser().parse_args(['--game','kuhn','--iterations','20','--node-budget','1',
                                    '--root',str(tmp_path/'limited')]))
    assert result['stopped_by_budget'] is not None
    assert result['completed_iterations']==0
