from argparse import Namespace
import json
import pytest
from scripts.expand_phase05_search import run

def test_cli_toy_json_and_no_overwrite(tmp_path):
    path=tmp_path/'out.json'
    args=Namespace(game='kuhn',players=2,checkpoint=None,seed=9,root_seed=10,
                   stack_bb=2,time_ms=1000,nodes=2500,memory_mb=64,iterations=10,depth=2,
                   output_json=str(path))
    result=run(args)
    assert result['iterations']>=1
    assert json.loads(path.read_text())['action_probabilities']==result['action_probabilities']
    with pytest.raises(FileExistsError):run(args)
