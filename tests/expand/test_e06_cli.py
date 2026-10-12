import argparse,json
import pytest
from scripts.expand_phase06_benchmark import run


def test_synthetic_blocked_benchmark(tmp_path):
    args=argparse.Namespace(blocks=4,warmup=80,post_shift=80,initial_style='tight',new_style='aggressive',seed=71006,
                            output_json=str(tmp_path/'out.json'),checkpoint=str(tmp_path/'stats.json'))
    out=run(args)
    assert out['blocks']==4
    assert len(out['rows'])==4
    assert out['max_observed_kl']<=.025+1e-9
    assert out['max_observed_tv']<=.075+1e-9
    assert (tmp_path/'stats.json').is_file()
    assert out['shift_detected_blocks']>=1
    with pytest.raises(FileExistsError):run(args)


def test_determinism(tmp_path):
    args=argparse.Namespace(blocks=3,warmup=50,post_shift=50,initial_style='tight',new_style='aggressive',seed=71006,
                            output_json=str(tmp_path/'out.json'),checkpoint=None)
    a=run(args)
    args.output_json=str(tmp_path/'again.json')
    b=run(args)
    assert a==b
