"""Small speed-test smoke; does not assert hardware-dependent throughput."""
import importlib.util
from pathlib import Path


def test_cpu_benchmark_small_run():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'phase6_benchmark.py'
    spec = importlib.util.spec_from_file_location('phase6_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    results = module.run_benchmark(4, 1, seed=37)
    assert results['hands'] == 4
    assert results['workers'] == 1
    assert results['decisions'] > 0
    assert results['hands_per_second'] > 0
