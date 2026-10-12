import pytest
from scripts.expand_phase03_train import build_parser,make_config


def test_e03_cli_has_safe_defaults_and_restricted_resume():
    p=build_parser()
    default=make_config(p.parse_args([]))
    assert default.rollout.num_players==(2,3,4,5,6)
    assert default.rollout.starting_stack_bb==100
    assert make_config(p.parse_args(['--resume'])) is None
    with pytest.raises(ValueError,match='resume loads immutable'):
        make_config(p.parse_args(['--resume','--hands-per-update','8']))


def test_e03_custom_curriculum():
    p=build_parser()
    cfg=make_config(p.parse_args(['--hands-per-update','4','--num-players','3,6',
      '--random-stack-range-bb','30,200','--shared-fraction','0.4','--snapshot-fraction','0.2']))
    assert cfg.rollout.hands_per_update==4
    assert cfg.rollout.num_players==(3,6)
    assert cfg.rollout.random_stack_range_bb==(30,200)
