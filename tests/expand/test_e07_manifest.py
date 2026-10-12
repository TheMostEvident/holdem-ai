import json
import pytest
from holdem_ai.expand.integration import release_manifest,write_new_json,sha256_file


def test_e07_manifest_checkpoint_hash_and_no_overwrite(tmp_path):
    src=tmp_path/'test.pt'
    src.write_bytes(b'checkpoint-v1')
    script=tmp_path/'script.py'
    script.write_text('print(1)')
    metadata=release_manifest(project_root=tmp_path,config={'a':1},seeds=[7],
             checkpoints={'ppo':str(src)},abstraction_id='test',scripts=['script.py'])
    assert metadata['checkpoints']['ppo']['sha256']==sha256_file(src)
    assert metadata['evaluation_script_sha256']['script.py']==sha256_file(script)
    output=tmp_path/'result.json'
    write_new_json(output,metadata)
    with pytest.raises(FileExistsError):write_new_json(output,metadata)
    assert json.loads(output.read_text())['format']=='expand-e07-manifest-v1'
