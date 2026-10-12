"""Reproducible experiment metadata; no checkpoint modification."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Mapping


def sha256_file(path: str|Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def release_manifest(*, project_root: str|Path, config: Mapping,
                     seeds: list[int], checkpoints: Mapping[str,str] | None = None,
                     abstraction_id: str='none', scripts: list[str] | None = None) -> dict:
    root=Path(project_root)
    paths={}
    for name,path in sorted((checkpoints or {}).items()):
        p=Path(path).resolve(strict=True)
        paths[name]={'absolute_path':str(p),'sha256':sha256_file(p)}
    files={}
    for name in scripts or []:
        p=(root/name).resolve(strict=True)
        if not p.is_relative_to(root.resolve()):
            raise ValueError('script must be beneath project root')
        files[name]=sha256_file(p)
    try:
        res=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,timeout=5,check=True)
        commit=res.stdout.strip()
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        commit=None
    return {'format':'expand-e07-manifest-v1','git_commit':commit,
            'checkpoints':paths,'config':dict(config),'seeds':list(seeds),
            'abstraction_id':abstraction_id,'evaluation_script_sha256':files}


def write_new_json(path: str|Path, record: Mapping) -> None:
    """Refuse overwrite using exclusive creation; interruption may leave partial output."""
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with dest.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
