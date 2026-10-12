"""Append-only deterministic journal; privileged seed log, not an actor observation."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any


class EventLedger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self.records: list[dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                raise FileExistsError('journal exists; use a fresh journal path')
            # File created at construction, never clobber an existing journal.
            self.path.open('x', encoding='utf-8').close()

    def append(self, event: str, **data: Any) -> dict[str, Any]:
        record = {'seq': len(self.records) + 1, 'event': event, **data}
        # Reject nonportable/nonserializable records up front.
        wire = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if self.path is not None:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(wire + '\n')
                f.flush()
                os.fsync(f.fileno())
        self.records.append(record)
        return record

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open(encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if any(row.get('seq') != i for i, row in enumerate(rows, 1)):
            raise ValueError('ledger sequence is discontinuous')
        return rows
