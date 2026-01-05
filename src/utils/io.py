from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
