from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import Instance, Solution


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def make_pair_row(inst: Instance, sol: Solution, meta: dict | None = None) -> dict:
    row = {
        "instance": inst.to_dict(),
        "solution": sol.to_dict(),
    }
    if meta is not None:
        row["meta"] = meta
    return row
