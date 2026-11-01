"""Create per-run output directories and write meta information."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

DEFAULT_ROOT = Path(".mreview-results")


def create_run_dir(command: str, args: list[str], target: str, mode: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = DEFAULT_ROOT / stamp
    root.mkdir(parents=True, exist_ok=False)
    (root / "tools").mkdir()
    (root / "weights").mkdir()
    meta = {
        "command": command,
        "args": args,
        "target": str(target),
        "mode": mode,
        "created_utc": stamp,
        "python": platform.python_version(),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
