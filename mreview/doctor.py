"""Environment self-check: confirm the review tools exist. Nothing more."""

from __future__ import annotations

import importlib
import json
import shutil
from typing import Any

REQUIRED_TOOLS = ["torch", "safetensors", "ruff"]
OPTIONAL_TOOLS = ["torchfix", "hotpath", "torch-preflight"]


def _tool_status(name: str) -> tuple[bool, str]:
    if name in {"torch", "safetensors"}:
        try:
            mod = importlib.import_module(name)
            return True, getattr(mod, "__version__", "installed")
        except ImportError:
            return False, "not installed"
    exe = shutil.which(name)
    if exe is None:
        return False, "not found on PATH"
    return True, exe


def run_doctor() -> dict[str, Any]:
    results: dict[str, Any] = {"required": {}, "optional": {}}
    ok = True
    for name in REQUIRED_TOOLS:
        present, info = _tool_status(name)
        results["required"][name] = {"present": present, "info": info}
        if name in {"torch", "ruff"} and not present:
            ok = False
    for name in OPTIONAL_TOOLS:
        present, info = _tool_status(name)
        results["optional"][name] = {"present": present, "info": info}
    results["ok"] = ok
    return results


def print_doctor() -> int:
    data = run_doctor()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0 if data["ok"] else 1
