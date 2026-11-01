"""Deterministic external tool drivers.

Tool output is merged into findings directly. Missing tools are skipped and
recorded as notes; they never fail the run.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .models import Finding

_SEVERITY_BY_PREFIX = {"F": "P2", "E": "P3", "W": "P3"}
_CATEGORY_BY_PREFIX = {"F": "correctness", "E": "style", "W": "style"}

_TEXT_LINE_RE = re.compile(r"^(.+?):(\d+):?(\d+)?:\s*([A-Z]+\d*)\s+(.*)$")


def _run(cmd: list[str], cwd: Path, timeout: int = 120) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"tool not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def _ruff_findings(rel_file: str, entry: dict) -> Finding:
    code = entry.get("code", "")
    prefix = code[0] if code else "E"
    return Finding(
        source="ruff",
        category=_CATEGORY_BY_PREFIX.get(prefix, "style"),
        severity=_SEVERITY_BY_PREFIX.get(prefix, "P3"),
        file=rel_file,
        line=entry.get("location", {}).get("row"),
        title=entry.get("message", code),
        evidence=f"ruff {code}",
        code=code,
    )


def run_ruff(files: list[Path], cwd: Path, rel_files: list[str]) -> tuple[list[Finding], str | None]:
    if shutil.which("ruff") is None:
        return [], None
    cmd = ["ruff", "check", "--no-cache", "--output-format", "json", *[str(f) for f in files]]
    rc, out, err = _run(cmd, cwd)
    if rc == 127:
        return [], None
    raw = out if out else err
    findings: list[Finding] = []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return [], raw
    rel_map = {str(f.resolve()): r for f, r in zip(files, rel_files)}
    for entry in data:
        fname = entry.get("filename", "")
        rel = rel_map.get(str((cwd / fname).resolve()))
        if rel is None:
            try:
                rel = str(Path(fname).resolve().relative_to(cwd.resolve()))
            except ValueError:
                rel = fname
        findings.append(_ruff_findings(rel, entry))
    return findings, raw


def run_torchfix(files: list[Path], cwd: Path, rel_files: list[str]) -> tuple[list[Finding], str | None]:
    if shutil.which("torchfix") is None:
        return [], None
    findings: list[Finding] = []
    raw_parts: list[str] = []
    for f in files:
        rc, out, err = _run(["torchfix", str(f)], cwd)
        if rc == 127:
            return [], None
        raw_parts.append(out if out else err)
        for line in (out + err).splitlines():
            m = _TEXT_LINE_RE.match(line.strip())
            if not m:
                continue
            path_part, line_no, _, code, message = m.groups()
            rel = str(f.relative_to(cwd)) if str(f.relative_to(cwd)) == path_part else path_part
            findings.append(
                Finding(
                    source="torchfix",
                    category="correctness",
                    severity="P2",
                    file=rel,
                    line=int(line_no),
                    title=message,
                    evidence=f"torchfix {code}",
                    code=code,
                )
            )
    return findings, "\n".join(raw_parts) if raw_parts else None


def run_tools(py_files: list[Path], cwd: Path, rel_files: list[str], raw_dir: Path | None = None) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    notes: list[str] = []
    for name, runner in (("ruff", run_ruff), ("torchfix", run_torchfix)):
        if shutil.which(name) is None:
            notes.append(f"{name}: not installed, skipped")
            continue
        fs, raw = runner(py_files, cwd, rel_files)
        findings.extend(fs)
        if raw is not None and raw_dir is not None:
            (raw_dir / f"{name}.txt").write_text(raw, encoding="utf-8")
    return findings, notes
