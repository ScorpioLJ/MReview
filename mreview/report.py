"""Findings merge, dedup, and report generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .models import Finding


def dedup(findings: list[Finding]) -> list[Finding]:
    merged: dict[tuple, Finding] = {}
    for f in findings:
        key = (f.source, f.file, f.line, f.code or f.title)
        if key not in merged:
            merged[key] = f
        elif not merged[key].evidence and f.evidence:
            merged[key].evidence = f.evidence
    return list(merged.values())


def write_findings(findings: list[Finding], out_dir: Path) -> None:
    data = {
        "count": len(findings),
        "findings": [_finding_dict(f, i) for i, f in enumerate(findings, 1)],
    }
    (out_dir / "findings.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _finding_dict(f: Finding, idx: int) -> dict:
    d = f.to_dict()
    d["id"] = f"F{idx:03d}"
    return d


def write_report(
    out_dir: Path,
    *,
    target: str,
    mode: str,
    notes: list[str],
    findings: list[Finding],
    units_count: int = 0,
    gate: dict | None = None,
    doctor: dict | None = None,
) -> None:
    lines: list[str] = ["# MReview Report", ""]
    lines.append(f"- target: `{target}`")
    lines.append(f"- mode: `{mode}`")
    lines.append(f"- units: {units_count}")
    if doctor:
        missing = [n for n, v in doctor.get("required", {}).items() if not v.get("present")]
        lines.append(f"- doctor: {'OK' if not missing else 'missing: ' + ', '.join(missing)}")
    lines.append("")

    if notes:
        lines.append("## 附注（环境 / 跳过项）")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    ordered = sorted(findings, key=lambda f: (f.severity, f.category))
    by_sev = Counter(f.severity for f in findings)
    by_cat = Counter(f.category for f in findings)
    lines.append("## 统计")
    lines.append(f"- findings: {len(findings)} | by severity: {dict(by_sev)} | by category: {dict(by_cat)}")
    lines.append("")

    if gate is not None:
        lines.append("## 覆盖校验（verify）")
        lines.append(
            f"- checks: {gate['checks_satisfied']} / {gate['checks_required']} satisfied; "
            f"{gate['checks_completed']} completed"
        )
        lines.append(f"- unverified units: {len(gate['unverified_units'])}")
        for u in gate["unverified_units"]:
            lines.append(f"  - {u}")
        for v in gate.get("violations", [])[:50]:
            lines.append(f"  - gap: `{v['unit_id']}` q{v['question_id']}: {v['reason']}")
        if gate.get("stale_units"):
            lines.append(f"- stale ledger units (not in source): {gate['stale_units']}")
        lines.append("")

    lines.append("## Findings")
    if not ordered:
        lines.append("No findings.")
    for f in ordered:
        loc = f"{f.file}:{f.line}" if f.line else f.file
        lines.append(f"- **[{f.severity}] {f.title}** — `{loc}` ({f.source})")
        if f.cause:
            lines.append(f"  - 原因: {f.cause}")
        if f.fix:
            lines.append(f"  - 方案: {f.fix}")
        if f.evidence:
            lines.append(f"  - 证据: {f.evidence}")
    lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
