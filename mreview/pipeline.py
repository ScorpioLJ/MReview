"""Orchestrate the deterministic review pipeline: split, tools, weights."""

from __future__ import annotations

from pathlib import Path

from . import outdir
from .doctor import run_doctor
from .models import Finding, Unit
from .questionnaire import questions
from .report import dedup, write_findings, write_report
from .splitter import split_target
from .tools import run_tools
from .weights import check_weights_file, discover_weight_files


def write_units_and_worksheet(out_dir: Path, units: list[Unit]) -> None:
    unit_data = [u.to_dict() for u in units]
    outdir.write_json(out_dir / "units.json", {"count": len(units), "units": unit_data})
    worksheet = {
        "instructions": (
            "对每个单元回答全部 10 题；verdict 取值 ok/issue/skip；"
            "evidence 必须给出落在该单元行号范围内的行号（如 'L12-L14'），"
            "verdict=ok 也必须带行号证据。issue 需要附带 title/cause/fix/category/severity。"
        ),
        "questions": [q.to_dict() for q in questions()],
        "units": [
            {
                "unit_id": u.unit_id,
                "file": u.file,
                "start_line": u.start_line,
                "end_line": u.end_line,
                "answers": [{"question_id": q.id, "verdict": "", "evidence": ""} for q in questions()],
            }
            for u in units
        ],
    }
    outdir.write_json(out_dir / "worksheet.json", worksheet)


def run_review(target: Path, dynamic: bool, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or outdir.create_run_dir("review", ["--dynamic" if dynamic else "--static"], target, "dynamic" if dynamic else "static")
    doctor = run_doctor()

    units = split_target(target)
    write_units_and_worksheet(out_dir, units)

    cwd = target if target.is_dir() else target.parent
    py_files = sorted({Path(u.file) for u in units}) if target.is_dir() else [target]
    rel_files = sorted({u.file for u in units})
    tool_findings, tool_notes = run_tools(py_files, cwd, rel_files, out_dir / "tools")

    weight_files = discover_weight_files(target)
    weight_findings: list[Finding] = []
    weight_notes: list[str] = []
    dynamic_statuses: list[str] = []
    for wf in weight_files:
        fs, _details, status = check_weights_file(wf, target, dynamic=dynamic, details_dir=out_dir / "weights")
        weight_findings.extend(fs)
        dynamic_statuses.append(status)
    if not weight_files:
        weight_notes.append("未发现权重文件（.pt/.pth/.ckpt/.safetensors），跳过权重检查")
    if dynamic:
        weight_notes.append("dynamic 状态: " + ", ".join(dynamic_statuses) if dynamic_statuses else "无权重文件")

    findings = dedup(tool_findings + weight_findings)
    write_findings(findings, out_dir)
    write_report(
        out_dir,
        target=str(target),
        mode="dynamic" if dynamic else "static",
        notes=tool_notes + weight_notes,
        findings=findings,
        units_count=len(units),
        doctor=doctor,
    )
    return out_dir
