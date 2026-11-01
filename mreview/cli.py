"""mreview CLI: doctor / split / review / verify."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import outdir
from .doctor import print_doctor, run_doctor
from .ledger import verify_from_files
from .pipeline import run_review, write_units_and_worksheet
from .report import dedup, write_findings, write_report
from .splitter import split_target


def _cmd_split(args: argparse.Namespace) -> int:
    out_dir = outdir.create_run_dir("split", [args.target], args.target, "static")
    units = split_target(Path(args.target))
    write_units_and_worksheet(out_dir, units)
    print(f"units={len(units)} out={out_dir}")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    out_dir = run_review(Path(args.target), dynamic=args.dynamic)
    print(f"done out={out_dir}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    out_dir = outdir.create_run_dir("verify", [args.ledger, args.target], args.target, "static")
    gate, units = verify_from_files(Path(args.ledger), Path(args.target))
    outdir.write_json(out_dir / "ledger-gate.json", gate)
    findings = dedup(gate["agent_findings"])
    write_findings(findings, out_dir)
    write_report(
        out_dir,
        target=str(args.target),
        mode="verify",
        notes=[],
        findings=findings,
        units_count=len(units),
        gate=gate,
        doctor=run_doctor(),
    )
    ok = not gate["unverified_units"]
    print(f"verified={ok} checks={gate['checks_satisfied']}/{gate['checks_required']} unverified={len(gate['unverified_units'])} out={out_dir}")
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mreview", description="Model design / training review CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="确认审查工具存在")

    p_split = sub.add_parser("split", help="切分单元并生成 worksheet")
    p_split.add_argument("target")

    p_review = sub.add_parser("review", help="确定性审查：split + 外部工具 + 权重检查")
    p_review.add_argument("target")
    p_review.add_argument("--dynamic", action="store_true", help="动态检查：加载模型并跑一次前向")

    p_verify = sub.add_parser("verify", help="校验 agent 回填的账本")
    p_verify.add_argument("ledger")
    p_verify.add_argument("target")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return print_doctor()
    if args.command == "split":
        return _cmd_split(args)
    if args.command == "review":
        return _cmd_review(args)
    if args.command == "verify":
        return _cmd_verify(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
