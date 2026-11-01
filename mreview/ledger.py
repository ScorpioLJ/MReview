"""Ledger validation: the coverage gate.

A ledger row is (unit, question, verdict, evidence). `verify` re-splits the
target, then checks every unit/question is answered, every evidence line falls
inside the unit, and "ok" verdicts still carry line-number evidence. Units
with any violation are marked "unverified" and the exact gaps are reported.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Finding, Unit
from .questionnaire import questions
from .splitter import split_target

VERDICTS = {"ok", "issue", "skip"}
_NUM_RE = re.compile(r"\d+")


def _parse_evidence(evidence: str) -> list[int]:
    return [int(n) for n in _NUM_RE.findall(evidence or "")]


def load_ledger(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_ledger(ledger: dict, units: list[Unit]) -> dict:
    unit_map = {u.unit_id: u for u in units}
    answers_by_unit: dict[str, dict[str, dict]] = {}
    for entry in ledger.get("units", []):
        answers_by_unit[entry.get("unit_id", "")] = {
            a.get("question_id", ""): a for a in entry.get("answers", [])
        }

    q_ids = [q.id for q in questions()]
    violations: list[dict] = []
    agent_findings: list[Finding] = []

    for unit in units:
        answers = answers_by_unit.get(unit.unit_id, {})
        for qid in q_ids:
            a = answers.get(qid)
            reason = None
            if a is None:
                reason = "missing answer"
            else:
                verdict = a.get("verdict", "")
                if verdict not in VERDICTS:
                    reason = f"bad verdict: {verdict}"
                else:
                    nums = _parse_evidence(a.get("evidence", ""))
                    if not nums:
                        reason = "no line-number evidence"
                    elif not any(unit.start_line <= n <= unit.end_line for n in nums):
                        reason = "evidence line outside unit"
                    elif verdict == "ok" and not nums:
                        reason = "ok without line-number evidence"
            if reason:
                violations.append({"unit_id": unit.unit_id, "question_id": qid, "reason": reason})
            elif a and a.get("verdict") == "issue":
                agent_findings.append(
                    Finding(
                        source="agent",
                        category=a.get("category", "design"),
                        severity=a.get("severity", "P2"),
                        file=unit.file,
                        line=int(re.search(r"\d+", a.get("evidence", "")).group()) if re.search(r"\d+", a.get("evidence", "")) else None,
                        title=a.get("title", "agent finding"),
                        cause=a.get("cause", ""),
                        fix=a.get("fix", ""),
                        evidence=a.get("evidence", ""),
                    )
                )

    unverified = sorted({v["unit_id"] for v in violations})
    checks_required = len(units) * len(q_ids)
    checks_satisfied = checks_required - len(violations)
    checks_completed = len(answers_by_unit) * len(q_ids) if answers_by_unit else 0

    return {
        "checks_required": checks_required,
        "checks_completed": checks_completed,
        "checks_satisfied": checks_satisfied,
        "unverified_units": unverified,
        "violations": violations,
        "agent_findings": agent_findings,
        "ledger_unit_ids": sorted(answers_by_unit),
        "stale_units": sorted(set(answers_by_unit) - set(unit_map)),
    }


def verify_from_files(ledger_path: Path, target: Path) -> tuple[dict, list[Unit]]:
    units = split_target(target)
    ledger = load_ledger(ledger_path)
    return verify_ledger(ledger, units), units
