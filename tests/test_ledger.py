
from mreview.ledger import verify_ledger
from mreview.questionnaire import questions
from mreview.splitter import split_file

SRC = (
    "import torch\n"
    "\n"
    "def foo(a):\n"
    "    return a + 1\n"
    "\n"
    "class C:\n"
    "    def m(self):\n"
    "        return 1\n"
    "\n"
)


def _units(tmp_path):
    p = tmp_path / "mod.py"
    p.write_text(SRC, encoding="utf-8")
    return split_file(p, "mod.py")


def _full_ledger(units, evidence=None):
    return {
        "units": [
            {
                "unit_id": u.unit_id,
                "answers": [
                    {
                        "question_id": q.id,
                        "verdict": "ok",
                        "evidence": evidence or f"L{u.start_line}-L{u.end_line}",
                    }
                    for q in questions()
                ],
            }
            for u in units
        ]
    }


def test_good_ledger_passes(tmp_path):
    units = _units(tmp_path)
    gate = verify_ledger(_full_ledger(units), units)
    assert gate["checks_satisfied"] == gate["checks_required"]
    assert not gate["unverified_units"]


def test_missing_answer_flagged(tmp_path):
    units = _units(tmp_path)
    ledger = _full_ledger(units)
    ledger["units"][0]["answers"] = ledger["units"][0]["answers"][1:]
    gate = verify_ledger(ledger, units)
    assert len(gate["violations"]) == 1
    assert gate["violations"][0]["question_id"] == "q1"
    assert gate["violations"][0]["reason"] == "missing answer"


def test_ok_without_evidence_flagged(tmp_path):
    units = _units(tmp_path)
    ledger = _full_ledger(units)
    ledger["units"][0]["answers"][0] = {"question_id": "q1", "verdict": "ok", "evidence": ""}
    gate = verify_ledger(ledger, units)
    assert any(v["question_id"] == "q1" for v in gate["violations"])


def test_evidence_outside_unit_flagged(tmp_path):
    units = _units(tmp_path)
    ledger = _full_ledger(units, evidence="L9999")
    gate = verify_ledger(ledger, units)
    assert gate["checks_satisfied"] == 0
