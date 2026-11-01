"""Tests for tools/version_scope.py."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("version_scope", ROOT / "tools" / "version_scope.py")
VERSION_SCOPE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERSION_SCOPE
SPEC.loader.exec_module(VERSION_SCOPE)

FIXTURE_DIR = ROOT / "fixtures" / "version_scope"


def test_load_config_values_finds_strings_by_value_not_key():
    values = VERSION_SCOPE.load_config_values(FIXTURE_DIR / "sample.yaml")
    found = {(v.key_path, v.value) for v in values}
    assert ("version", "ldm-ispdiffusion-v10.2-s1-bm256-finetune") in found
    assert ("diffusion_version", "v10.2-s1") in found
    assert ("mode", "train") in found


def test_scan_conditions_and_evaluation():
    conditions, unrecognized = VERSION_SCOPE._scan_file(FIXTURE_DIR / "sample_code.py")
    by_line = {c.lineno: c for c in conditions}
    value = "ldm-ispdiffusion-v10.2-s1-bm256-finetune"
    hit_lines = [line for line, c in by_line.items() if VERSION_SCOPE.evaluate_condition(c, value)]
    assert hit_lines == [3, 5, 7]
    assert "a" not in {c.literal for c in conditions}
    assert any(u.lineno == 13 for u in unrecognized)


def test_eq_matches_config_mode_value():
    conditions, _ = VERSION_SCOPE._scan_file(FIXTURE_DIR / "sample_code.py")
    by_line = {c.lineno: c for c in conditions}
    assert VERSION_SCOPE.evaluate_condition(by_line[11], "train")
    assert not VERSION_SCOPE.evaluate_condition(by_line[11], "infer")


def test_map_line_to_unit():
    units = {
        "units": [
            {"unit_id": "sample_code.py:Model.run", "file": "sample_code.py", "start_line": 2, "end_line": 16}
        ]
    }
    by_file = {}
    by_file["sample_code.py"] = units["units"]
    mapped = VERSION_SCOPE.map_line(by_file, "sample_code.py", 5)
    assert mapped == "sample_code.py:Model.run (2-16)"
    assert VERSION_SCOPE.map_line(by_file, "sample_code.py", 99) is None


def test_build_report_batch(tmp_path):
    config_dir = tmp_path / "configs"
    code_dir = tmp_path / "code"
    config_dir.mkdir()
    code_dir.mkdir()
    (config_dir / "a.yaml").write_text("version: v10.2-s1unet\n", encoding="utf-8")
    (config_dir / "b.yaml").write_text("version: v3.8\n", encoding="utf-8")
    (code_dir / "m.py").write_text(
        "if 'v10.' in v:\n    pass\nif 'v1' in v:\n    pass\n", encoding="utf-8"
    )
    report = VERSION_SCOPE.build_report(config_dir, code_dir, None)
    a_hits = report["matches"][0]["hits"]
    b_hits = report["matches"][1]["hits"]
    assert [h["line"] for h in a_hits] == [1, 3]
    assert [h["line"] for h in b_hits] == []
