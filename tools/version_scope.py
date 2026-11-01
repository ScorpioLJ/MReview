"""Map a config's version string(s) to the code lines that execute for it.

Reads every string scalar from YAML config(s) - key names do not matter - then
scans Python sources for string-literal comparisons (`'X' in expr`,
`'X' not in expr`, `expr == 'X'`) and reports each condition that evaluates
TRUE for a config value. Optionally maps hit lines to mreview units with
`--units` (output of `mreview split`).

Usage:
    python tools/version_scope.py <config|configs_dir> <code|code_dir> [--units units.json] [--json out.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

EXCLUDE_DIRS = {".git", "__pycache__", ".mreview-results", ".venv", "venv", "node_modules", ".tox"}
NON_VERSION_EXPRS = (ast.List, ast.Dict, ast.Set, ast.Tuple, ast.Constant)
CONDITION_OPS = {ast.In: "in", ast.NotIn: "not in", ast.Eq: "=="}
UNRECOGNIZED_METHODS = {"startswith", "endswith", "contains", "match", "search", "findall", "fullmatch"}


@dataclass
class ConfigValue:
    config_file: str
    key_path: str
    value: str


@dataclass
class Condition:
    file: str
    lineno: int
    op: str
    literal: str
    expr: str
    text: str


@dataclass
class UnrecognizedCondition:
    file: str
    lineno: int
    text: str


def _walk_yaml(node, key_path: str, out: list[ConfigValue], config_file: str) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{key_path}.{key}" if key_path else str(key)
            _walk_yaml(value, child, out, config_file)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            child = f"{key_path}[{index}]"
            _walk_yaml(value, child, out, config_file)
    elif isinstance(node, str):
        out.append(ConfigValue(config_file=config_file, key_path=key_path, value=node))


def load_config_values(path: Path) -> list[ConfigValue]:
    """Collect every string scalar from a YAML file, regardless of key name."""
    values: list[ConfigValue] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, yaml.YAMLError) as exc:
        print(f"WARN: skip config {path}: {exc}", file=sys.stderr)
        return values
    if isinstance(data, dict):
        _walk_yaml(data, "", values, str(path))
    return values


def _scan_file(path: Path) -> tuple[list[Condition], list[UnrecognizedCondition]]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"WARN: skip source {path}: {exc}", file=sys.stderr)
        return [], []
    lines = source.splitlines()
    conditions: list[Condition] = []
    unrecognized: list[UnrecognizedCondition] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in UNRECOGNIZED_METHODS:
                arg = node.args[0] if node.args else None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    unrecognized.append(
                        UnrecognizedCondition(file=str(path), lineno=node.lineno, text=lines[node.lineno - 1].strip())
                    )
            continue
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        op = node.ops[0]
        if not isinstance(op, tuple(CONDITION_OPS)):
            continue
        comp = node.comparators[0]
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            literal, expr = node.left.value, comp
        elif isinstance(comp, ast.Constant) and isinstance(comp.value, str):
            literal, expr = comp.value, node.left
        else:
            continue
        if isinstance(expr, NON_VERSION_EXPRS):
            continue
        lineno = node.lineno
        text = lines[lineno - 1].strip() if lineno <= len(lines) else ""
        conditions.append(
            Condition(
                file=str(path),
                lineno=lineno,
                op=CONDITION_OPS[type(op)],
                literal=literal,
                expr=ast.unparse(expr),
                text=text,
            )
        )
    conditions = _dedupe_conditions(conditions)
    return conditions, unrecognized


def _dedupe_conditions(conditions: list[Condition]) -> list[Condition]:
    seen: set[tuple[str, int, str, str]] = set()
    deduped: list[Condition] = []
    for condition in conditions:
        key = (condition.file, condition.lineno, condition.op, condition.literal)
        if key not in seen:
            seen.add(key)
            deduped.append(condition)
    return deduped


def _iter_py_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*.py")
        if not any(part in EXCLUDE_DIRS for part in candidate.parts)
    )


def _iter_config_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}
    )


def evaluate_condition(condition: Condition, value: str) -> bool:
    if condition.op == "in":
        return condition.literal in value
    if condition.op == "not in":
        return condition.literal not in value
    return condition.literal == value


def load_units(path: Path) -> dict[str, list[dict]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"WARN: skip units {path}: {exc}", file=sys.stderr)
        return {}
    by_file: dict[str, list[dict]] = {}
    for unit in data.get("units", []):
        by_file.setdefault(Path(unit.get("file", "")).name, []).append(unit)
    return by_file


def map_line(units_by_file: dict[str, list[dict]], file_name: str, lineno: int) -> str | None:
    for unit in units_by_file.get(file_name, []):
        if unit.get("start_line", 0) <= lineno <= unit.get("end_line", 0):
            return f"{unit.get('unit_id', '')} ({unit.get('start_line')}-{unit.get('end_line')})"
    return None


def build_report(config_path: Path, code_path: Path, units_path: Path | None) -> dict:
    config_files = _iter_config_files(config_path)
    code_files = _iter_py_files(code_path)
    config_values: list[ConfigValue] = []
    for config_file in config_files:
        config_values.extend(load_config_values(config_file))
    conditions: list[Condition] = []
    unrecognized: list[UnrecognizedCondition] = []
    for code_file in code_files:
        file_conditions, file_unrecognized = _scan_file(code_file)
        conditions.extend(file_conditions)
        unrecognized.extend(file_unrecognized)
    units_by_file = load_units(units_path) if units_path else {}

    matches: list[dict] = []
    for config_value in config_values:
        hits = []
        for condition in conditions:
            if not evaluate_condition(condition, config_value.value):
                continue
            hits.append(
                {
                    "file": condition.file,
                    "line": condition.lineno,
                    "op": condition.op,
                    "literal": condition.literal,
                    "expr": condition.expr,
                    "text": condition.text,
                    "unit": map_line(units_by_file, Path(condition.file).name, condition.lineno),
                }
            )
        matches.append(
            {
                "config_file": config_value.config_file,
                "key_path": config_value.key_path,
                "value": config_value.value,
                "hits": hits,
            }
        )
    return {
        "config_files": [str(p) for p in config_files],
        "code_files": [str(p) for p in code_files],
        "matches": matches,
        "unrecognized": [
            {"file": item.file, "line": item.lineno, "text": item.text} for item in unrecognized
        ],
    }


def print_report(report: dict) -> None:
    total_hits = 0
    for match in report["matches"]:
        if not match["hits"]:
            continue
        total_hits += len(match["hits"])
        print(f"=== config: {match['config_file']}")
        print(f"value \"{match['value']}\" [{match['key_path']}] -> {len(match['hits'])} hit(s)")
        for hit in match["hits"]:
            unit = f" -> {hit['unit']}" if hit.get("unit") else ""
            print(
                f"  {Path(hit['file']).name}:{hit['line']}  "
                f"{hit['op']} '{hit['literal']}'  ({hit['text']}){unit}"
            )
        print()
    for item in report["unrecognized"]:
        print(f"unrecognized: {Path(item['file']).name}:{item['line']}  {item['text']}")
    if report["unrecognized"]:
        print()
    print(f"summary: {len(report['config_files'])} config(s), {len(report['code_files'])} file(s), "
          f"{len(report['matches'])} value(s), {total_hits} hit(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="version_scope",
        description="Map config version string(s) to the code lines that execute for them.",
    )
    parser.add_argument("configs", help="YAML config file or directory of configs")
    parser.add_argument("code", help="Python file or directory of Python files")
    parser.add_argument("--units", help="mreview split units.json for line-to-unit mapping")
    parser.add_argument("--json", dest="json_out", help="write structured report to this JSON path")
    args = parser.parse_args(argv)

    report = build_report(Path(args.configs), Path(args.code), Path(args.units) if args.units else None)
    print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
