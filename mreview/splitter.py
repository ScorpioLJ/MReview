"""Split Python sources into review units.

Every line of every reviewed file belongs to exactly one unit. Units are
functions / methods (split at syntactic seams when longer than MAX_LINES),
class-body statements, and module-level flow statements.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .models import Unit

MAX_LINES = 150

_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".mreview-results",
    ".venv",
    "venv",
    "node_modules",
    ".tox",
}


def _node_start(node: ast.AST) -> int:
    if getattr(node, "decorator_list", None):
        return node.decorator_list[0].lineno
    return node.lineno


def _node_end(node: ast.AST) -> int:
    return getattr(node, "end_lineno", node.lineno)


def _chunk_spans(statements: list[ast.stmt], lead_start: int, lead_end: int) -> list[tuple[int, int]]:
    """Group statements into contiguous spans of at most MAX_LINES lines."""
    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    for st in statements:
        s, e = st.lineno, getattr(st, "end_lineno", st.lineno)
        if cur_start is None:
            cur_start, cur_end = s, e
        elif (e - cur_start + 1) <= MAX_LINES:
            cur_end = e
        else:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    if not spans:
        spans.append((lead_start, lead_end))
    spans[0] = (lead_start, spans[0][1])
    return spans


def _collect_intervals(tree: ast.Module) -> tuple[list[tuple[int, int, str, str]], list[tuple[int, int, str]]]:
    """Return (function/method intervals, class-body intervals)."""
    funcs: list[tuple[int, int, str, str]] = []
    class_body: list[tuple[int, int, str]] = []

    def walk(body: list[ast.stmt], prefix: str, in_class: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append(
                    (_node_start(node), _node_end(node), f"{prefix}{node.name}",
                     "method" if in_class else "function")
                )
            elif isinstance(node, ast.ClassDef):
                for st in node.body:
                    if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        class_body.append((st.lineno, getattr(st, "end_lineno", st.lineno), node.name))
                walk(node.body, f"{node.name}.", True)

    walk(tree.body, "", False)
    return funcs, class_body


def _split_function(start: int, end: int, qualname: str, unit_type: str, body: list[ast.stmt]) -> list[Unit]:
    if end - start + 1 <= MAX_LINES:
        return [Unit("", "", qualname, unit_type, start, end)]
    units = []
    for i, (s, e) in enumerate(_chunk_spans(body, start, end)):
        name = f"{qualname}#{i}" if i else qualname
        units.append(Unit("", "", name, unit_type, s, e))
    return units


def _chunk_run(start: int, end: int) -> list[tuple[int, int]]:
    if end - start + 1 <= MAX_LINES:
        return [(start, end)]
    chunks = []
    s = start
    while s <= end:
        e = min(s + MAX_LINES - 1, end)
        chunks.append((s, e))
        s = e + 1
    return chunks


def _is_blank_or_comment(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith("#")


def split_file(path: Path, rel_path: str) -> list[Unit]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    total = len(lines)
    tree = ast.parse(source)

    funcs, class_body = _collect_intervals(tree)
    body_stmts: dict[tuple[int, int], list[ast.stmt]] = {}

    def collect_bodies(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = (_node_start(child), _node_end(child))
                body_stmts[key] = [st for st in child.body]
                collect_bodies(child)

    collect_bodies(tree)

    units: list[Unit] = []
    for start, end, qualname, unit_type in funcs:
        body = body_stmts.get((start, end), [])
        units.extend(_split_function(start, end, qualname, unit_type, body))

    seen_names: set[tuple[str, str]] = set()
    for start, end, class_name in class_body:
        key = (class_name, start)
        if key in seen_names:
            continue
        seen_names.add(key)
        units.append(Unit("", "", f"{class_name}.class_body", "class_body", start, end))

    units = sorted(units, key=lambda u: (u.start_line, u.end_line))

    # Coverage pass: attribute every uncovered line.
    fixed: list[Unit] = []
    cursor = 1
    for u in units:
        if u.start_line > cursor:
            run = list(range(cursor, u.start_line))
            code_lines = [i for i in run if not _is_blank_or_comment(lines[i - 1])]
            if not code_lines and fixed:
                last = fixed[-1]
                fixed[-1] = Unit(last.unit_id, last.file, last.name, last.unit_type,
                                 last.start_line, u.start_line - 1)
            else:
                for s, e in _chunk_run(run[0], run[-1]):
                    fixed.append(Unit("", "", f"module_flow@{s}", "module_flow", s, e))
        fixed.append(u)
        cursor = u.end_line + 1
    if cursor <= total:
        run = list(range(cursor, total + 1))
        code_lines = [i for i in run if not _is_blank_or_comment(lines[i - 1])]
        if not code_lines and fixed:
            last = fixed[-1]
            fixed[-1] = Unit(last.unit_id, last.file, last.name, last.unit_type,
                             last.start_line, total)
        else:
            for s, e in _chunk_run(run[0], run[-1]):
                fixed.append(Unit("", "", f"module_flow@{s}", "module_flow", s, e))

    for u in fixed:
        u.file = rel_path

    seen_ids: dict[str, int] = {}
    for u in fixed:
        base = f"{rel_path}:{u.name}"
        n = seen_ids.get(base, 0)
        seen_ids[base] = n + 1
        u.unit_id = base if n == 0 else f"{base}#{n}"

    return sorted(fixed, key=lambda u: (u.file, u.start_line))


def split_target(target: Path) -> list[Unit]:
    if target.is_file():
        return split_file(target, target.name)
    units: list[Unit] = []
    for path in sorted(target.rglob("*.py")):
        rel = path.relative_to(target)
        if any(part in _EXCLUDE_DIRS for part in rel.parts):
            continue
        units.extend(split_file(path, rel.as_posix()))
    return sorted(units, key=lambda u: (u.file, u.start_line))
