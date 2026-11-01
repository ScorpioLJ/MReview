from pathlib import Path

from mreview.splitter import split_file


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_every_line_assigned_once(tmp_path):
    src = (
        "import os\n"
        "import torch\n"
        "\n"
        "def foo(a):\n"
        "    return a + 1\n"
        "\n"
        "class C:\n"
        "    x = 1\n"
        "    def m(self):\n"
        "        return self.x\n"
        "\n"
        "y = foo(1)\n"
        "\n"
    )
    p = _write(tmp_path, "mod.py", src)
    units = split_file(p, "mod.py")
    lines = src.splitlines()
    coverage = [False] * len(lines)
    for u in units:
        for i in range(u.start_line - 1, u.end_line):
            assert not coverage[i], f"line {i + 1} double-assigned"
            coverage[i] = True
    assert all(coverage), [i + 1 for i, c in enumerate(coverage) if not c]
    types = {u.unit_type for u in units}
    assert "module_flow" in types
    assert "function" in types
    assert "class_body" in types
    assert "method" in types


def test_long_function_split(tmp_path):
    body = [f"    x{i} = {i}" for i in range(200)]
    src = "def long_fn():\n" + "\n".join(body)
    p = _write(tmp_path, "long.py", src)
    units = split_file(p, "long.py")
    assert len(units) >= 2
    for u in units:
        assert u.end_line - u.start_line + 1 <= 155


def test_imports_become_module_flow(tmp_path):
    src = "import os\nimport sys\n\nVALUE = 3\n"
    p = _write(tmp_path, "imp.py", src)
    units = split_file(p, "imp.py")
    assert any(u.unit_type == "module_flow" for u in units)
    assert units[0].start_line == 1
