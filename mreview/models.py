from __future__ import annotations

from dataclasses import asdict, dataclass

SEVERITIES = ("P0", "P1", "P2", "P3")
CATEGORIES = ("design", "correctness", "performance", "weights", "style")


@dataclass
class Unit:
    unit_id: str
    file: str
    name: str
    unit_type: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Question:
    id: str
    title: str
    prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Finding:
    source: str
    category: str
    severity: str
    file: str | None
    line: int | None
    title: str
    cause: str = ""
    fix: str = ""
    evidence: str = ""
    code: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = ""
        return d
