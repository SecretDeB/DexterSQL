"""
Typed SQL intermediate representation (IR) for Dependency-Tree SQL Generation.

The IR is the contract between the deterministic decomposition (parser -> units ->
rules) and the LLM. The LLM may RESOLVE unresolved mappings and RENDER the IR as
SQL, but must not silently redefine the decomposition — every structural slot here
is populated by deterministic code or by an explicitly-scoped grounding call.

Every element carries `source_tokens` so the coverage report can connect question
elements to SQL components (and so validation can prove nothing was invented).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ── operator vocabulary ───────────────────────────────────────────────────────
OPS = {"=", "!=", ">", "<", ">=", "<=", "LIKE", "NOT LIKE", "IN", "NOT IN",
       "IS NULL", "IS NOT NULL", "BETWEEN"}
AGGS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


@dataclass
class SelectItem:
    expression: str                                  # e.g. "COUNT(*)", "Patient.SEX"
    source_tokens: List[str] = field(default_factory=list)
    alias: Optional[str] = None

    def to_sql(self) -> str:
        return f"{self.expression} AS {self.alias}" if self.alias else self.expression


@dataclass
class Join:
    left: str                                        # "Examination.ID"
    right: str                                       # "Patient.ID"
    type: str = "INNER"                              # INNER | LEFT


@dataclass
class Filter:
    column: str                                      # "Patient.Diagnosis"
    operator: str                                    # one of OPS
    value: Any = None                                # literal, list (IN), or [lo,hi]
    source_tokens: List[str] = field(default_factory=list)
    connector: str = "AND"                           # AND | OR  (joins to PREVIOUS filter)
    negated: bool = False

    def is_valid(self) -> bool:
        return bool(self.column) and self.operator in OPS


@dataclass
class OrderBy:
    expression: str
    direction: str = "ASC"                           # ASC | DESC
    source_tokens: List[str] = field(default_factory=list)


@dataclass
class Unresolved:
    """A mapping deterministic grounding could NOT decide — the only thing the
    IR-completion LLM call is allowed to touch."""
    kind: str                                        # column | value | table | operator | aggregate
    mention: str                                     # surface text from the question
    source_tokens: List[str] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    note: str = ""


@dataclass
class SQLIR:
    dialect: str = "sqlite"
    select: List[SelectItem] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    joins: List[Join] = field(default_factory=list)
    filters: List[Filter] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    having: List[Filter] = field(default_factory=list)
    order_by: List[OrderBy] = field(default_factory=list)
    limit: Optional[int] = None
    distinct: bool = False
    unresolved: List[Unresolved] = field(default_factory=list)
    coverage: List[str] = field(default_factory=list)   # question units the IR realizes

    # ── serialization ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SQLIR":
        """Rebuild from (possibly LLM-produced) dict. Unknown keys are ignored and
        malformed entries dropped — fail-closed, never raise."""
        ir = SQLIR()
        if not isinstance(d, dict):
            return ir
        ir.dialect = d.get("dialect") or "sqlite"
        ir.distinct = bool(d.get("distinct", False))
        for s in d.get("select") or []:
            if isinstance(s, dict) and s.get("expression"):
                ir.select.append(SelectItem(str(s["expression"]),
                                            list(s.get("source_tokens") or []),
                                            s.get("alias")))
            elif isinstance(s, str):
                ir.select.append(SelectItem(s))
        ir.tables = [str(t) for t in (d.get("tables") or []) if t]
        for j in d.get("joins") or []:
            if isinstance(j, dict) and j.get("left") and j.get("right"):
                ir.joins.append(Join(str(j["left"]), str(j["right"]), j.get("type", "INNER")))
        for key, sink in (("filters", ir.filters), ("having", ir.having)):
            for f in d.get(key) or []:
                if not isinstance(f, dict) or not f.get("column"):
                    continue
                op = str(f.get("operator", "=")).upper()
                if op not in OPS:
                    op = "="
                sink.append(Filter(str(f["column"]), op, f.get("value"),
                                   list(f.get("source_tokens") or []),
                                   str(f.get("connector", "AND")).upper(),
                                   bool(f.get("negated", False))))
        ir.group_by = [str(g) for g in (d.get("group_by") or []) if g]
        for o in d.get("order_by") or []:
            if isinstance(o, dict) and o.get("expression"):
                ir.order_by.append(OrderBy(str(o["expression"]),
                                           str(o.get("direction", "ASC")).upper(),
                                           list(o.get("source_tokens") or [])))
            elif isinstance(o, str):
                ir.order_by.append(OrderBy(o))
        lim = d.get("limit")
        ir.limit = int(lim) if isinstance(lim, (int, float, str)) and str(lim).strip().isdigit() else None
        for u in d.get("unresolved") or []:
            if isinstance(u, dict):
                ir.unresolved.append(Unresolved(str(u.get("kind", "column")),
                                                str(u.get("mention", "")),
                                                list(u.get("source_tokens") or []),
                                                [str(c) for c in (u.get("candidates") or [])],
                                                str(u.get("note", ""))))
        ir.coverage = [str(c) for c in (d.get("coverage") or []) if c]
        return ir

    # ── convenience ───────────────────────────────────────────────────────────
    def all_columns(self) -> List[str]:
        """Every qualified column the IR references (for grounding validation).

        Slot values are used AS-IS (they are already qualified); only aggregate
        wrappers are stripped. Regex-scraping these strings is wrong because BIRD
        column names legitimately contain spaces and parentheses — e.g.
        'frpm.FRPM Count (Ages 5-17)' — and any token-boundary pattern truncates
        them into names that then look 'unknown'.
        """
        cols: List[str] = []
        for f in list(self.filters) + list(self.having):
            if f.column:
                cols.append(f.column)
        cols.extend(self.group_by)
        for o in self.order_by:
            cols.append(_strip_agg(o.expression))
        for s in self.select:
            cols.append(_strip_agg(s.expression))
        out, seen = [], set()
        for c in cols:
            c = (c or "").strip()
            if not c or c == "*" or "." not in c:
                continue
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def is_empty(self) -> bool:
        return not self.select and not self.filters and not self.tables


def _strip_agg(expr: str) -> str:
    """COUNT(Patient.ID) -> Patient.ID ; MAX(frpm.Enrollment (Ages 5-17)) -> the column.
    Greedy inner match so column names that themselves contain parentheses survive."""
    import re
    s = (expr or "").strip()
    m = re.match(r"^(?:COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?(.*)\)$", s, re.I | re.S)
    return m.group(1).strip() if m else s
