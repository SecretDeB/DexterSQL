"""
Step 4: ground semantic units to schema elements. DETERMINISTIC, retrieval-first.

Sources, in priority order:
  1. retrieved_values      (value-retrieval hits: strongest evidence for literals)
  2. exact / normalized column-name match
  3. column description + value_examples match
  4. table-name match
  5. fuzzy (token-overlap) match above a threshold

Anything that stays ambiguous or unmatched becomes an `Unresolved` entry — the
ONLY thing the IR-completion LLM call is permitted to decide. That boundary is
what keeps the decomposition deterministic (handoff §10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from .units import Unit


@dataclass
class ColumnInfo:
    table: str
    column: str
    ctype: str = ""
    primary_key: bool = False
    foreign_keys: List[str] = field(default_factory=list)
    description: str = ""
    value_examples: List[str] = field(default_factory=list)
    distinct_count: Optional[int] = None

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass
class GroundedSchema:
    columns: List[ColumnInfo] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    fks: List[Tuple[str, str]] = field(default_factory=list)   # (Tab.col -> Tab2.col2)
    pks: Dict[str, List[str]] = field(default_factory=dict)

    def by_qualified(self, q: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.qualified.lower() == (q or "").lower():
                return c
        return None

    def columns_of(self, table: str) -> List[ColumnInfo]:
        return [c for c in self.columns if c.table.lower() == (table or "").lower()]


def load_schema(database_schema: Dict[str, Any]) -> GroundedSchema:
    """Parse a focused (post-schema-linking) database schema dict."""
    gs = GroundedSchema()
    tables = (database_schema or {}).get("tables", {}) or {}
    for tname, tinfo in tables.items():
        gs.tables.append(tname)
        pk_cols = []
        for cname, cinfo in (tinfo.get("columns", {}) or {}).items():
            ci = ColumnInfo(
                table=tname, column=cname,
                ctype=(cinfo.get("column_type") or ""),
                primary_key=bool(cinfo.get("primary_key")),
                foreign_keys=list(cinfo.get("foreign_keys") or []),
                description=(cinfo.get("description") or ""),
                value_examples=[str(v) for v in (cinfo.get("value_examples") or []) if v != ""],
                distinct_count=((cinfo.get("value_statistics") or {}) or {}).get("distinct_count"),
            )
            gs.columns.append(ci)
            if ci.primary_key:
                pk_cols.append(cname)
            for fk in ci.foreign_keys:
                tgt = _fk_target(fk)
                if tgt:
                    gs.fks.append((ci.qualified, tgt))
        if pk_cols:
            gs.pks[tname] = pk_cols
    return gs


def _fk_target(fk: Any) -> Optional[str]:
    """foreign_keys entries appear as 'Tab.col', {'table':..,'column':..}, or [t,c]."""
    if isinstance(fk, str) and "." in fk:
        return fk
    if isinstance(fk, dict):
        t = fk.get("table") or fk.get("referenced_table")
        c = fk.get("column") or fk.get("referenced_column")
        if t and c:
            return f"{t}.{c}"
    if isinstance(fk, (list, tuple)) and len(fk) == 2:
        return f"{fk[0]}.{fk[1]}"
    return None


# ── normalization helpers ────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _tokens(s: str) -> List[str]:
    return [t for t in _norm(s).split() if t]


def _singular(s: str) -> str:
    """Crude de-pluralization so 'patients' matches the Patient table."""
    w = (s or "").strip()
    for suf, rep in (("ies", "y"), ("ses", "s"), ("s", "")):
        if len(w) > 3 and w.lower().endswith(suf):
            return w[: -len(suf)] + rep
    return w


def _sim(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if ta and tb:
        jac = len(ta & tb) / len(ta | tb)
        if ta <= tb or tb <= ta:
            jac = max(jac, 0.85)
    else:
        jac = 0.0
    return max(jac, SequenceMatcher(None, na, nb).ratio())


# marker for "the LLM grounder explicitly declined to map this mention"
DECLINED = "llm_declined"

COL_MATCH_THRESHOLD = 0.72
QUAL_MATCH_THRESHOLD = 0.88    # "Table Column" match must be near-exact to count
TABLE_MATCH_THRESHOLD = 0.80   # mention that names a table resolves to the table
AMBIGUITY_MARGIN = 0.06        # two candidates this close => unresolved


@dataclass
class Grounding:
    """Result of grounding one unit."""
    unit: Unit
    column: Optional[str] = None            # "Table.Column"
    table: Optional[str] = None
    value: Any = None
    confidence: float = 0.0
    candidates: List[str] = field(default_factory=list)
    method: str = ""                        # value_hit | name | description | table | fuzzy
    ambiguous: bool = False


class Grounder:
    def __init__(self, schema: GroundedSchema,
                 retrieved_values: Optional[Dict[str, Dict[str, Any]]] = None,
                 notes: Optional[List[str]] = None):
        self.schema = schema
        self.retrieved = retrieved_values or {}
        self.notes = notes or []
        self._note_pref = self._parse_note_preferences(self.notes)
        # Value retrieval covers the WHOLE database, but grounding must stay inside
        # the focused schema — otherwise a literal can land on a column the linker
        # deliberately excluded (observed: "1930" -> Laboratory.RF).
        self._allowed = {c.qualified.lower() for c in schema.columns}

    def _in_focus(self, qualified: str) -> bool:
        return (qualified or "").lower() in self._allowed

    # ── value grounding (strongest signal — for STRING literals) ─────────────
    def ground_value(self, literal: str, numeric: bool = False) -> List[Grounding]:
        """Find columns whose retrieved values / examples contain this literal.

        `numeric=True` heavily downweights the evidence: a bare number like 1930 or
        30609 occurs incidentally in many numeric columns, so a raw value match is
        NOT sufficient grounding. Numeric literals are instead attached by syntactic
        proximity to a grounded attribute mention (see ir_builder), and these hits
        only act as a tie-breaker.
        """
        hits: List[Tuple[float, str, Any]] = []
        lit_n = _norm(literal)
        if not lit_n:
            return []
        penalty = 0.45 if numeric else 0.0
        for tname, cols in (self.retrieved or {}).items():
            for cname, vals in (cols or {}).items():
                q = f"{tname}.{cname}"
                if not self._in_focus(q):
                    continue
                for v in (vals or []):
                    val = v.get("value") if isinstance(v, dict) else v
                    dist = v.get("distance", 1.0) if isinstance(v, dict) else 1.0
                    if val is None:
                        continue
                    if _norm(str(val)) == lit_n:
                        hits.append((1.0 - float(dist) * 0.1 - penalty, q, val))
                    elif lit_n and lit_n in _norm(str(val)):
                        hits.append((0.8 - float(dist) * 0.1 - penalty, q, val))
        for c in self.schema.columns:                      # schema value_examples
            for ex in c.value_examples[:20]:
                if _norm(str(ex)) == lit_n:
                    hits.append((0.95 - penalty, c.qualified, ex))
        hits.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for score, q, val in hits:
            if q in seen:
                continue
            seen.add(q)
            out.append(Grounding(unit=Unit("literal", literal), column=q,
                                 table=q.split(".")[0], value=val, confidence=score,
                                 method="value_hit"))
        return out

    # ── column grounding ─────────────────────────────────────────────────────
    def ground_mention(self, mention: str) -> List[Grounding]:
        scored: List[Tuple[float, ColumnInfo, str]] = []
        for c in self.schema.columns:
            s_name = _sim(mention, c.column)
            # "Table Column" similarity is only a TIE-BREAKER, never sufficient on
            # its own: "patients" vs "Patient SEX" scores high on raw string overlap
            # purely because of the table half, which produced COUNT(Patient.SEX).
            s_qual = _sim(mention, f"{c.table} {c.column}")
            s_desc = 0.0
            if c.description:
                desc_head = c.description.split("|")[0]
                if _norm(mention) and _norm(mention) in _norm(desc_head):
                    s_desc = 0.80
            best = max(s_name, s_desc)
            method = "name" if s_name >= s_desc else "description"
            if best < COL_MATCH_THRESHOLD and s_qual >= QUAL_MATCH_THRESHOLD:
                best, method = s_qual, "qualified"
            if best >= COL_MATCH_THRESHOLD:
                scored.append((best + self._note_bonus(c.qualified), c, method))
        scored.sort(key=lambda x: -x[0])
        return [Grounding(unit=Unit("attribute", mention), column=c.qualified,
                          table=c.table, confidence=min(s, 1.0), method=m)
                for s, c, m in scored[:5]]

    def ground_table(self, mention: str, strong: bool = False) -> Optional[str]:
        """`strong=True` demands a near-exact table-name match (singular/plural of
        the table itself), used to short-circuit column matching."""
        best, bs = None, 0.0
        for t in self.schema.tables:
            s = max(_sim(mention, t), _sim(_singular(mention), t))
            if s > bs:
                best, bs = t, s
        return best if bs >= (TABLE_MATCH_THRESHOLD if strong else COL_MATCH_THRESHOLD) else None

    # ── disambiguation notes bias the ranking (handoff: notes are an input) ───
    @staticmethod
    def _parse_note_preferences(notes: List[str]) -> Dict[str, float]:
        pref: Dict[str, float] = {}
        for n in notes or []:
            for m in re.finditer(r"use\s+([A-Za-z_]\w*\.[A-Za-z_][\w ]*?)\s+when", n, re.I):
                pref[m.group(1).strip().lower()] = 0.08
            for m in re.finditer(r"(?:Filter on|Use)\s+([A-Za-z_]\w*\.[A-Za-z_][\w ]*)", n, re.I):
                pref.setdefault(m.group(1).strip().lower(), 0.05)
        return pref

    def _note_bonus(self, qualified: str) -> float:
        return self._note_pref.get((qualified or "").lower(), 0.0)

    # ── LLM-assisted grounding overrides (step 4, optional) ──────────────────
    def set_overrides(self, overrides: Dict[int, Dict[str, Any]]) -> None:
        """Column decisions made by the LLM grounding call, keyed by unit index.

        These take precedence over deterministic matching because proximity/fuzzy
        matching mis-binds descriptive mentions ("total enrollment", "ages of 5 and
        17") onto whatever column is nearest. Structure stays deterministic; only
        the mention->column choice is delegated.
        """
        self._overrides = dict(overrides or {})

    def _override_for(self, idx: int) -> Optional[Dict[str, Any]]:
        return getattr(self, "_overrides", {}).get(idx)

    # ── top-level: ground every unit ─────────────────────────────────────────
    def ground_units(self, us: List[Unit]) -> Tuple[Dict[int, Grounding], List[Dict[str, Any]]]:
        """Returns (grounding per unit index, unresolved records)."""
        out: Dict[int, Grounding] = {}
        unresolved: List[Dict[str, Any]] = []
        for idx, u in enumerate(us):
            ov = self._override_for(idx)
            if ov is not None and ov.get("column"):
                col = ov["column"]
                out[idx] = Grounding(unit=u, column=col, table=col.split(".")[0],
                                     value=(ov.get("value")
                                            if u.kind == "literal" else None),
                                     confidence=0.95, method="llm")
                continue
            if ov is not None and not ov.get("column"):
                # The LLM looked at this mention and DELIBERATELY said "no column"
                # (it names a table, or is part of a column name like "SAT test").
                # Record the refusal EXPLICITLY rather than just skipping: an absent
                # grounding is indistinguishable from "not yet tried", and the IR
                # builder's proximity fallback then resurrects the mention as a
                # predicate (schools.MailStreet = 'FRPM', satscores.cds = 'SAT').
                if u.kind == "literal" and ov.get("value") is not None:
                    u.payload["value"] = ov["value"]
                out[idx] = Grounding(unit=u, column=None, table=None,
                                     confidence=0.0, method=DECLINED)
                continue
            if u.kind == "literal":
                is_num = (u.payload.get("literal_type") in {"number", "date"}
                          or u.payload.get("ent_type") in {"CARDINAL", "DATE", "MONEY", "PERCENT"})
                cands = self.ground_value(str(u.payload.get("value", u.text)), numeric=is_num)
                if not cands:
                    continue
                if is_num:
                    # numeric literals are resolved by proximity in the IR builder;
                    # record candidates only, never a committed column.
                    g = Grounding(unit=u, column=None, table=None,
                                  value=u.payload.get("value", u.text),
                                  confidence=cands[0].confidence,
                                  candidates=[c.column for c in cands[:4]],
                                  method="numeric_weak")
                    out[idx] = g
                    continue
                g = cands[0]
                g.unit = u
                g.candidates = [c.column for c in cands[:4]]
                if len(cands) > 1 and (cands[0].confidence - cands[1].confidence) < AMBIGUITY_MARGIN:
                    g.ambiguous = True
                    unresolved.append({"kind": "value", "mention": u.text,
                                       "source_tokens": u.source_tokens,
                                       "candidates": g.candidates,
                                       "note": "literal matches multiple columns"})
                out[idx] = g
            elif u.kind in {"entity", "attribute"}:
                # A mention that names a TABLE ("patients") is an entity reference,
                # not a column: resolve it to the table so downstream rules use its
                # key rather than fuzzy-matching some column of that table.
                t_exact = self.ground_table(u.text, strong=True)
                if t_exact:
                    out[idx] = Grounding(unit=u, table=t_exact, confidence=0.9,
                                         method="table")
                    continue
                cands = self.ground_mention(u.text)
                if not cands:
                    t = self.ground_table(u.text)
                    if t:
                        out[idx] = Grounding(unit=u, table=t, confidence=0.8, method="table")
                    else:
                        unresolved.append({"kind": "column", "mention": u.text,
                                           "source_tokens": u.source_tokens,
                                           "candidates": [],
                                           "note": "no schema element matched"})
                    continue
                g = cands[0]
                g.unit = u
                g.candidates = [c.column for c in cands[:4]]
                if len(cands) > 1 and (cands[0].confidence - cands[1].confidence) < AMBIGUITY_MARGIN:
                    g.ambiguous = True
                    unresolved.append({"kind": "column", "mention": u.text,
                                       "source_tokens": u.source_tokens,
                                       "candidates": g.candidates,
                                       "note": "mention matches multiple columns"})
                out[idx] = g
        return out, unresolved


# ── join-graph utilities (validation §7: connected through valid keys) ───────
def find_join_path(schema: GroundedSchema, tables: List[str]) -> Tuple[List[Tuple[str, str]], bool]:
    """Return (join edges, connected?) linking `tables` through FK edges (BFS)."""
    tabs = [t for t in dict.fromkeys(tables) if t]
    if len(tabs) <= 1:
        return [], True
    adj: Dict[str, List[Tuple[str, str, str]]] = {}
    for a, b in schema.fks:
        ta, tb = a.split(".")[0], b.split(".")[0]
        adj.setdefault(ta, []).append((tb, a, b))
        adj.setdefault(tb, []).append((ta, b, a))
    edges: List[Tuple[str, str]] = []
    reached = {tabs[0]}
    for target in tabs[1:]:
        if target in reached:
            continue
        path = _bfs(adj, reached, target)
        if path is None:
            return edges, False
        for (l, r) in path:
            if (l, r) not in edges and (r, l) not in edges:
                edges.append((l, r))
            reached.add(l.split(".")[0])
            reached.add(r.split(".")[0])
    return edges, True


def _bfs(adj, sources: set, target: str) -> Optional[List[Tuple[str, str]]]:
    from collections import deque
    q = deque((s, []) for s in sources)
    seen = set(sources)
    while q:
        cur, path = q.popleft()
        if cur == target:
            return path
        for (nxt, lcol, rcol) in adj.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, path + [(lcol, rcol)]))
    return None
