"""
Step 5: build the draft SQL IR by traversing the grounded dependency tree
bottom-up and populating typed SQL slots. DETERMINISTIC — no LLM.

Implements the handoff §3 mapping table as INTERPRETATION rules: a dependency
signal proposes a SQL role, and grounding must confirm the actual column/table.
Nothing reaches the IR that grounding could not tie to a schema element; anything
else is recorded in `unresolved` for the (narrowly scoped) LLM completion call.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .parser import DepTree
from .units import Unit, coverage_mentions
from .grounding import Grounder, GroundedSchema, Grounding, find_join_path, DECLINED
from .ir import SQLIR, SelectItem, Filter, Join, OrderBy, Unresolved

# aggregate cues that imply the SELECT is a measure, not a projection
_COUNTING = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
_DERIVED = {"PCT", "RATIO"}          # need arithmetic — flagged, not silently dropped


def build_ir(tree: DepTree, units: List[Unit], grounder: Grounder,
             dialect: str = "sqlite") -> SQLIR:
    ir = SQLIR(dialect=dialect)
    if tree is None:
        return ir

    groundings, unresolved = grounder.ground_units(units)
    schema = grounder.schema

    # ── collect typed unit views ─────────────────────────────────────────────
    aggs = [u for u in units if u.kind == "aggregate"]
    comps = [u for u in units if u.kind == "comparison"]
    orders = [u for u in units if u.kind == "ordering"]
    groups = [u for u in units if u.kind == "group"]
    negs = [u for u in units if u.kind == "negation"]
    conjs = [u for u in units if u.kind == "conjunction"]
    limits = [u for u in units if u.kind == "limit"]

    used_unit_idx: set = set()
    tables: List[str] = []

    def _touch_table(q: Optional[str]):
        if not q:
            return
        t = q.split(".")[0]
        if t and t not in tables:
            tables.append(t)

    # ── FILTERS: literal groundings + comparison operators ───────────────────
    default_conn = "OR" if any(c.payload.get("connector") == "OR" for c in conjs) else "AND"

    def _add_filter(col: str, op: str, val: Any, toks: List[str]) -> None:
        """Append with de-duplication: the same predicate can be proposed by several
        rules (a literal unit AND its comparison cue), and '30609' vs 30609 must not
        become two filters."""
        if not col or not op:
            return
        # A "literal" equal to the column's own name is a COLUMN REFERENCE that was
        # mis-typed as a value ("LDH beyond normal range" -> Laboratory.LDH = 'LDH').
        # Emitting it produces a nonsense self-referential predicate.
        if val is not None and str(val).strip().lower() == col.split(".")[-1].strip().lower():
            return
        key = (col.lower(), op, _val_key(val))
        for f in ir.filters:
            if (f.column.lower(), f.operator, _val_key(f.value)) == key:
                return
        ir.filters.append(Filter(column=col, operator=op, value=val,
                                 source_tokens=toks, connector=default_conn))
        _touch_table(col)

    # BETWEEN ranges first ("born between 1930 to 1940") so their endpoints are not
    # also emitted as two independent '=' filters.
    consumed_lits: set = set()
    for consumed, anchor_char, lo, hi, toks in _between_ranges(tree, units):
        consumed_lits.update(consumed)
        ranked = _ranked_columns_by_char(tree, anchor_char, units, groundings, schema)
        col = next((c for c in ranked if _type_compatible(schema, c, lo)), None)
        if col:
            _add_filter(col, "BETWEEN", [_coerce(lo), _coerce(hi)], toks)
        else:
            ir.unresolved.append(Unresolved(
                kind="column", mention=f"between {lo} and {hi}", source_tokens=toks,
                candidates=ranked[:4],
                note="range endpoints could not be tied to a compatible column"))

    seen_lit_vals: set = set()
    for idx, u in enumerate(units):
        if u.kind != "literal" or idx in consumed_lits:
            continue
        # the same literal can surface twice (quoted span + numeric token)
        vkey = _val_key(u.payload.get("value", u.text))
        if vkey in seen_lit_vals:
            continue
        seen_lit_vals.add(vkey)

        g = groundings.get(idx)
        if g is not None and g.method == DECLINED:
            continue      # the grounder judged this mention is not a value at all
        if g is not None and g.method == "llm" and g.value is None:
            # The grounding prompt instructs: a phrase that is part of a COLUMN NAME
            # maps to that column with value=null. Honour that contract — "FRPM count"
            # identifies frpm."FRPM Count (K-12)"; it is not the predicate
            # frpm."FRPM Count (K-12)" = 'FRPM'.
            continue
        col = g.column if g else None
        if col is not None and not _type_compatible(schema, col, u.payload.get("value", u.text)):
            # Type compatibility applies to LLM groundings too, not just the
            # proximity fallback: the model can still bind a number to a name column
            # (schools.School > 500). A wrong-but-resolved slot is locked in by
            # _merge_ir, so downgrade it to an explicit question instead.
            ir.unresolved.append(Unresolved(
                kind="value", mention=str(u.text), source_tokens=u.source_tokens,
                candidates=[col], note=f"literal {u.text!r} is not type-compatible "
                                       f"with {col}; choose the intended column"))
            continue
        if col is None:
            # numeric/date literal, or one with no value hit at all: attach by
            # syntactic proximity to a grounded mention (a table mention resolves to
            # its PK). Weak value-hit candidates are only a last resort — this is
            # what stops "1930" from landing on Laboratory.RF.
            lit_val = u.payload.get("value", u.text)
            anchor_char = min((tree.node(i).start_char for i in (u.node_ids or [])
                               if tree.node(i)), default=0)
            ranked = _ranked_columns_by_char(tree, anchor_char, units, groundings, schema)
            col = next((c for c in ranked if _type_compatible(schema, c, lit_val)), None)
            if col is None and g and g.candidates:
                col = next((c for c in g.candidates
                            if _type_compatible(schema, c, lit_val)), None)
            if col is None:
                ir.unresolved.append(Unresolved(
                    kind="value", mention=str(u.text), source_tokens=u.source_tokens,
                    candidates=(g.candidates if g else []),
                    note="literal could not be tied to a type-compatible column"))
                continue
        op, val = _operator_for_literal(tree, u, comps, g)
        if _is_negated(tree, u, negs):
            op = "!=" if op == "=" else op
        _add_filter(col, op, val, u.source_tokens)
        used_unit_idx.add(idx)

    # comparisons whose value never grounded to a column (e.g. "albumin < 3.5")
    for cu in comps:
        cval = cu.payload.get("value")
        if cval is None:
            continue
        if any(_val_key(f.value) == _val_key(_coerce(cval)) for f in ir.filters):
            continue
        col = _column_for_comparison(tree, cu, units, groundings)
        if col:
            _add_filter(col, cu.payload.get("operator", "="), _coerce(cval),
                        cu.source_tokens)

    # ── SELECT ───────────────────────────────────────────────────────────────
    agg_fn = next((a.payload["function"] for a in aggs
                   if a.payload.get("function") in _COUNTING), None)
    derived = next((a.payload["function"] for a in aggs
                    if a.payload.get("function") in _DERIVED), None)

    projections = _projection_columns(units, groundings, used_unit_idx)
    for p in projections:
        _touch_table(p)

    if agg_fn == "COUNT" and not projections:
        ir.select.append(SelectItem("COUNT(*)", _agg_tokens(aggs, "COUNT")))
    elif agg_fn and projections:
        # Attach the aggregate to the projection the cue actually modifies, not to
        # whichever happened to be first — "the highest eligible free rate ... in
        # Alameda County" otherwise yields MAX(schools.County).
        target = _agg_target(tree, aggs, agg_fn, units, groundings, projections, schema)
        if target is None:
            ir.unresolved.append(Unresolved(
                kind="aggregate", mention=agg_fn.lower(),
                source_tokens=_agg_tokens(aggs, agg_fn), candidates=projections[:4],
                note=f"{agg_fn} target is ambiguous: pick which column/expression it "
                     f"applies to (it may be a computed expression, not a bare column)"))
            for p in projections:
                ir.select.append(SelectItem(p))
        else:
            ir.select.append(SelectItem(f"{agg_fn}({target})", _agg_tokens(aggs, agg_fn)))
            for p in projections:
                if p != target:
                    ir.select.append(SelectItem(p))
    elif projections:
        for p in projections:
            ir.select.append(SelectItem(p))
    elif agg_fn:
        ir.select.append(SelectItem(f"{agg_fn}(*)" if agg_fn == "COUNT" else f"{agg_fn}(*)",
                                    _agg_tokens(aggs, agg_fn)))

    if derived:   # percentage/ratio need arithmetic the rules must not invent
        ir.unresolved.append(Unresolved(
            kind="aggregate", mention=derived.lower(),
            source_tokens=_agg_tokens(aggs, derived),
            candidates=[], note=f"{derived} requires an arithmetic expression "
                                f"(e.g. CAST(SUM(CASE WHEN ...) AS REAL)/COUNT(*)); "
                                f"resolve the numerator/denominator explicitly"))

    if not ir.select:
        # The question named an entity but no projectable column grounded (e.g.
        # "Which schools ...", where 'schools' resolves to the TABLE). Do not guess a
        # column — emit an explicitly scoped task so the completion call has a
        # well-defined question instead of an empty slot.
        ent_tables = [g.table for g in groundings.values() if g.table and not g.column]
        cand_cols: List[str] = []
        for t in dict.fromkeys(ent_tables + tables):
            cand_cols.extend(c.qualified for c in schema.columns_of(t)[:12])
        ir.unresolved.append(Unresolved(
            kind="column",
            mention=(ent_tables[0] if ent_tables else (tables[0] if tables else "result")),
            source_tokens=[], candidates=cand_cols[:12],
            note="no projectable column grounded; choose what the question asks to "
                 "return (a column, or COUNT(*) if it asks how many)"))

    # ── GROUP BY ─────────────────────────────────────────────────────────────
    for gu in groups:
        col = _column_for_group(gu, units, groundings)
        if col:
            if col not in ir.group_by:
                ir.group_by.append(col)
            _touch_table(col)

    # ── ORDER BY / LIMIT (superlatives => ORDER BY + LIMIT 1) ────────────────
    for ou in orders:
        col = _column_for_order(ou, units, groundings, projections)
        if not col:
            continue
        if any(o.expression == col for o in ir.order_by):
            continue
        ir.order_by.append(OrderBy(col, ou.payload.get("direction", "ASC"),
                                   ou.source_tokens))
        _touch_table(col)
        if ou.payload.get("superlative") and ir.limit is None and not agg_fn:
            ir.limit = 1
    for lu in limits:
        ir.limit = int(lu.payload.get("value", 1))

    # ── TABLES + JOINS ───────────────────────────────────────────────────────
    ir.tables = tables or _fallback_tables(schema, groundings)
    edges, connected = find_join_path(schema, ir.tables)
    for (l, r) in edges:
        ir.joins.append(Join(l, r, "INNER"))
        _touch_table(l)
        _touch_table(r)
    ir.tables = list(dict.fromkeys(ir.tables + [e[0].split(".")[0] for e in edges] +
                                   [e[1].split(".")[0] for e in edges]))
    if not connected:
        ir.unresolved.append(Unresolved(
            kind="table", mention=", ".join(ir.tables), source_tokens=[],
            candidates=schema.tables,
            note="no FK path connects these tables; choose the correct join or drop a table"))

    # ── unresolved from grounding + coverage report ──────────────────────────
    for u in unresolved:
        ir.unresolved.append(Unresolved(kind=u["kind"], mention=u["mention"],
                                        source_tokens=u.get("source_tokens", []),
                                        candidates=u.get("candidates", []),
                                        note=u.get("note", "")))
    ir.coverage = coverage_mentions(units)
    ir.distinct = _wants_distinct(tree)
    return ir


# ── helpers ──────────────────────────────────────────────────────────────────
_NON_AGGREGABLE = re.compile(r"name|type|street|phone|city|county|district|address|"
                             r"website|status|code|zip|school$", re.I)


def _agg_target(tree: DepTree, aggs: List[Unit], fn: str, units: List[Unit],
                groundings: Dict[int, Grounding], projections: List[str],
                schema: GroundedSchema) -> Optional[str]:
    """Which projection does the aggregate cue modify?

    Nearest grounded projection to the cue token, skipping columns that make no
    sense to aggregate (MAX/AVG over a name/phone/county is always a mis-binding).
    Returns None when nothing sensible is nearby, so the choice becomes an explicit
    unresolved task instead of a wrong guess.
    """
    numeric_ok = []
    for p in projections:
        ci = schema.by_qualified(p)
        textual = _NON_AGGREGABLE.search(p.split(".")[-1]) is not None
        if fn in {"SUM", "AVG"}:
            is_num = ci is not None and any(t in (ci.ctype or "").upper()
                                            for t in ("INT", "REAL", "NUM", "FLOAT", "DEC"))
            if not is_num:
                continue
        elif textual:
            continue
        numeric_ok.append(p)
    if not numeric_ok:
        return None
    if len(numeric_ok) == 1:
        return numeric_ok[0]
    anchor = None
    for a in aggs:
        if a.payload.get("function") == fn and a.node_ids:
            anchor = min((tree.node(i).start_char for i in a.node_ids if tree.node(i)),
                         default=None)
            break
    if anchor is None:
        return numeric_ok[0]
    best, bestd = None, 10 ** 9
    for idx, u in enumerate(units):
        g = groundings.get(idx)
        if not g or g.column not in numeric_ok:
            continue
        d = min((abs(tree.node(i).start_char - anchor) for i in (u.node_ids or [])
                 if tree.node(i)), default=10 ** 9)
        if d < bestd:
            best, bestd = g.column, d
    return best or numeric_ok[0]


def _agg_tokens(aggs: List[Unit], fn: str) -> List[str]:
    for a in aggs:
        if a.payload.get("function") == fn:
            return a.source_tokens
    return []


def _coerce(v: Any) -> Any:
    s = str(v).replace(",", "").strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return v


def _val_key(v: Any) -> str:
    """Normalized value identity so 30609 and '30609' collapse to one filter."""
    if isinstance(v, (list, tuple)):
        return "|".join(_val_key(x) for x in v)
    c = _coerce(v)
    if isinstance(c, float) and c.is_integer():
        c = int(c)
    return str(c).strip().strip("'\"").lower()


_BETWEEN_RE = re.compile(
    r"\bbetween\b[^.;]*?(-?\d[\d,]*(?:\.\d+)?)\s*(?:and|to|-|through|until)\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.I)


def _between_ranges(tree: DepTree, units: List[Unit]):
    """Detect 'between X and Y' / 'from X to Y' ranges in the question.

    Works off CHARACTER offsets rather than literal-unit values, because spaCy
    collapses date entities ("Year 1930 to 1940") into a single node — matching by
    value alone silently missed those and emitted the whole span as one '=' filter.

    Yields (consumed_unit_idxs, anchor_char, lo, hi, source_tokens).
    """
    q = tree.question or ""
    out = []
    for m in _BETWEEN_RE.finditer(q):
        lo, hi = m.group(1), m.group(2)
        s, e = m.start(), m.end()
        consumed = {i for i, u in enumerate(units)
                    if u.kind == "literal" and _unit_overlaps(tree, u, s, e)}
        out.append((consumed, s, lo, hi, [m.group(0)]))
    return out


def _unit_overlaps(tree: DepTree, u: Unit, s: int, e: int) -> bool:
    for i in (u.node_ids or []):
        n = tree.node(i)
        if n and n.start_char < e and n.end_char > s:
            return True
    return False


# ── literal / column type compatibility ──────────────────────────────────────
_YEARISH = re.compile(r"^(1[6-9]\d{2}|20\d{2})$")
_DATEISH = re.compile(r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?$")
_DATE_NAME = re.compile(r"date|birth|year|day|time", re.I)


def _type_compatible(schema: GroundedSchema, qualified: str, value: Any) -> bool:
    """Reject grounding a literal onto a column whose type it cannot plausibly be.

    Without this, proximity attachment happily produced `Patient.ID = '1930'` — a
    birth year bound to an integer primary key. A wrong-but-resolved filter is worse
    than an unresolved one, because the LLM is instructed to keep resolved slots
    byte-identical.
    """
    ci = schema.by_qualified(qualified)
    if ci is None:
        return True
    s = str(value).strip().strip("'\"")
    ctype = (ci.ctype or "").upper()
    is_year = bool(_YEARISH.match(s))
    is_date = bool(_DATEISH.match(s))
    numeric = False
    try:
        float(s.replace(",", ""))
        numeric = True
    except ValueError:
        pass

    if is_year or is_date:
        # a year/date may only land on a date-ish column: by name/description, by
        # declared type, or by its observed values actually looking like dates.
        # (A bare TEXT type is NOT enough — Patient.Admission is TEXT holding '+'/'-'.)
        if _DATE_NAME.search(ci.column) or _DATE_NAME.search(ci.description or ""):
            return True
        if ci.primary_key:
            return False
        if "DATE" in ctype or "TIME" in ctype:
            return True
        ex = [str(e).strip() for e in ci.value_examples[:10] if str(e).strip()]
        return any(_DATEISH.match(e) or _YEARISH.match(e) for e in ex) if ex else False
    if numeric and ctype in {"TEXT", "VARCHAR", "CHAR"}:
        # numeric literal on a text column: only if its observed values look numeric
        ex = [e for e in ci.value_examples[:10] if str(e).strip()]
        return any(str(e).strip().replace(".", "").isdigit() for e in ex) if ex else True
    if not numeric and ctype in {"INTEGER", "INT", "REAL", "NUMERIC", "FLOAT"}:
        return False
    return True


def _ranked_columns_by_char(tree: DepTree, anchor_char: int, units: List[Unit],
                            groundings: Dict[int, Grounding],
                            schema: Optional[GroundedSchema] = None) -> List[str]:
    """Grounded columns ordered by character distance from an anchor.

    Returns a RANKED LIST (not just the closest) so the caller can skip candidates
    that fail type compatibility instead of giving up on the nearest one — that is
    how 'between 1930 and 1940' reaches Patient.Birthday past Patient.Admission.
    """
    scored: List[Tuple[int, str]] = []
    for idx, u in enumerate(units):
        if u.kind not in {"attribute", "entity"}:
            continue
        g = groundings.get(idx)
        if not g:
            continue
        cands = []
        if g.column:
            cands.append(g.column)
        cands.extend([c for c in (g.candidates or []) if c and c != g.column])
        if not cands and schema is not None and g.table:
            pks = schema.pks.get(g.table) or []
            if pks:
                cands.append(f"{g.table}.{pks[0]}")
        if not cands:
            continue
        d = min((abs((tree.node(i).start_char if tree.node(i) else 10**9) - anchor_char)
                 for i in (u.node_ids or [])), default=10**9)
        for rank, c in enumerate(cands):
            scored.append((d + rank, c))
    scored.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, c in scored:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _nearest_column_by_char(tree: DepTree, anchor_char: int, units: List[Unit],
                            groundings: Dict[int, Grounding],
                            schema: Optional[GroundedSchema] = None) -> Optional[str]:
    r = _ranked_columns_by_char(tree, anchor_char, units, groundings, schema)
    return r[0] if r else None


def _nearest_column(tree: DepTree, anchor: int, units: List[Unit],
                    groundings: Dict[int, Grounding],
                    schema: Optional[GroundedSchema] = None,
                    prefer_key: bool = False) -> Optional[str]:
    """Closest grounded mention to a token position, as a qualified column.

    A mention that grounded to a TABLE rather than a column ("patient '30609'")
    resolves to that table's primary key when `prefer_key` is set — the standard
    'entity + identifier' reading. Without this, an id literal drifts onto whatever
    column happens to be nearest (observed: Patient.Diagnosis = 30609).
    """
    best, bestd = None, 10**9
    for idx, u in enumerate(units):
        if u.kind not in {"attribute", "entity"}:
            continue
        g = groundings.get(idx)
        if not g:
            continue
        cand = g.column
        if cand is None and prefer_key and schema is not None and g.table:
            pks = schema.pks.get(g.table) or []
            cand = f"{g.table}.{pks[0]}" if pks else None
        if not cand:
            continue
        d = min((abs(i - anchor) for i in (u.node_ids or [])), default=10**9)
        if d < bestd:
            best, bestd = cand, d
    return best


def _operator_for_literal(tree: DepTree, u: Unit, comps: List[Unit],
                          g: Optional[Grounding]) -> Tuple[str, Any]:
    """A literal defaults to '='; a nearby comparison cue overrides it."""
    val = (g.value if (g is not None and g.value is not None)
           else u.payload.get("value", u.text))
    for c in comps:
        if set(c.node_ids) & set(u.node_ids):
            return c.payload.get("operator", "="), _coerce(u.payload.get("value", u.text))
        if c.payload.get("value") is not None and str(c.payload["value"]) == str(u.text):
            return c.payload.get("operator", "="), _coerce(u.text)
    if u.payload.get("literal_type") == "number" or u.payload.get("ent_type") == "CARDINAL":
        return "=", _coerce(val)
    return "=", val


def _is_negated(tree: DepTree, u: Unit, negs: List[Unit]) -> bool:
    for n in negs:
        if set(n.node_ids) & set(u.node_ids):
            return True
    return False


def _column_for_comparison(tree: DepTree, cu: Unit, units: List[Unit],
                           groundings: Dict[int, Grounding]) -> Optional[str]:
    """Nearest grounded attribute mention to this comparison cue."""
    best, bestd = None, 10**9
    anchor = min(cu.node_ids) if cu.node_ids else 0
    for idx, u in enumerate(units):
        if u.kind not in {"attribute", "entity"}:
            continue
        g = groundings.get(idx)
        if not g or not g.column:
            continue
        d = min((abs(i - anchor) for i in (u.node_ids or [anchor])), default=10**9)
        if d < bestd:
            best, bestd = g.column, d
    return best


def _projection_columns(units: List[Unit], groundings: Dict[int, Grounding],
                        used: set) -> List[str]:
    """Columns the question asks to SEE (list/state/what is ...), not filter on."""
    out: List[str] = []
    for idx, u in enumerate(units):
        if idx in used or u.kind not in {"attribute", "entity"}:
            continue
        g = groundings.get(idx)
        if not g or not g.column:
            continue
        if u.payload.get("in_prep"):     # prep attachment => filter/relation, not projection
            continue
        if g.column not in out:
            out.append(g.column)
    return out


def _column_for_group(gu: Unit, units: List[Unit],
                      groundings: Dict[int, Grounding]) -> Optional[str]:
    for idx, u in enumerate(units):
        if u.kind in {"attribute", "entity"} and set(u.node_ids) & set(gu.node_ids):
            g = groundings.get(idx)
            if g and g.column:
                return g.column
    return None


def _column_for_order(ou: Unit, units: List[Unit], groundings: Dict[int, Grounding],
                      projections: List[str]) -> Optional[str]:
    anchor = min(ou.node_ids) if ou.node_ids else 0
    best, bestd = None, 10**9
    for idx, u in enumerate(units):
        if u.kind not in {"attribute", "entity"}:
            continue
        g = groundings.get(idx)
        if not g or not g.column:
            continue
        d = min((abs(i - anchor) for i in (u.node_ids or [anchor])), default=10**9)
        if d < bestd:
            best, bestd = g.column, d
    return best or (projections[0] if projections else None)


def _fallback_tables(schema: GroundedSchema, groundings: Dict[int, Grounding]) -> List[str]:
    ts = []
    for g in groundings.values():
        t = g.table or (g.column.split(".")[0] if g.column else None)
        if t and t not in ts:
            ts.append(t)
    return ts or schema.tables[:1]


_DISTINCT_RE = re.compile(r"\b(distinct|unique|different)\b", re.I)


def _wants_distinct(tree: DepTree) -> bool:
    return bool(_DISTINCT_RE.search(tree.question or ""))
