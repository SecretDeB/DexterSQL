"""
Steps 7 & 9: IR validation and SQL-vs-IR AST validation. DETERMINISTIC.

§7 requires the generated SQL to PRESERVE the IR, not merely be inspired by it.
`validate_sql` implements every listed check against the sqlglot AST:
  1. every IR table and required join occurs in the AST
  2. every IR predicate appears with the correct column, operator, value
  3. aggregates / GROUP BY / HAVING / ORDER BY / LIMIT agree with the IR
  4. every grounded question unit has an SQL realization (coverage)
  5. no ungrounded table, column, literal, or condition was introduced
  6. the join graph is connected through valid keys
  7. the query parses for the target dialect

Violations are returned as human-readable strings that feed the targeted repair
prompt verbatim (§8), so the repair agent sees exactly what failed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .ir import SQLIR
from .grounding import GroundedSchema, find_join_path

try:
    import sqlglot
    from sqlglot import exp
    _HAS_SQLGLOT = True
except Exception:                                   # pragma: no cover
    _HAS_SQLGLOT = False


# ── §7 first half: is the IR itself coherent? ────────────────────────────────
def validate_ir(ir: SQLIR, schema: GroundedSchema,
                required_units: Optional[List[str]] = None) -> List[str]:
    v: List[str] = []
    if ir.is_empty():
        v.append("IR is empty: no SELECT, no filters, no tables.")
        return v
    if not ir.select:
        v.append("IR has no SELECT items.")
    if not ir.tables:
        v.append("IR has no tables.")

    known_cols = {c.qualified.lower() for c in schema.columns}
    known_tabs = {t.lower() for t in schema.tables}
    # all_columns() already strips aggregate wrappers; do NOT split on '(' here —
    # BIRD column names contain parentheses, e.g. 'frpm.FRPM Count (Ages 5-17)'.
    for q in ir.all_columns():
        base = (q or "").strip()
        if base.endswith("*"):
            continue
        if base.lower() not in known_cols and base.lower() not in known_tabs:
            v.append(f"IR references unknown schema element '{q}'.")
    # A table pulled in purely as a JOIN waypoint is legitimate: the FK path may
    # need a bridge table the linker did not surface. Flag only tables that carry
    # no join edge (i.e. genuinely invented).
    join_tabs = {t.split(".")[0].lower() for j in ir.joins for t in (j.left, j.right)}
    for t in ir.tables:
        if t.lower() not in known_tabs and t.lower() not in join_tabs:
            v.append(f"IR references unknown table '{t}'.")
    for f in ir.filters + ir.having:
        if not f.is_valid():
            v.append(f"IR filter on '{f.column}' has invalid operator '{f.operator}'.")

    _, connected = find_join_path(schema, ir.tables)
    if not connected:
        v.append(f"IR tables {ir.tables} are not connected by any FK path.")

    if ir.group_by:
        for s in ir.select:
            e = s.expression
            if "(" not in e and e not in ir.group_by:
                v.append(f"SELECT '{e}' is neither aggregated nor in GROUP BY.")
    if required_units:
        missing = [u for u in required_units if u.strip().lower() not in
                   " ".join(ir.coverage).lower()]
        if missing:
            v.append(f"IR coverage omits question element(s): {missing}")
    return v


# ── §9: does the SQL preserve the IR? ────────────────────────────────────────
def validate_sql(sql: str, ir: SQLIR, schema: GroundedSchema,
                 dialect: str = "sqlite") -> List[str]:
    v: List[str] = []
    if not (sql or "").strip():
        return ["SQL is empty."]
    if not _HAS_SQLGLOT:
        return []                       # cannot verify; caller degrades to "guides"

    # (7) syntactic validity for the dialect
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception as e:
        return [f"SQL does not parse for dialect {dialect}: {str(e)[:200]}"]
    if tree is None:
        return ["SQL parsed to an empty tree."]

    sql_tables = _sql_tables(tree)
    sql_cols = _sql_columns(tree)
    lowered = sql.lower()

    # (1) IR tables present
    for t in ir.tables:
        if t.lower() not in {x.lower() for x in sql_tables}:
            v.append(f"IR table '{t}' does not appear in the SQL.")
    # required joins present (either as JOIN ... ON or as a WHERE equality)
    for j in ir.joins:
        lc, rc = j.left.split(".")[-1].lower(), j.right.split(".")[-1].lower()
        if not (lc in lowered and rc in lowered):
            v.append(f"IR join {j.left} = {j.right} is missing from the SQL.")

    # (2) predicates preserved
    for f in ir.filters:
        col = f.column.split(".")[-1].lower()
        if col not in lowered:
            v.append(f"IR filter column '{f.column}' is missing from the SQL.")
            continue
        if f.value is not None and not _value_present(lowered, f.value):
            v.append(f"IR filter value '{f.value}' (on {f.column}) is missing from the SQL.")
        if f.operator not in ("=", "IS NULL", "IS NOT NULL") and \
                f.operator.lower() not in lowered.replace("<>", "!="):
            v.append(f"IR operator '{f.operator}' on {f.column} is missing from the SQL.")

    # (3) aggregates / grouping / ordering / limit
    for s in ir.select:
        m = re.match(r"\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(", s.expression, re.I)
        if m and m.group(1).lower() not in lowered:
            v.append(f"IR aggregate '{m.group(1)}' is missing from the SQL.")
    for g in ir.group_by:
        if "group by" not in lowered:
            v.append("IR requires GROUP BY but the SQL has none.")
            break
        if g.split(".")[-1].lower() not in lowered:
            v.append(f"IR GROUP BY column '{g}' is missing from the SQL.")
    if ir.having and "having" not in lowered:
        v.append("IR requires HAVING but the SQL has none.")
    for o in ir.order_by:
        if "order by" not in lowered:
            v.append("IR requires ORDER BY but the SQL has none.")
            break
        if o.expression.split(".")[-1].lower() not in lowered:
            v.append(f"IR ORDER BY expression '{o.expression}' is missing from the SQL.")
        if o.direction == "DESC" and "desc" not in lowered:
            v.append("IR requires descending order but the SQL does not use DESC.")
    if ir.limit is not None and not re.search(rf"limit\s+{ir.limit}\b", lowered):
        v.append(f"IR requires LIMIT {ir.limit} but the SQL does not have it.")
    if ir.distinct and "distinct" not in lowered:
        v.append("IR requires DISTINCT but the SQL does not use it.")

    # (5) nothing ungrounded introduced
    known_tabs = {t.lower() for t in schema.tables}
    # A table that only serves as a JOIN WAYPOINT is legitimate: linking satscores
    # to frpm genuinely requires `schools` as a bridge. Rejecting it discarded
    # correct SQL, so only flag tables that are not FK-reachable at all.
    fk_tabs = {e.split(".")[0].lower() for pair in schema.fks for e in pair}
    for t in sql_tables:
        if t.lower() not in known_tabs and t.lower() not in fk_tabs:
            v.append(f"SQL introduces table '{t}' that is not in the focused schema.")
    known_cols = {c.column.lower() for c in schema.columns}
    for c in sql_cols:
        if c.lower() not in known_cols and c != "*":
            v.append(f"SQL introduces column '{c}' that is not in the focused schema.")

    # (6) join graph connected through valid keys
    if len(sql_tables) > 1:
        _, connected = find_join_path(schema, list(sql_tables))
        if not connected:
            v.append(f"SQL joins tables {sorted(sql_tables)} with no valid FK path.")
        # A multi-table query with no ON clause and no cross-table equality is a
        # CARTESIAN PRODUCT ("FROM Patient, Examination"), which silently returns
        # garbage rather than erroring — the IR checks above cannot see it.
        if not _has_join_condition(tree):
            v.append(f"SQL selects from {sorted(sql_tables)} with no join condition "
                     f"(cartesian product): add an explicit join on the key columns.")

    return v


def _has_join_condition(tree) -> bool:
    """True if the query joins its tables via ON/USING or a cross-table equality."""
    for j in tree.find_all(exp.Join):
        if j.args.get("on") is not None or j.args.get("using"):
            return True
    for eq in tree.find_all(exp.EQ):
        l, r = eq.left, eq.right
        if isinstance(l, exp.Column) and isinstance(r, exp.Column):
            lt = l.table or ""
            rt = r.table or ""
            if lt and rt and lt.lower() != rt.lower():
                return True
    return False


# ── AST helpers ──────────────────────────────────────────────────────────────
def _sql_tables(tree) -> Set[str]:
    out = set()
    for t in tree.find_all(exp.Table):
        name = t.name or (t.this.name if hasattr(t.this, "name") else None)
        if name:
            out.add(name)
    return out


def _sql_columns(tree) -> Set[str]:
    out = set()
    for c in tree.find_all(exp.Column):
        if c.name:
            out.add(c.name)
    return out


def _value_present(lowered_sql: str, value: Any) -> bool:
    s = str(value).strip().strip("'\"").lower()
    if not s:
        return True
    if s in lowered_sql:
        return True
    try:                                   # 3.5 vs 3.50 / 1930 vs '1930'
        f = float(s)
        return bool(re.search(rf"\b{re.escape(str(int(f)) if f.is_integer() else str(f))}\b",
                              lowered_sql))
    except ValueError:
        return False


# ── convenience: is the SQL executable at all? ───────────────────────────────
def dry_run(sql: str, db_path: str, timeout_s: float = 5.0) -> Optional[str]:
    """Execute with LIMIT 0-ish semantics to catch runtime errors. Returns an error
    string or None. Read-only connection; never mutates the benchmark DB."""
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
        try:
            con.execute(f"SELECT * FROM ({sql.rstrip().rstrip(';')}) LIMIT 1").fetchall()
            return None
        finally:
            con.close()
    except Exception as e:
        return str(e)[:300]
