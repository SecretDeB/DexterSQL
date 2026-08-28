"""
Low-level, dependency-free SQLite helpers shared by every phase: schema
introspection, column-name tokenization, declared/validated foreign-key
loading, and the noise filters that keep Phase 1 (candidate_pairs.py) from
flooding the LLM triage step with structurally-explained pairs.

Every function here is pure SQL/Python -- no LLM calls anywhere in this
file. That split (deterministic evidence vs. LLM judgment) is the same
principle the rest of the pipeline follows: compute what can be computed,
only ask a model to interpret it.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def fetchall(conn: sqlite3.Connection, sql: str) -> List[Tuple]:
    cur = conn.execute(sql)
    try:
        return cur.fetchall()
    finally:
        cur.close()


def fetchone(conn: sqlite3.Connection, sql: str) -> Optional[Tuple]:
    cur = conn.execute(sql)
    try:
        return cur.fetchone()
    finally:
        cur.close()


def tokenize_column_name(name: str) -> Set[str]:
    """'aCL IgG' -> {'acl', 'igg'}; 'First Date' -> {'first', 'date'}."""
    parts = re.split(r"[\s_\-]+", name)
    tokens: Set[str] = set()
    for p in parts:
        sub = re.sub(r"([a-z])([A-Z])", r"\1 \2", p)
        sub = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", sub)
        for t in sub.split():
            t_lower = t.lower().strip()
            if t_lower:
                tokens.add(t_lower)
    return tokens


def load_schema(conn: sqlite3.Connection) -> Dict[str, List[Dict[str, Any]]]:
    tables = [
        r[0]
        for r in fetchall(
            conn,
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
    ]
    schema: Dict[str, List[Dict[str, Any]]] = {}
    for t in tables:
        cols = []
        for row in fetchall(conn, f"PRAGMA table_info({quote_ident(t)})"):
            _cid, name, typ, notnull, _dflt, pk = row
            cols.append({"name": name, "type": typ or "TEXT", "pk": pk > 0, "notnull": notnull > 0})
        schema[t] = cols
    return schema


def load_create_statements(conn: sqlite3.Connection) -> Dict[str, str]:
    rows = fetchall(
        conn,
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
    )
    return {name: sql for name, sql in rows}


def load_declared_fks(conn: sqlite3.Connection, schema: Dict) -> Dict[str, str]:
    """0a: PRAGMA foreign_key_list, taken as ground truth."""
    fk_map: Dict[str, str] = {}
    for table in schema:
        for row in fetchall(conn, f"PRAGMA foreign_key_list({quote_ident(table)})"):
            _, _, ref_table, from_col, to_col, *_ = row
            fk_map[f"{table}.{from_col}"] = f"{ref_table}.{to_col}"
    return fk_map


def validate_fk_containment(
    conn: sqlite3.Connection, child_key: str, parent_key: str, min_containment: float = 0.95
) -> bool:
    """The child column's non-NULL distinct values must be (almost) entirely
    contained in the parent column's values. Guards every non-declared FK
    edge (0b name-matched, 0c LLM-proposed) against a name coincidence
    that isn't actually a reference -- e.g. two unrelated tables that both
    happen to call their own primary key 'id'."""
    ct, cc = child_key.split(".", 1)
    pt, pc = parent_key.split(".", 1)
    try:
        row = fetchone(
            conn,
            f"SELECT "
            f"(SELECT COUNT(DISTINCT {quote_ident(cc)}) FROM {quote_ident(ct)} WHERE {quote_ident(cc)} IS NOT NULL), "
            f"(SELECT COUNT(DISTINCT {quote_ident(cc)}) FROM {quote_ident(ct)} "
            f"   WHERE {quote_ident(cc)} IS NOT NULL AND {quote_ident(cc)} IN "
            f"   (SELECT {quote_ident(pc)} FROM {quote_ident(pt)}))",
        )
    except Exception:
        return False
    if not row or not row[0]:
        return False
    n_child, n_contained = row[0], row[1] or 0
    return (n_contained / n_child) >= min_containment


def value_overlap_fraction(conn: sqlite3.Connection, col_a: str, col_b: str) -> float:
    """Case-insensitive distinct-value overlap as a fraction of the smaller
    column's distinct set. Fails open (returns 1.0) on a SQL error so a
    type-mismatch column pair isn't silently dropped -- let triage judge it
    instead of a query error."""
    ta, ca = col_a.split(".", 1)
    tb, cb = col_b.split(".", 1)
    qa, qta = quote_ident(ca), quote_ident(ta)
    qb, qtb = quote_ident(cb), quote_ident(tb)
    try:
        row = fetchone(
            conn,
            f"SELECT COUNT(*) FROM ("
            f"  SELECT DISTINCT UPPER(CAST({qa} AS TEXT)) v FROM {qta} WHERE {qa} IS NOT NULL"
            f"  INTERSECT"
            f"  SELECT DISTINCT UPPER(CAST({qb} AS TEXT)) v FROM {qtb} WHERE {qb} IS NOT NULL)",
        )
        inter = row[0] if row else 0
        da = fetchone(conn, f"SELECT COUNT(*) FROM (SELECT DISTINCT {qa} FROM {qta} WHERE {qa} IS NOT NULL)")
        db_ = fetchone(conn, f"SELECT COUNT(*) FROM (SELECT DISTINCT {qb} FROM {qtb} WHERE {qb} IS NOT NULL)")
        smaller = min(da[0] if da else 0, db_[0] if db_ else 0)
        if smaller == 0:
            return 0.0
        return inter / smaller
    except Exception:
        return 1.0


def gather_column_stats(conn: sqlite3.Connection, table: str, col: str, col_type: str) -> Dict[str, Any]:
    """Cheap per-column stats shown to the triage LLM (Phase 2) -- a lighter
    version of Phase 3's compute_value_domain, since triage only needs
    enough evidence to decide whether a pair is worth a full investigation."""
    qcol, qtab = quote_ident(col), quote_ident(table)
    stats: Dict[str, Any] = {"col_type": col_type}

    row = fetchone(conn, f"SELECT COUNT(*) FROM {qtab}")
    stats["row_count"] = row[0] if row else 0
    row = fetchone(conn, f"SELECT COUNT(*) FROM {qtab} WHERE {qcol} IS NULL")
    stats["null_count"] = row[0] if row else 0
    row = fetchone(conn, f"SELECT COUNT(*) FROM (SELECT DISTINCT {qcol} FROM {qtab})")
    stats["distinct_count"] = row[0] if row else 0

    if col_type.upper() in ("INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE"):
        try:
            row = fetchone(conn, f"SELECT MIN({qcol}), MAX({qcol}), AVG({qcol}) FROM {qtab} WHERE {qcol} IS NOT NULL")
            if row:
                stats["min"], stats["max"] = row[0], row[1]
                stats["avg"] = round(row[2], 2) if row[2] is not None else None
        except Exception:
            pass

    try:
        rows = fetchall(
            conn,
            f"SELECT {qcol}, COUNT(*) as cnt FROM {qtab} WHERE {qcol} IS NOT NULL "
            f"GROUP BY {qcol} ORDER BY cnt DESC LIMIT 15",
        )
        stats["value_samples"] = [{"value": str(r[0]), "count": r[1]} for r in rows]
    except Exception:
        stats["value_samples"] = []

    return stats


def build_noise_filter_sets(schema: Dict) -> Tuple[Set[str], Set[str], Set[str]]:
    """PK set, table-context-only skip set (dates/country), and ID-column
    set -- the inputs `passes_noise_filter` needs to drop structurally
    explained "look-alike" pairs before they ever cost an LLM call."""
    pk_set = {f"{table}.{c['name']}" for table, cols in schema.items() for c in cols if c["pk"]}
    skip_types = {"DATE", "DATETIME", "TIMESTAMP"}
    skip_name_tokens = {"date", "country"}
    skip_col_set: Set[str] = set()
    id_col_set: Set[str] = set()
    for table, cols in schema.items():
        for c in cols:
            key = f"{table}.{c['name']}"
            name_lower = c["name"].lower().strip()
            name_tokens = tokenize_column_name(c["name"])
            if c["type"].upper() in skip_types or (name_tokens & skip_name_tokens):
                skip_col_set.add(key)
            if name_lower == "id" or "id" in name_tokens or "uuid" in name_tokens:
                id_col_set.add(key)
    return pk_set, skip_col_set, id_col_set


def passes_noise_filter(
    a: str, b: str, pk_set: Set[str], skip_col_set: Set[str], id_col_set: Set[str], fk_map: Dict[str, str]
) -> bool:
    """True if (a, b) is NOT structural noise: PK-vs-PK, a known FK edge,
    both table-specific (date/country), either side an ID column, or
    sibling/direct FKs (same-parent or one-references-the-other)."""
    a_pk, b_pk = a in pk_set, b in pk_set

    if a_pk and b_pk:
        return False
    if a_pk and not b_pk and fk_map.get(b) == a:
        return False
    if b_pk and not a_pk and fk_map.get(a) == b:
        return False
    if a in skip_col_set and b in skip_col_set:
        return False
    if a in id_col_set or b in id_col_set:
        return False

    pa, pb = fk_map.get(a), fk_map.get(b)
    if pa is not None and pa == pb:
        return False
    if pa == b or pb == a:
        return False

    return True
