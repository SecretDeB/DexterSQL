"""
Phase 3 -- Deep investigation. This is the "deep schema exploration" core:
for every flagged pair, compute the distributional and relational
statistics that expose the latent distinction a shared column name hides.
Pure SQL, no LLM. The branch depends on whether the two tables are
connected by a foreign key.

  - Join-path discovery: direct (one table FKs to the other), bridge (both
    FK to a common parent), or unconnected -- fk_map only, deterministic.
  - Always computed: value domains (row/null/distinct counts, numeric
    range, top values), value overlap between the two columns'
    distinct sets, and a categorical inventory for low-cardinality text
    columns.
  - Only when connected: coverage (what fraction of parent entities have
    child rows -- reveals a subset/partial table), fan-out (rows per key
    value, labeled with which table fans out -- ~1 = one record per
    entity, >>1 = per-event data), and agreement (join on the key, measure
    how often the two columns' values actually match on the same entity).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from .db_stats import fetchall, fetchone, quote_ident


def find_join_path(table_a: str, table_b: str, fk_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
    for child, parent in fk_map.items():
        child_table, parent_table = child.split(".")[0], parent.split(".")[0]
        if child_table == table_a and parent_table == table_b:
            return {"type": "direct", "from": child, "to": parent}
        if child_table == table_b and parent_table == table_a:
            return {"type": "direct", "from": child, "to": parent}

    a_parents, b_parents = {}, {}
    for child, parent in fk_map.items():
        child_table, parent_table = child.split(".")[0], parent.split(".")[0]
        if child_table == table_a:
            a_parents[parent_table] = (child, parent)
        if child_table == table_b:
            b_parents[parent_table] = (child, parent)

    common = set(a_parents) & set(b_parents)
    if common:
        bridge = sorted(common)[0]
        return {
            "type": "bridge",
            "bridge_table": bridge,
            "path": [
                {"from": a_parents[bridge][0], "to": a_parents[bridge][1]},
                {"from": b_parents[bridge][0], "to": b_parents[bridge][1]},
            ],
        }
    return None


def compute_value_domain(conn: sqlite3.Connection, table: str, col: str, col_type: str) -> Dict[str, Any]:
    qcol, qtab = quote_ident(col), quote_ident(table)
    domain: Dict[str, Any] = {}

    row = fetchone(conn, f"SELECT COUNT(*) FROM {qtab}")
    domain["row_count"] = row[0] if row else 0
    row = fetchone(conn, f"SELECT COUNT(*) FROM {qtab} WHERE {qcol} IS NULL")
    domain["null_count"] = row[0] if row else 0
    row = fetchone(conn, f"SELECT COUNT(*) FROM (SELECT DISTINCT {qcol} FROM {qtab})")
    domain["distinct_count"] = row[0] if row else 0

    if col_type.upper() in ("INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE"):
        try:
            row = fetchone(conn, f"SELECT MIN({qcol}), MAX({qcol}), AVG({qcol}) FROM {qtab} WHERE {qcol} IS NOT NULL")
            if row:
                domain["min"], domain["max"] = row[0], row[1]
                domain["avg"] = round(row[2], 2) if row[2] is not None else None
        except Exception:
            pass

    try:
        rows = fetchall(
            conn,
            f"SELECT {qcol}, COUNT(*) as cnt FROM {qtab} WHERE {qcol} IS NOT NULL "
            f"GROUP BY {qcol} ORDER BY cnt DESC LIMIT 20",
        )
        domain["top_20"] = [{"value": str(r[0]), "count": r[1]} for r in rows]
    except Exception:
        domain["top_20"] = []

    return domain


def compute_value_overlap(conn: sqlite3.Connection, table_a: str, col_a: str, table_b: str, col_b: str) -> Dict[str, Any]:
    qa, qta = quote_ident(col_a), quote_ident(table_a)
    qb, qtb = quote_ident(col_b), quote_ident(table_b)

    try:
        row = fetchone(
            conn,
            f"SELECT COUNT(*) FROM ("
            f"  SELECT DISTINCT {qa} FROM {qta} WHERE {qa} IS NOT NULL"
            f"  INTERSECT SELECT DISTINCT {qb} FROM {qtb} WHERE {qb} IS NOT NULL)",
        )
        overlap_exact = row[0] if row else 0
    except Exception:
        overlap_exact = 0
    try:
        row = fetchone(
            conn,
            f"SELECT COUNT(*) FROM ("
            f"  SELECT DISTINCT UPPER({qa}) FROM {qta} WHERE {qa} IS NOT NULL"
            f"  INTERSECT SELECT DISTINCT UPPER({qb}) FROM {qtb} WHERE {qb} IS NOT NULL)",
        )
        overlap_ci = row[0] if row else 0
    except Exception:
        overlap_ci = 0

    row_a = fetchone(conn, f"SELECT COUNT(*) FROM (SELECT DISTINCT {qa} FROM {qta})")
    row_b = fetchone(conn, f"SELECT COUNT(*) FROM (SELECT DISTINCT {qb} FROM {qtb})")
    dist_a, dist_b = (row_a[0] if row_a else 1), (row_b[0] if row_b else 1)

    return {
        "overlap_exact": overlap_exact,
        "overlap_case_insensitive": overlap_ci,
        "distinct_a": dist_a,
        "distinct_b": dist_b,
        "overlap_pct_of_a": round(overlap_exact / max(dist_a, 1) * 100, 1),
        "overlap_pct_of_b": round(overlap_exact / max(dist_b, 1) * 100, 1),
    }


def compute_coverage(conn: sqlite3.Connection, parent_table: str, parent_col: str, child_table: str, child_col: str) -> Dict[str, Any]:
    qp_col, qp_tab = quote_ident(parent_col), quote_ident(parent_table)
    qc_col, qc_tab = quote_ident(child_col), quote_ident(child_table)

    row = fetchone(conn, f"SELECT COUNT(DISTINCT {qp_col}) FROM {qp_tab}")
    parent_total = row[0] if row else 0
    row = fetchone(conn, f"SELECT COUNT(DISTINCT {qc_col}) FROM {qc_tab}")
    child_distinct = row[0] if row else 0

    return {
        "parent_table": parent_table, "parent_col": parent_col, "parent_total": parent_total,
        "child_table": child_table, "child_col": child_col, "child_distinct": child_distinct,
        "coverage_pct": round(child_distinct / max(parent_total, 1) * 100, 1),
    }


def compute_fan_out(conn: sqlite3.Connection, child_table: str, child_col: str) -> Dict[str, Any]:
    """Average/max rows per key value in `child_table`, i.e. how many rows
    of this table share one key -- the granularity signal: ~1 = one row
    per entity (a "final"/master value), >>1 = per-event records."""
    qc_col, qc_tab = quote_ident(child_col), quote_ident(child_table)
    base = {"table": child_table, "column": child_col}
    row = fetchone(
        conn,
        f"SELECT AVG(cnt), MAX(cnt) FROM ("
        f"  SELECT {qc_col}, COUNT(*) as cnt FROM {qc_tab} WHERE {qc_col} IS NOT NULL GROUP BY {qc_col})",
    )
    if row and row[0] is not None:
        return {**base, "avg_rows_per_parent": round(row[0], 1), "max_rows_per_parent": row[1]}
    return {**base, "avg_rows_per_parent": None, "max_rows_per_parent": None}


def compute_agreement(
    conn: sqlite3.Connection, table_a: str, col_a: str, table_b: str, col_b: str, join_col_a: str, join_col_b: str
) -> Dict[str, Any]:
    """Join the two tables on the key and measure how often the two
    columns' values match on the same entity. Low agreement means the
    columns capture different things despite a shared name; the
    exact-vs-case-insensitive gap exposes casing traps."""
    qta, qtb = quote_ident(table_a), quote_ident(table_b)
    qca, qcb = quote_ident(col_a), quote_ident(col_b)
    qja, qjb = quote_ident(join_col_a), quote_ident(join_col_b)

    try:
        row = fetchone(
            conn,
            f"SELECT COUNT(*) as total,"
            f"  SUM(CASE WHEN a.{qca} = b.{qcb} THEN 1 ELSE 0 END) as exact_match,"
            f"  SUM(CASE WHEN UPPER(a.{qca}) = UPPER(b.{qcb}) THEN 1 ELSE 0 END) as case_match"
            f" FROM {qta} a JOIN {qtb} b ON a.{qja} = b.{qjb}"
            f" WHERE a.{qca} IS NOT NULL AND b.{qcb} IS NOT NULL",
        )
        total = row[0] if row else 0
        exact = (row[1] or 0) if row else 0
        case_m = (row[2] or 0) if row else 0
    except Exception as e:
        return {"error": str(e)}

    mismatches = []
    try:
        rows = fetchall(
            conn,
            f"SELECT a.{qja}, a.{qca}, b.{qcb} FROM {qta} a JOIN {qtb} b ON a.{qja} = b.{qjb}"
            f" WHERE a.{qca} IS NOT NULL AND b.{qcb} IS NOT NULL AND a.{qca} != b.{qcb} LIMIT 5",
        )
        mismatches = [{"id": str(r[0]), "a": str(r[1]), "b": str(r[2])} for r in rows]
    except Exception:
        pass

    return {
        "total_overlap": total, "exact_match": exact,
        "exact_match_pct": round(exact / max(total, 1) * 100, 1),
        "case_insensitive_match": case_m,
        "case_insensitive_pct": round(case_m / max(total, 1) * 100, 1),
        "sample_mismatches": mismatches,
    }


def compute_categorical_inventory(conn: sqlite3.Connection, table: str, col: str) -> Optional[List[Dict[str, Any]]]:
    """Full value inventory for text columns with < 50 distinct values."""
    qcol, qtab = quote_ident(col), quote_ident(table)
    row = fetchone(conn, f"SELECT COUNT(*) FROM (SELECT DISTINCT {qcol} FROM {qtab})")
    distinct = row[0] if row else 0
    if distinct == 0 or distinct > 50:
        return None
    rows = fetchall(
        conn, f"SELECT {qcol}, COUNT(*) as cnt FROM {qtab} WHERE {qcol} IS NOT NULL GROUP BY {qcol} ORDER BY cnt DESC"
    )
    return [{"value": str(r[0]), "count": r[1]} for r in rows]


def investigate_pair(conn: sqlite3.Connection, pair: Dict[str, Any], fk_map: Dict[str, str], schema: Dict) -> Dict[str, Any]:
    col_a, col_b = pair["col_a"], pair["col_b"]
    table_a, cname_a = col_a.split(".", 1)
    table_b, cname_b = col_b.split(".", 1)

    def col_type(table, col):
        return next((c["type"] for c in schema.get(table, []) if c["name"] == col), "TEXT")

    type_a, type_b = col_type(table_a, cname_a), col_type(table_b, cname_b)

    result: Dict[str, Any] = {"pair": [col_a, col_b], "flag_strength": pair.get("flag_strength", "hard")}
    result["value_domain"] = {
        col_a: compute_value_domain(conn, table_a, cname_a, type_a),
        col_b: compute_value_domain(conn, table_b, cname_b, type_b),
    }
    result["value_overlap"] = compute_value_overlap(conn, table_a, cname_a, table_b, cname_b)

    join_path = find_join_path(table_a, table_b, fk_map)
    result["fk_path"] = join_path

    if join_path is not None:
        result["connected"] = True
        if join_path["type"] == "direct":
            from_key, to_key = join_path["from"], join_path["to"]
            from_table, from_col = from_key.split(".", 1)
            to_table, to_col = to_key.split(".", 1)

            result["coverage"] = [compute_coverage(conn, to_table, to_col, from_table, from_col)]
            result["fan_out"] = [compute_fan_out(conn, from_table, from_col)]
            if from_table == table_a:
                result["agreement"] = compute_agreement(conn, table_a, cname_a, table_b, cname_b, from_col, to_col)
            else:
                result["agreement"] = compute_agreement(conn, table_a, cname_a, table_b, cname_b, to_col, from_col)

        else:  # bridge
            bridge = join_path["bridge_table"]
            path = join_path["path"]
            coverages, fan_outs = [], []
            for leg in path:
                from_table_l, from_col_l = leg["from"].split(".", 1)
                to_table_l, to_col_l = leg["to"].split(".", 1)
                coverages.append(compute_coverage(conn, to_table_l, to_col_l, from_table_l, from_col_l))
                fan_outs.append(compute_fan_out(conn, from_table_l, from_col_l))
            result["coverage"] = coverages
            result["fan_out"] = fan_outs

            a_fk_col = path[0]["from"].split(".", 1)[1]
            b_fk_col = path[1]["from"].split(".", 1)[1]
            try:
                row = fetchone(
                    conn,
                    f"SELECT COUNT(*) FROM ("
                    f"  SELECT DISTINCT {quote_ident(a_fk_col)} FROM {quote_ident(table_a)}"
                    f"  INTERSECT SELECT DISTINCT {quote_ident(b_fk_col)} FROM {quote_ident(table_b)})",
                )
                result["cross_coverage"] = row[0] if row else 0
            except Exception:
                result["cross_coverage"] = None
            result["agreement"] = compute_agreement(conn, table_a, cname_a, table_b, cname_b, a_fk_col, b_fk_col)
    else:
        result["connected"] = False
        result["coverage"] = None
        result["fan_out"] = None
        result["agreement"] = None

    inv_a = compute_categorical_inventory(conn, table_a, cname_a)
    inv_b = compute_categorical_inventory(conn, table_b, cname_b)
    result["categorical_inventory"] = {col_a: inv_a, col_b: inv_b} if (inv_a or inv_b) else None

    return result
