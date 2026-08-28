"""
Phase 0 -- Foreign-key discovery.

Builds a trustworthy fk_map ("child.col" -> "parent.col"). Every later phase
(candidate filtering, join-path discovery, coverage/fan-out/agreement) reads
this map, so a wrong edge poisons everything downstream -- hence every
non-declared edge is validated against the data before being added.

  0a. Declared FKs        -- PRAGMA foreign_key_list, taken as ground truth.
  0b. Name-matched FKs     -- same-named column as some table's PK, gated by
                              a sole-PK guard (two unrelated tables both
                              calling their key "id" must not be linked) and
                              a containment check.
  0c. LLM-implicit FKs     -- only runs if some PK still has no FK pointing
                              at it. The LLM proposes candidates from the
                              schema text; only high/medium-confidence
                              proposals are kept, and each must still pass
                              the same containment check.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, List, Optional

from .db_stats import fetchall, quote_ident, validate_fk_containment


def discover_foreign_keys(
    conn: sqlite3.Connection,
    schema: Dict[str, List[Dict[str, Any]]],
    create_statements: Dict[str, str],
    llm=None,
) -> Dict[str, str]:
    fk_map = _load_declared_fks(conn, schema)

    pk_map: Dict[str, List[str]] = {}
    for table, cols in schema.items():
        pks = [c["name"] for c in cols if c["pk"]]
        if pks:
            pk_map[table] = pks

    # 0b -- name-matched, validated by containment.
    for pk_table, pk_cols in pk_map.items():
        for pk_col in pk_cols:
            for other_table, other_cols in schema.items():
                if other_table == pk_table:
                    continue
                sole_pk = pk_map.get(other_table, [])
                for oc in other_cols:
                    key = f"{other_table}.{oc['name']}"
                    if key in fk_map or oc["name"].lower() != pk_col.lower():
                        continue
                    if oc["pk"] and len(sole_pk) == 1:
                        continue  # own single-column PK is identity, not a reference
                    if not validate_fk_containment(conn, key, f"{pk_table}.{pk_col}"):
                        continue
                    fk_map[key] = f"{pk_table}.{pk_col}"

    # 0c -- LLM-implicit, only for PKs still unreferenced, still containment-checked.
    if llm is not None:
        referenced_pks = set(fk_map.values())
        unmatched_pks = [
            f"{table}.{pk}" for table, pks in pk_map.items() for pk in pks if f"{table}.{pk}" not in referenced_pks
        ]
        if unmatched_pks:
            for fk_key, ref_key in _propose_implicit_fks(llm, create_statements, unmatched_pks):
                if fk_key in fk_map:
                    continue
                if validate_fk_containment(conn, fk_key, ref_key):
                    fk_map[fk_key] = ref_key

    return fk_map


def _load_declared_fks(conn: sqlite3.Connection, schema: Dict) -> Dict[str, str]:
    fk_map: Dict[str, str] = {}
    for table in schema:
        for row in fetchall(conn, f"PRAGMA foreign_key_list({quote_ident(table)})"):
            _, _, ref_table, from_col, to_col, *_ = row
            fk_map[f"{table}.{from_col}"] = f"{ref_table}.{to_col}"
    return fk_map


def _propose_implicit_fks(llm, create_statements: Dict[str, str], unmatched_pks: List[str]):
    schema_text = "\n\n".join(f"-- {t}\n{sql}" for t, sql in create_statements.items())
    prompt = (
        "You are analyzing a database schema to find implicit foreign key relationships.\n\n"
        f"DATABASE SCHEMA:\n{schema_text}\n\n"
        "The following primary keys have NO foreign key referencing them from any other table:\n"
        f"{', '.join(unmatched_pks)}\n\n"
        "For each unmatched PK, check if any column in another table could be an implicit "
        "foreign key (same data, acts as a reference, even if the name differs).\n\n"
        "Answer strictly as a JSON array. If no implicit FKs found, return [].\n"
        'Format: [{"fk": "table.column", "references": "table.pk_column", '
        '"confidence": "high|medium|low", "reason": "one line"}]'
    )
    try:
        responses, _usage = llm.ask(
            [{"role": "user", "content": prompt}],
            system_message={"role": "system", "content": "You are a database schema analyst. Return only valid JSON."},
            n=1,
            max_tokens=1024,
        )
        text = responses[0].content or ""
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        entries = json.loads(m.group())
    except Exception:
        return []

    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("confidence") not in ("high", "medium"):
            continue
        fk_key, ref_key = entry.get("fk"), entry.get("references")
        if fk_key and ref_key and "." in fk_key and "." in ref_key:
            out.append((fk_key, ref_key))
    return out
