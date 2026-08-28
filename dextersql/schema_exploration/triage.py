"""
Phase 2 -- LLM triage.

Decides which Phase-1 candidates are REFERENTIALLY ambiguous: a realistic
question could plausibly map to either column, so a text-to-SQL model might
silently pick the wrong one. This is narrower than "these columns are
related" -- two different lab tests, or a code vs. its description, are
related but not ambiguous, and _AMBIGUITY_DEF exists specifically to stop a
model from flagging those.

Simplified from the reference this module is based on, which ran three
prompt variants per pair (no schema / filtered schema / full schema) and
took a majority vote to assign HARD (3/3) vs. SOFT (2/3, or maybe-only)
confidence. This version asks once, with the full schema, and maps
yes -> HARD, maybe -> SOFT, no -> SKIP. That's a 3x cheaper triage pass at
the cost of the extra precision the multi-variant consensus bought; if
triage precision turns out to matter, restoring the 3-variant vote here is
the natural next step.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from .db_stats import gather_column_stats

_AMBIGUITY_DEF = (
    "We are screening for REFERENTIAL AMBIGUITY between two columns: a case "
    "where a realistic natural-language question could plausibly refer to "
    "EITHER column, so a text-to-SQL model might pick the wrong one and "
    "silently return an incorrect result.\n"
    "Answer 'yes' ONLY if BOTH hold:\n"
    "  (1) the columns represent the SAME or overlapping real-world concept "
    "(e.g. two 'diagnosis' fields, two 'name' fields), AND\n"
    "  (2) some plausible user phrasing would not clearly disambiguate which "
    "column is meant.\n"
    "Answer 'no' if the columns are merely related or from the same domain "
    "but represent DISTINCT, separately-named things that a question would "
    "refer to specifically (e.g. two different lab tests, two different "
    "measurements, a code vs its description). Relatedness is NOT ambiguity."
)


def _col_stats_block(col_key: str, stats: Dict[str, Any], dev_docs: Dict[str, Dict]) -> str:
    s = stats
    block = f"Column: {col_key}\n  Type: {s.get('col_type', 'unknown')}\n"
    block += f"  Row count: {s['row_count']}\n  Distinct values: {s['distinct_count']}\n"
    if "min" in s:
        block += f"  Range: {s['min']} to {s['max']}, avg={s.get('avg')}\n"
    samples = s.get("value_samples", [])
    if samples:
        block += "  Top values: " + ", ".join(f"{v['value']} ({v['count']})" for v in samples[:15]) + "\n"
    doc = dev_docs.get(col_key, {})
    if doc.get("column_description"):
        block += f"  Description: {doc['column_description']}\n"
    if doc.get("value_description"):
        block += f"  Value info: {doc['value_description']}\n"
    return block


def _build_triage_prompt(
    pair: Dict[str, Any], stats: Dict[str, Dict], dev_docs: Dict[str, Dict],
    create_statements: Dict[str, str], fk_map: Dict[str, str],
) -> str:
    col_a, col_b = pair["col_a"], pair["col_b"]
    schema_text = "\n\n".join(f"{sql};" for sql in create_statements.values())
    fk_text = "\n".join(f"  {k} -> {v}" for k, v in sorted(fk_map.items())) or "  (none)"
    return (
        "You are analyzing two database columns for text-to-SQL.\n\n"
        f"{_AMBIGUITY_DEF}\n\n"
        f"FULL DATABASE SCHEMA:\n{schema_text}\n\n"
        f"ALL FK RELATIONSHIPS:\n{fk_text}\n\n"
        f"{_col_stats_block(col_a, stats[col_a], dev_docs)}\n"
        f"{_col_stats_block(col_b, stats[col_b], dev_docs)}\n"
        "Given the full database context, are these two columns referentially "
        "ambiguous as defined above?\n\n"
        'Answer strictly as JSON:\n{"verdict": "yes" | "no" | "maybe", "reason": "one line explanation"}'
    )


def _parse_verdict(response: str) -> Dict[str, str]:
    try:
        m = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            if isinstance(parsed, dict) and parsed.get("verdict") in ("yes", "no", "maybe"):
                return parsed
    except json.JSONDecodeError:
        pass
    lower = (response or "").lower()
    if '"yes"' in lower:
        return {"verdict": "yes", "reason": response[:200]}
    if '"no"' in lower:
        return {"verdict": "no", "reason": response[:200]}
    return {"verdict": "maybe", "reason": response[:200]}


def triage_pairs(
    candidate_pairs: List[Dict[str, Any]],
    conn: sqlite3.Connection,
    schema: Dict[str, List[Dict[str, Any]]],
    dev_docs: Dict[str, Dict],
    create_statements: Dict[str, str],
    fk_map: Dict[str, str],
    llm,
) -> List[Dict[str, Any]]:
    """Returns flagged_pairs: each candidate with a `flag_strength` of
    "hard" or "soft" appended; pairs judged "no" are dropped."""
    if not candidate_pairs:
        return []

    stats_cache: Dict[str, Dict[str, Any]] = {}

    def stats_for(col_key: str) -> Dict[str, Any]:
        if col_key not in stats_cache:
            table, col = col_key.split(".", 1)
            col_type = next(
                (c["type"] for c in schema.get(table, []) if c["name"] == col), "TEXT"
            )
            stats_cache[col_key] = gather_column_stats(conn, table, col, col_type)
        return stats_cache[col_key]

    flagged = []
    for pair in candidate_pairs:
        stats = {pair["col_a"]: stats_for(pair["col_a"]), pair["col_b"]: stats_for(pair["col_b"])}
        prompt = _build_triage_prompt(pair, stats, dev_docs, create_statements, fk_map)
        try:
            responses, _usage = llm.ask([{"role": "user", "content": prompt}], n=1, max_tokens=512)
            verdict = _parse_verdict(responses[0].content or "")
        except Exception:
            verdict = {"verdict": "maybe", "reason": "LLM call failed"}

        if verdict["verdict"] == "no":
            continue
        strength = "hard" if verdict["verdict"] == "yes" else "soft"
        flagged.append({**pair, "flag_strength": strength, "triage_reason": verdict.get("reason", "")})

    return flagged
