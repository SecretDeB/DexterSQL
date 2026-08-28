"""
Phase 4 -- LLM verdict & note.

Distills one pair's Phase-3 evidence bundle (+ official BIRD
database_description docs, if present) into a short, self-contained
disambiguation note. The note is what gets shown to the SQL-writing model
later (gate.py); the raw statistics never leave this offline pipeline --
the design principle is that all heavy statistics are reasoned over once,
here, so the online gate only ever has to render distilled text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def _build_verdict_prompt(investigation: Dict[str, Any], dev_docs: Optional[Dict[str, Dict]] = None) -> str:
    pair = investigation["pair"]
    connected = investigation["connected"]
    sections = [
        f"COLUMN PAIR: {pair[0]}  vs  {pair[1]}",
        f"Connected: {'yes' if connected else 'no (no FK path between tables)'}",
    ]

    if dev_docs:
        for col_key in pair:
            doc = dev_docs.get(col_key, {})
            desc, val_desc = doc.get("column_description", ""), doc.get("value_description", "")
            if desc or val_desc:
                line = f"\nDocumentation for {col_key}: {desc}"
                if val_desc:
                    line += f" (values: {val_desc})"
                sections.append(line)

    for col_key, domain in investigation.get("value_domain", {}).items():
        s = f"\n{col_key}:\n  Rows: {domain.get('row_count')}, Nulls: {domain.get('null_count')}"
        s += f"\n  Distinct: {domain.get('distinct_count')}"
        if "min" in domain:
            s += f"\n  Range: {domain['min']} to {domain['max']}, avg={domain.get('avg')}"
        top = domain.get("top_20", [])
        if top:
            s += "\n  Top values: " + ", ".join(f"{v['value']} ({v['count']})" for v in top[:15])
        sections.append(s)

    vo = investigation.get("value_overlap")
    if vo:
        sections.append(
            f"\nValue overlap:\n  Exact: {vo['overlap_exact']} values shared "
            f"({vo['overlap_pct_of_a']}% of A, {vo['overlap_pct_of_b']}% of B)"
            f"\n  Case-insensitive: {vo['overlap_case_insensitive']} values shared"
        )

    fp = investigation.get("fk_path")
    if fp:
        if fp["type"] == "direct":
            sections.append(f"\nFK path: {fp['from']} -> {fp['to']} (direct)")
        else:
            legs = " -> ".join(f"{l['from']} -> {l['to']}" for l in fp["path"])
            sections.append(f"\nFK path: bridge via {fp['bridge_table']}: {legs}")

    cov = investigation.get("coverage")
    if cov:
        for c in cov:
            sections.append(
                f"\nCoverage ({c['parent_table']} -> {c['child_table']}): "
                f"{c['child_distinct']}/{c['parent_total']} = {c['coverage_pct']}%"
            )
    cross = investigation.get("cross_coverage")
    if cross is not None:
        sections.append(f"\nCross-coverage (entities in both tables): {cross}")

    fo = investigation.get("fan_out")
    if fo:
        for f in fo:
            label = f"{f.get('table')}.{f.get('column')}" if f.get("table") else "n/a"
            sections.append(
                f"\nFan-out of {label}: avg {f['avg_rows_per_parent']} rows per key value, "
                f"max {f['max_rows_per_parent']} (>1 means this table has MULTIPLE rows per "
                f"key -- per-event granularity; ~1 means one row per key)"
            )

    agr = investigation.get("agreement")
    if agr and "error" not in agr:
        sections.append(
            f"\nAgreement (same-ID value match):\n  Total overlapping entities: {agr['total_overlap']}"
            f"\n  Exact match: {agr['exact_match']} ({agr['exact_match_pct']}%)"
            f"\n  Case-insensitive: {agr['case_insensitive_match']} ({agr['case_insensitive_pct']}%)"
        )
        if agr.get("sample_mismatches"):
            mm = "\n  ".join(f"ID {m['id']}: A=\"{m['a']}\", B=\"{m['b']}\"" for m in agr["sample_mismatches"])
            sections.append(f"\n  Sample mismatches:\n  {mm}")

    cat = investigation.get("categorical_inventory")
    if cat:
        for col_key, inv in cat.items():
            if inv:
                vals = ", ".join(f"'{v['value']}' ({v['count']})" for v in inv[:30])
                sections.append(f"\nCategorical inventory for {col_key}:\n  {vals}")

    analysis_text = "\n".join(sections)

    return f"""You are analyzing two database columns that were flagged as potentially ambiguous for SQL generation (text-to-SQL).

Based on the statistical evidence below, write a short, self-contained
disambiguation note. The note will later be shown to a SQL-writing model
WITHOUT any of these statistics, so it must capture the semantic
distinction in plain language. Ground every claim in the evidence
(fan-out, coverage, agreement, value domains) -- do not invent semantics
the data does not support.

ANALYSIS:
{analysis_text}

Reason about:
1. Are these the same concept, different concepts, or overlapping? Weigh ALL
   the evidence: sparsity (a high-NULL column vs. a fully-populated
   counterpart suggests optional/per-event data vs. a mandatory master
   attribute), coverage (if only a fraction of parent entities have rows in
   the other table, that table records a subset, not the authoritative
   value), fan-out (~1 row per key = one record per entity/final value; >>1
   = per-event records), agreement (low agreement on joined rows = the
   columns capture different things despite the shared name).
2. What is each column's purpose, in one short sentence?
3. When should a SQL query use each one?
4. Pitfalls (wrong joins, missing rows due to partial coverage, casing,
   wrong filter values)?

Output ONLY a JSON object with this exact structure:
{{
    "columns": ["{pair[0]}", "{pair[1]}"],
    "verdict": "same_concept" | "different_concept" | "overlapping",
    "col_a_purpose": "one sentence: what {pair[0]} records",
    "col_b_purpose": "one sentence: what {pair[1]} records",
    "use_a_when": "one sentence: question patterns where {pair[0]} is correct",
    "use_b_when": "one sentence: question patterns where {pair[1]} is correct",
    "sql_note": "concise guidance for text-to-SQL, max 2 sentences",
    "caution": "ONLY for same_concept pairs with partial coverage or imperfect agreement: one sentence on the trap. Otherwise null.",
    "pitfalls": ["pitfall 1", "pitfall 2"]
}}

Output ONLY the JSON, no other text."""


def _parse_verdict_json(response: str) -> Optional[Dict[str, Any]]:
    if not response:
        return None
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "columns" not in obj or "verdict" not in obj:
        return None
    return obj


def synthesize_verdict(investigation: Dict[str, Any], llm, dev_docs: Optional[Dict[str, Dict]] = None) -> Optional[Dict[str, Any]]:
    """One LLM call -> a note dict, or None if the call/parse failed."""
    prompt = _build_verdict_prompt(investigation, dev_docs)
    try:
        responses, _usage = llm.ask([{"role": "user", "content": prompt}], n=1, max_tokens=2048)
        return _parse_verdict_json(responses[0].content or "")
    except Exception:
        return None
