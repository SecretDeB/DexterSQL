"""
Prompt builders for the three narrowly-scoped LLM roles (handoff §5, §6, §8).

Role boundaries (§10) — the LLM may RESOLVE and RENDER the decomposition, but must
not silently redefine it:
  * IR completion : fills `unresolved` only; structure is fixed input.
  * SQL renderer  : converts a validated IR to SQL; adds nothing.
  * Repair agent  : fixes only the listed validation violations.

All responses are wrapped in <result>...</result> so a simple rule-based
extractor can recover the payload from a chatty model.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .ir import SQLIR


# ── shared schema rendering ──────────────────────────────────────────────────
def render_schema(database_schema: Dict[str, Any], max_values: int = 6) -> str:
    lines = []
    for tname, tinfo in ((database_schema or {}).get("tables", {}) or {}).items():
        cols = []
        for cname, cinfo in (tinfo.get("columns", {}) or {}).items():
            bits = [f"{cname} {cinfo.get('column_type','')}".strip()]
            if cinfo.get("primary_key"):
                bits.append("PK")
            fks = cinfo.get("foreign_keys") or []
            if fks:
                bits.append(f"FK->{fks[0] if isinstance(fks[0], str) else fks[0]}")
            desc = (cinfo.get("description") or "").split("|")[0].strip()
            if desc:
                bits.append(desc[:90])
            vals = [str(v) for v in (cinfo.get("value_examples") or []) if v != ""][:max_values]
            if vals:
                bits.append("e.g. " + ", ".join(vals))
            cols.append("      - " + " | ".join(bits))
        lines.append(f"  {tname}:\n" + "\n".join(cols))
    return "\n".join(lines)


def render_keys(schema) -> str:
    out = []
    for t, pks in (schema.pks or {}).items():
        out.append(f"  PK {t}({', '.join(pks)})")
    for a, b in (schema.fks or []):
        out.append(f"  FK {a} -> {b}")
    return "\n".join(out) or "  (none declared)"


def render_values(retrieved_values: Optional[Dict[str, Dict[str, Any]]], cap: int = 5) -> str:
    if not retrieved_values:
        return "  (none)"
    out = []
    for t, cols in retrieved_values.items():
        for c, vals in (cols or {}).items():
            vs = []
            for v in (vals or [])[:cap]:
                vs.append(str(v.get("value") if isinstance(v, dict) else v))
            if vs:
                out.append(f"  {t}.{c}: " + ", ".join(vs))
    return "\n".join(out[:60]) or "  (none)"


# ── schema grounding (step 4, LLM-assisted mode) ─────────────────────────────
GROUNDING_SYSTEM = """You map question mentions to database columns.
Do not write SQL. Do not invent columns: use only the supplied schema.
The dependency structure is fixed — you decide WHICH column a mention refers to,
never what role it plays in the query.
A mention that refers to an entity/table as a whole, or that has no column
counterpart, must be mapped to null."""


def build_grounding_prompt(question: str, evidence: str, tree_render: str,
                           mentions: List[Dict[str, Any]], schema_str: str,
                           keys_str: str, values_str: str,
                           notes: str) -> List[Dict[str, str]]:
    """Ask the LLM to ground every mention/literal at once.

    Each mention carries the deterministic candidates as a hint, but the LLM may
    choose any column in the focused schema — deterministic proximity matching is
    unreliable for mentions like "total enrollment" or "ages of 5 and 17", which
    latch onto whatever column happens to be nearest.
    """
    mention_lines = []
    for m in mentions:
        hint = f"  candidates: {m['candidates']}" if m.get("candidates") else ""
        mention_lines.append(
            f"  - id={m['id']} kind={m['kind']} text={m['text']!r}{hint}")
    user = f"""Question: {question}

SME evidence: {evidence or '(none)'}

Dependency tree (fixed structural input):
{tree_render}

Focused schema:
{schema_str}

Keys:
{keys_str}

Retrieved values (a literal matching one of these is strong evidence):
{values_str}

Disambiguation notes:
{notes or '  (none)'}

Mentions to ground:
{chr(10).join(mention_lines)}

For EACH mention id return an object:
  {{"id": <id>, "column": "Table.Column" or null, "value": <normalized literal or null>}}
Rules:
- "column" must be exactly one of the columns in the focused schema above.
- For a literal, "column" is the column it should be COMPARED AGAINST, and "value"
  is the literal normalized to how it is stored (check retrieved values).
- If the mention names a table/entity as a whole, or nothing fits, use null.
- A phrase may be part of a COLUMN NAME rather than a value (e.g. "ages 5-17");
  in that case map it to that column and set "value" to null.

Return a JSON list of these objects, wrapped in <result></result> tags."""
    return [{"role": "system", "content": GROUNDING_SYSTEM},
            {"role": "user", "content": user}]


def parse_grounding_response(text: str, valid_cols: set) -> Dict[int, Dict[str, Any]]:
    """Fail-closed: unparseable or unknown columns are dropped, leaving the
    deterministic grounding in place for those mentions."""
    body = _strip_fence(_result_block(text) or "")
    if not body:
        return {}
    obj = None
    try:
        obj = json.loads(body)
    except Exception:
        for span in reversed(_balanced(body, "[", "]")):
            try:
                obj = json.loads(span)
                break
            except Exception:
                continue
    if not isinstance(obj, list):
        return {}
    lower = {c.lower(): c for c in valid_cols}
    out: Dict[int, Dict[str, Any]] = {}
    for e in obj:
        if not isinstance(e, dict) or e.get("id") is None:
            continue
        try:
            mid = int(e["id"])
        except (ValueError, TypeError):
            continue
        col = e.get("column")
        col = lower.get(str(col).strip().lower()) if col else None
        out[mid] = {"column": col, "value": e.get("value")}
    return out


# ── §5 Tree-to-IR (completion) ───────────────────────────────────────────────
IR_SYSTEM = """You ground a dependency-derived SQL intermediate representation.
Do not write SQL. Do not remove or invent question conditions.
Use only the supplied schema, values, keys, notes, and evidence.
Treat the dependency tree as fixed structural input.
For every mapping, record the supporting question tokens.
Leave uncertain mappings in "unresolved"."""


def build_ir_completion_prompt(question: str, tree_render: str, schema_str: str,
                               keys_str: str, values_str: str, notes: str,
                               evidence: str, draft_ir: SQLIR) -> List[Dict[str, str]]:
    user = f"""Question: {question}

Dependency tree (fixed structural input):
{tree_render}

Focused schema:
{schema_str}

Keys:
{keys_str}

Retrieved values:
{values_str}

Disambiguation notes:
{notes or '  (none)'}

SME evidence:
{evidence or '  (none)'}

Draft IR (deterministic; resolve ONLY its "unresolved" entries):
{draft_ir.to_json()}

Return the completed IR as valid JSON matching the same schema.
Keep every already-resolved slot byte-identical. Resolve what you can from
"unresolved" into the proper slots (select/filters/group_by/order_by/joins) and
delete those entries; leave genuinely undecidable ones in "unresolved".
Wrap the JSON in <result></result> tags."""
    return [{"role": "system", "content": IR_SYSTEM},
            {"role": "user", "content": user}]


# ── §6 IR-to-SQL (rendering) ─────────────────────────────────────────────────
SQL_SYSTEM = """Generate one executable SQL query that implements the supplied IR exactly.
Preserve every filter, relation, aggregate, grouping, ordering, and limit.
Use only schema elements listed in the IR.
Do not introduce unsupported conditions."""


def build_sql_render_prompt(dialect: str, question: str, schema_str: str,
                            validated_ir: SQLIR, notes: str,
                            variant: str = "canonical") -> List[Dict[str, str]]:
    style = {
        "canonical": "Write the most direct, canonical query for this IR.",
        "alternative": ("Write a query that is STRUCTURALLY DIFFERENT from the canonical "
                        "form (e.g. subquery instead of join, or CTE) but semantically "
                        "identical to the IR."),
    }.get(variant, "Write the most direct, canonical query for this IR.")
    user = f"""SQL dialect: {dialect}
Question: {question}

Focused schema:
{schema_str}

Validated IR:
{validated_ir.to_json()}

Disambiguation notes:
{notes or '  (none)'}

{style}

Return JSON with keys "sql" and "ir_component_mapping" (mapping each IR component
to the SQL fragment that realizes it), wrapped in <result></result> tags."""
    return [{"role": "system", "content": SQL_SYSTEM},
            {"role": "user", "content": user}]


# ── §8 targeted repair ───────────────────────────────────────────────────────
def build_repair_prompt(question: str, validated_ir: SQLIR, sql: str,
                        violations: List[str], schema_str: str) -> List[Dict[str, str]]:
    user = f"""Revise the SQL only to resolve the listed validation failures.
Preserve all components that passed validation.

Question: {question}

Focused schema:
{schema_str}

IR:
{validated_ir.to_json()}

Current SQL:
{sql}

Failed checks:
{chr(10).join('  - ' + v for v in violations)}

Return only the revised SQL inside <result></result> tags."""
    return [{"role": "system", "content":
             "You repair SQL against a fixed IR. Make localized changes only."},
            {"role": "user", "content": user}]


# ── response parsers (fail-closed) ───────────────────────────────────────────
def _result_block(text: str) -> Optional[str]:
    import re
    m = re.search(r"<result>(.*?)</result>", text or "", re.DOTALL)
    if m:
        return m.group(1).strip()
    return (text or "").strip() or None


def _strip_fence(s: str) -> str:
    s = (s or "").strip()
    for fence in ("```json", "```sql", "```"):
        if s.startswith(fence):
            s = s[len(fence):]
            break
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _balanced(text: str, o: str, c: str) -> List[str]:
    spans, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == o:
            if depth == 0:
                start = i
            depth += 1
        elif ch == c and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start:i + 1])
    return spans


def parse_ir_response(text: str) -> Optional[Dict[str, Any]]:
    body = _strip_fence(_result_block(text) or "")
    if not body:
        return None
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    for span in reversed(_balanced(body, "{", "}")):
        try:
            obj = json.loads(span)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def parse_sql_response(text: str) -> Optional[str]:
    """Accepts {"sql": ...} JSON or a bare SQL string."""
    body = _strip_fence(_result_block(text) or "")
    if not body:
        return None
    obj = None
    try:
        obj = json.loads(body)
    except Exception:
        for span in reversed(_balanced(body, "{", "}")):
            try:
                obj = json.loads(span)
                break
            except Exception:
                continue
    if isinstance(obj, dict) and obj.get("sql"):
        return _strip_fence(str(obj["sql"]))
    low = body.lower()
    if low.startswith("select") or low.startswith("with"):
        return body
    import re
    m = re.search(r"\b(SELECT|WITH)\b[\s\S]+", body, re.I)
    return _strip_fence(m.group(0)) if m else None
