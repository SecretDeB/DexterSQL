"""
DexterSQL — the `dep_tree` SQL generation method.

Overview
--------
A single-call, three-stage generation method whose prompt is grounded in a
DETERMINISTIC linguistic decomposition of the question:

  1. deterministic pre-pass (no LLM)
       dependency parse -> semantic units -> schema grounding candidates
  2. ONE LLM call, sampled `n` times from the same prompt
       Plan -> Skeleton -> Complete, plus a Coverage Check step that forces the
       model to account for every element the parser found
  3. deterministic post-pass (no LLM)
       parse the SQL and validate it against a draft IR built from the same
       decomposition (cartesian products, unknown schema elements, dropped
       predicates)

The dependency analysis constrains the prompt's *discipline* — the coverage
check, the projection/filter rules, the explicit key relationships — rather than
being dumped into the prompt as raw parse data. Ablation showed the raw parse and
element checklist as prompt DATA measured no better than the leaner form, so
`dep_tree` ships the instructions and omits the raw blocks.

Few-shot examples (`Question -> SQL`, cross-domain) are included when a
FewShotStore is supplied. This is the configuration validated end-to-end.

Cost: exactly ONE LLM call per question (n samples from it). Validation is
deterministic and free; `repair=True` optionally spends extra calls to fix
violations, and is OFF by default.

Usage
-----
    from dextersql import DepTreeGenerator, FewShotStore, TraceRecorder

    gen = DepTreeGenerator(
        fewshot_store=FewShotStore("few_shots.json"),
        trace_recorder=TraceRecorder(trace_dir="traces", per_db=5),
    )
    sqls, usage, report = gen.generate_for(
        question="How many charter schools are in Fresno?",
        database_schema=focused_schema,   # post-schema-linking schema dict
        evidence="charter schools refers to Charter = 1",
        retrieved_values=values,
        db_id="california_schools", question_id=42,
        sampling_budget=4,
        llm_call=my_llm_call,             # (messages, n) -> (List[str], usage)
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import prompts as P
from .parser import parse_question, nlp_error, DepTree
from .units import Unit, extract_units
from .grounding import Grounder, GroundedSchema, load_schema
from .validate import validate_sql
from .ir import SQLIR
from .ir_builder import build_ir
from .trace import TraceRecorder
from .fewshot import STYLE_INSTRUCTIONS


# ── the generation prompt ────────────────────────────────────────────────────
DEP_TREE_PROMPT = """# Task:
You are an expert SQL developer who uses a systematic approach to generate complex SQL queries.
Your task is to analyze the given question and database schema, then generate a SQL query using a three-step process:
1. **Plan**: Identify the required SQL components and logical structure
2. **Skeleton**: Create a structured SQL skeleton with placeholders
3. **Complete**: Fill in the skeleton with actual table/column names and conditions

The question has been analyzed with a dependency parser to determine how its parts
relate. Use that structure to make sure no part of the question is dropped or invented.
{FEW_SHOT_INSTRUCTION}
# Instructions:

## Step 1: Plan (SQL Components Analysis)
Account for EVERY element of the question. For each one, state which SQL component
realizes it (SELECT / FROM / JOIN / WHERE / GROUP BY / HAVING / ORDER BY / LIMIT),
or state explicitly why it needs none (e.g. it names a table, or it is part of a
column name). Then identify:
- **SELECT clause**: What data needs to be retrieved? (columns, aggregations, calculations)
- **FROM clause**: Which tables are needed?
- **JOIN clauses**: What relationships need to be established?
- **WHERE clause**: What filtering conditions are required?
- **GROUP BY clause**: What grouping is needed for aggregations?
- **HAVING clause**: What post-aggregation filtering is needed?
- **ORDER BY clause**: What sorting is required?
- **LIMIT clause**: Are there any row limits?
- **Subqueries**: Are nested queries needed?
- **Special functions**: Date functions, string functions, mathematical operations

## Step 2: Skeleton (Structured Template)
Create a SQL skeleton with:
- Clear structure showing the logical flow
- Placeholders for table names, column names, and conditions
- Comments explaining the purpose of each section
- Proper indentation and formatting

## Step 3: Complete (Final SQL)
Fill in the skeleton with:
- Exact table and column names from the schema
- Specific values and conditions from the question
- Proper {DIALECT} syntax and functions
- Final validation of the query logic

## Step 4: Coverage Check
Before writing the final SQL, re-read the question and confirm every element is
either realized in the query or explicitly justified as needing no SQL component.
A modifier that describes an entity ("charter schools", "virtual schools",
"direct-funded") is almost always a WHERE condition, not a column to select.

# Important Rules:
1. **Schema Accuracy**: Use exact table and column names from the provided schema
2. **{DIALECT} Compatibility**: Use only {DIALECT}-compatible functions and syntax
3. **Logical Flow**: Ensure the query logic matches the question requirements
4. **Performance**: Prefer efficient JOIN patterns over nested subqueries when possible
5. **Readability**: Use clear aliases and proper formatting
6. **Completeness**: Address all aspects mentioned in the question and hint. Do NOT drop
   a condition because it seems minor, and do NOT add a condition the question did not ask for.
7. **Foreign Key Constraints**: If there are multiple tables to JOIN, you MUST ensure that the joined tables have EXPLICIT FOREIGN KEYS between them. For example, "TableA -> TableB, TableC -> TableB", directly join TableA and TableC is NOT ALLOWED, you must join TableA and TableB, and then join TableB and TableC.
8. **Projection Discipline**: SELECT exactly what the question asks to see — no extra
   columns. A value used only for filtering belongs in WHERE, not in SELECT.

# Output Format:
Please respond with XML code structured as follows:
<reasoning>
    Your comprehensive analysis and planning for the SQL query generation and the SQL skeleton with placeholders, including the Step 4 coverage check.
</reasoning>
<result>
    The final SQL query that answers the target question and can be executed on the target {DIALECT} database, ensure there is not any SQL comment and not any other explanation text in the SQL query.
    The SQL query must not include XML-specific characters (e.g., `&lt;`, `&gt;`, `&amp;`); only SQL-valid characters are allowed.
</result>

# Input:
{FEW_SHOT_BLOCK}## Database Schema:
{DATABASE_SCHEMA}

## Key Relationships:
{KEYS}

## Question:
{QUESTION}

## Hint:
{HINT}

# Output:
"""


def build_prompt(question: str, evidence: str, database_schema: Dict[str, Any],
                 schema: GroundedSchema, dialect: str = "SQLite",
                 fewshot_block: str = "", fewshot_instruction: str = "") -> str:
    """Render the generation prompt. `schema` supplies the explicit PK/FK list."""
    fs_block = f"## Few-Shot Examples:\n{fewshot_block}\n\n" if fewshot_block else ""
    fs_instr = f"\n{fewshot_instruction}\n" if fewshot_instruction else ""
    return (DEP_TREE_PROMPT
            .replace("{FEW_SHOT_BLOCK}", fs_block)
            .replace("{FEW_SHOT_INSTRUCTION}", fs_instr)
            .replace("{DATABASE_SCHEMA}", P.render_schema(database_schema))
            .replace("{KEYS}", P.render_keys(schema))
            .replace("{QUESTION}", question or "")
            .replace("{HINT}", (evidence or "").strip() or "(none)")
            .replace("{DIALECT}", dialect))


class DepTreeGenerator:
    """The `dep_tree` method: one LLM call over a deterministic decomposition.

    `llm_call(messages, n) -> (List[str] raw_responses, usage_dict)` — the n
    samples are drawn from the SAME prompt.
    """

    def __init__(self, fewshot_store=None, example_style: str = "plain",
                 trace_recorder: Optional[TraceRecorder] = None,
                 validate: bool = True, repair: bool = False,
                 max_repair_rounds: int = 1):
        self.tracer = trace_recorder or TraceRecorder()
        # Share ONE store across generator instances: the pool and masked index are
        # large and should be parsed once, not per instance.
        self.fewshot_store = fewshot_store
        self.example_style = example_style
        # Deterministic and free; only annotates the report unless `repair` is on.
        # repair=False keeps the cost at exactly one LLM call per question.
        self.validate = validate
        self.repair = repair
        self.max_repair_rounds = max_repair_rounds

    def generate_for(self, question: str, database_schema: Dict[str, Any],
                     evidence: str = "", retrieved_values: Optional[Dict] = None,
                     notes: str = "", dialect: str = "sqlite",
                     db_id: str = "", question_id: Any = "",
                     sampling_budget: int = 4,
                     llm_call=None) -> Tuple[List[str], Dict[str, int], Dict[str, Any]]:
        """Returns (sql_candidates, token_usage, report)."""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tr = self.tracer.new(db_id, question_id, question)
        report: Dict[str, Any] = {"elements": [], "violations": [], "n_calls": 0,
                                  "trace_path": None}

        def _acc(u):
            for k in usage:
                usage[k] += int((u or {}).get(k, 0) or 0)

        # ── 1. deterministic decomposition (no LLM) ──────────────────────────
        tree = parse_question(question)
        if tree is None:
            # Fail closed: contribute no candidates rather than an ungrounded guess.
            tr.step("A_parse_failed", {"error": nlp_error() or "spaCy unavailable"})
            tr.save()
            return [], usage, report
        tr.step("A_dependency_tree", {"render": tree.render()})

        units = extract_units(tree)
        tr.step("B_semantic_units", [u.to_dict() for u in units])

        schema = load_schema(database_schema)
        note_list = [n for n in (notes or "").split("\n") if n.strip()]
        grounder = Grounder(schema, retrieved_values, note_list)
        report["elements"] = [u.text for u in units]

        if llm_call is None:
            tr.save()
            return [], usage, report

        # ── 2. ONE LLM call, n samples from the same prompt ──────────────────
        fs_block = fs_instr = ""
        if self.fewshot_store is not None:
            fs_block = self.fewshot_store.render(question_id, style=self.example_style)
            fs_instr = STYLE_INSTRUCTIONS.get(self.example_style, "")
            tr.step("C_fewshot", {"style": self.example_style,
                                  "n_chars": len(fs_block),
                                  "block_head": fs_block[:400]})

        prompt = build_prompt(
            question, evidence, database_schema, schema,
            dialect=("SQLite" if dialect == "sqlite" else dialect),
            fewshot_block=fs_block, fewshot_instruction=fs_instr)
        messages = [{"role": "user", "content": prompt}]
        raws, u = llm_call(messages, n=max(1, sampling_budget))
        _acc(u)
        report["n_calls"] = 1
        tr.llm("dep_tree", messages, raws, None, {"n_samples": max(1, sampling_budget)})

        # ── 3. deterministic validation (no LLM unless repair=True) ──────────
        candidates: List[str] = []
        for i, raw in enumerate(raws or []):
            sql = P.parse_sql_response(raw)
            if not sql:
                continue
            violations = validate_sql(sql, self._draft_ir(tree, units, grounder, dialect),
                                      schema, dialect=dialect) if self.validate else []
            if violations and self.repair:
                sql, violations, ru = self._repair(question, sql, violations, schema,
                                                   database_schema, llm_call, tr, i)
                _acc(ru)
            tr.candidate(sql, f"sample{i}", violations,
                         repaired=bool(violations and self.repair))
            if sql not in candidates:
                candidates.append(sql)
            report["violations"].extend(violations)

        tr.stat(n_candidates=len(candidates), n_llm_calls=report["n_calls"])
        report["trace_path"] = tr.save()
        return candidates, usage, report

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _draft_ir(tree: DepTree, units: List[Unit], grounder: Grounder,
                  dialect: str) -> SQLIR:
        """A deterministic draft IR used ONLY to sanity-check the produced SQL
        (cartesian products, unknown schema elements, dropped predicates). It is
        never sent to the model and never constrains generation — the single-call
        design deliberately lets the LLM own the SQL structure."""
        try:
            return build_ir(tree, units, grounder, dialect=dialect)
        except Exception:
            return SQLIR()

    def _repair(self, question, sql, violations, schema, database_schema,
                llm_call, tr, idx):
        """Targeted repair: send ONLY the failed checks. Off by default."""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        rounds = 0
        while violations and rounds < self.max_repair_rounds:
            msgs = P.build_repair_prompt(question, SQLIR(), sql, violations,
                                         P.render_schema(database_schema))
            raws, u = llm_call(msgs, n=1)
            for k in usage:
                usage[k] += int((u or {}).get(k, 0) or 0)
            fixed = P.parse_sql_response(raws[0] if raws else "")
            tr.llm(f"repair:{idx}:{rounds}", msgs, raws[0] if raws else "",
                   {"sql": fixed, "violations_sent": violations})
            rounds += 1
            if not fixed:
                break
            sql = fixed
            violations = validate_sql(sql, SQLIR(), schema)
        return sql, violations, usage
