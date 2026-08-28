"""
Prompt templates for every LLM call in the Rule Creator pipeline.

All structured outputs follow a <reasoning>...</reasoning><result>...</result>
convention -- <reasoning> is for the model to think out loud (and for
humans debugging a bad output later), <result> is the only part any
parser reads.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Step 1a: candidate SQL generation
# ---------------------------------------------------------------------------

SQL_GENERATION_PROMPT = """You are a SQLite expert. Write a SQL query that answers the natural language question below, using the given database schema.

# Database Schema:
{DATABASE_SCHEMA}

# Question:
{QUESTION}

# Evidence:
{EVIDENCE}

Return ONLY the SQL query, wrapped in <result></result> tags, e.g.:
<result>SELECT ...</result>

No explanation, no markdown code fences, just the raw SQL inside the tags."""


def format_sql_generation_prompt(database_schema: Any, question: str, evidence: str) -> str:
    return SQL_GENERATION_PROMPT.format(
        DATABASE_SCHEMA=database_schema,
        QUESTION=question,
        EVIDENCE=evidence or "(none)",
    )


# ---------------------------------------------------------------------------
# Step 1b: failure explanation
# ---------------------------------------------------------------------------

FAILURE_EXPLANATION_PROMPT = """You compare a generated SQL query against the correct (gold) SQL query for the same
question, and explain precisely why the generated query is wrong. The generated
query's execution result does NOT match the gold query's execution result --
your job is to explain why, in terms precise enough that someone could turn
your explanation into a general SQL-writing rule.

# Database Schema:
{DATABASE_SCHEMA}

# Question:
{QUESTION}

# Evidence:
{EVIDENCE}

# Gold SQL (correct):
{GOLD_SQL}

# Generated SQL (incorrect -- its result differs from gold's):
{CANDIDATE_SQL}

Explain what is wrong with the generated SQL, compared to gold. If the generated
SQL has MULTIPLE independent problems (e.g. it both drops a filter AND
mis-orders columns), explain each one SEPARATELY -- do not merge unrelated
problems into one explanation. If it has only one problem, return only one
explanation.

Each explanation should be a self-contained paragraph: state the specific SQL
pattern the generated query used, the specific pattern gold used instead, and
why that difference changes the result. Do not just restate that the results
differ -- explain the underlying SQL-formulation reason.

Think step by step in <reasoning></reasoning>, then give your final answer in
<result></result> as a JSON array of explanation strings, e.g.:
<result>["The generated query ... whereas gold ... because ...", "Separately, the generated query also ..."]</result>

If you cannot identify any real difference in SQL formulation (e.g. the
mismatch looks like a gold-SQL quirk or a formatting artifact, not a genuine
generation error), return an empty array: <result>[]</result>"""


def format_failure_explanation_prompt(
    database_schema: Any, question: str, evidence: str, gold_sql: str, candidate_sql: str
) -> str:
    return FAILURE_EXPLANATION_PROMPT.format(
        DATABASE_SCHEMA=database_schema,
        QUESTION=question,
        EVIDENCE=evidence or "(none)",
        GOLD_SQL=gold_sql,
        CANDIDATE_SQL=candidate_sql,
    )


# ---------------------------------------------------------------------------
# Step 2: database-agnostic error isolation
# ---------------------------------------------------------------------------

DB_AGNOSTIC_CLASSIFICATION_PROMPT = """You decide whether a SQL-generation failure explanation describes a REUSABLE,
database-agnostic SQL-formulation issue, or a DATABASE-SPECIFIC artifact that
would not generalize to a different database and schema.

A failure is DATABASE-AGNOSTIC when the underlying mistake is about how SQL
is *formulated* in general -- e.g. wrong aggregation approach, missing type
cast, wrong column ordering, wrong use of DISTINCT, wrong extremum pattern,
wrong handling of NULLs, wrong join cardinality reasoning -- something that
could happen in the exact same shape on a completely different schema.

A failure is DATABASE-SPECIFIC when the mistake is really about *this*
database's particular tables, columns, or values -- e.g. the model picked the
wrong column because two columns in *this* schema have confusingly similar
names or meanings, it didn't know a domain-specific value/enum mapping specific
to *this* database, or it mis-linked to the wrong table due to *this*
schema's structure. Those are schema-linking / value-knowledge problems, not
reusable SQL-formulation problems, even though they also cause wrong SQL.

# Question:
{QUESTION}

# Gold SQL:
{GOLD_SQL}

# Generated SQL:
{CANDIDATE_SQL}

# Failure explanation to classify:
{EXPLANATION}

Think step by step in <reasoning></reasoning>, then answer in <result></result>
as JSON: {{"classification": "database_agnostic" | "database_specific"}}"""


def format_db_agnostic_classification_prompt(
    question: str, gold_sql: str, candidate_sql: str, explanation: str
) -> str:
    return DB_AGNOSTIC_CLASSIFICATION_PROMPT.format(
        QUESTION=question, GOLD_SQL=gold_sql, CANDIDATE_SQL=candidate_sql, EXPLANATION=explanation
    )


# ---------------------------------------------------------------------------
# Step 3: clustering -- batch-level grouping (within one database)
# ---------------------------------------------------------------------------

BATCH_CLUSTERING_PROMPT = """Below is a numbered list of SQL-generation failure explanations, all from
questions against the SAME database. Group together explanations that describe
the SAME underlying SQL-formulation error -- i.e. the same recurring mistake
pattern, even if the specific tables/columns/values differ.

Every explanation must be assigned to exactly one group. An explanation that
doesn't share its error pattern with any other explanation in this batch
should be its own group of size 1.

# Explanations:
{EXPLANATIONS_BLOCK}

Think step by step in <reasoning></reasoning>, then answer in <result></result>
as a JSON array of groups:
<result>[{{"label": "short description of the shared error pattern", "member_ids": [1, 4, 7]}}, {{"label": "...", "member_ids": [2]}}, ...]</result>

Every id from 1 to {N} must appear in exactly one group's member_ids."""


def format_batch_clustering_prompt(explanations: List[str]) -> str:
    block = "\n\n".join(f"[{i+1}] {exp}" for i, exp in enumerate(explanations))
    return BATCH_CLUSTERING_PROMPT.format(EXPLANATIONS_BLOCK=block, N=len(explanations))


# ---------------------------------------------------------------------------
# Step 3: clustering -- group merging (within-DB batch merge, and cross-DB merge)
# ---------------------------------------------------------------------------

GROUP_MERGE_PROMPT = """Below is a numbered list of error groups. Each group was formed by clustering
SQL-generation failure explanations that share the same underlying error
pattern, and is described by a label plus 1-2 representative examples.

Identify which groups describe the SAME underlying SQL-formulation error as
each other (even if their labels are worded differently) and should be
MERGED into one group. Groups that don't match anything else stay singleton.

# Groups:
{GROUPS_BLOCK}

Think step by step in <reasoning></reasoning>, then answer in <result></result>
as a JSON array of merge decisions:
<result>[{{"merged_label": "short description of the shared error pattern", "source_group_ids": [1, 3]}}, {{"merged_label": "...", "source_group_ids": [2]}}, ...]</result>

Every id from 1 to {N} must appear in exactly one merge decision's
source_group_ids -- a group that merges with nothing is its own
source_group_ids list of length 1."""


def format_group_merge_prompt(groups: List[Dict[str, Any]]) -> str:
    lines = []
    for i, g in enumerate(groups):
        examples = g.get("example_explanations", [])[:2]
        examples_txt = " | ".join(examples)
        lines.append(f"[{i+1}] label=\"{g['label']}\" support={g.get('support', '?')} examples: {examples_txt}")
    return GROUP_MERGE_PROMPT.format(GROUPS_BLOCK="\n".join(lines), N=len(groups))


# ---------------------------------------------------------------------------
# Step 4: rule synthesis
# ---------------------------------------------------------------------------

RULE_SYNTHESIS_PROMPT = """You convert a dominant, recurring SQL-generation error pattern into a canonical
correction rule. The rule will later be shown to an LLM at inference time,
alongside a candidate SQL query, so it can detect and fix this exact error
pattern in NEW questions against DIFFERENT databases it has never seen mined
examples from -- so the rule must describe the error and fix generically, not
in terms of this group's specific tables/columns/questions.

# Error group label:
{GROUP_LABEL}

# Representative examples from this group (question / gold SQL / generated SQL / why it's wrong):
{EXAMPLES_BLOCK}

Produce a rule with these four fields:
- gist: 1-3 sentences summarizing the recurring failure and the situation in which it occurs.
- bad_pattern: the faulty SQL formulation to detect -- specific and recognizable, so an LLM can reliably tell whether a NEW candidate query exhibits it.
- correct_pattern: the corresponding correct formulation.
- fix: a precise, minimal instruction for transforming the faulty formulation into the correct one -- explicit about what to change and, just as importantly, what NOT to touch (don't let the fix license unrelated changes to the query).

Think step by step in <reasoning></reasoning>, then answer in <result></result>
as JSON:
<result>{{"rule_name": "RC-SHORT-SLUG-IN-CAPS", "gist": "...", "bad_pattern": "...", "correct_pattern": "...", "fix": "..."}}</result>

rule_name must start with "RC-", use only uppercase letters and hyphens, and be short (under 40 characters)."""


def format_rule_synthesis_prompt(group_label: str, examples: List[Dict[str, Any]]) -> str:
    lines = []
    for i, ex in enumerate(examples):
        lines.append(
            f"[{i+1}] Q: {ex['question']}\n"
            f"    gold: {ex['gold_sql']}\n"
            f"    generated: {ex['candidate_sql']}\n"
            f"    why wrong: {ex['explanation']}"
        )
    return RULE_SYNTHESIS_PROMPT.format(GROUP_LABEL=group_label, EXAMPLES_BLOCK="\n\n".join(lines))
