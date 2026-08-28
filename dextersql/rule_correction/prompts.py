"""
Prompt for the sql_rule_correction stage.

One combined LLM call per candidate: show every rule, ask the model which (if
any) apply and to return the corrected SQL. This fuses the paper's Step 3
(rule relevance selection) and Step 4 (rule-guided correction) into a single
call, since the rule set is small (currently 4 active) -- there is nothing to gain from
a separate selection pass, but "applied_rules" is still reported for an
audit trail.

Follows this codebase's existing <reasoning>/<result> convention (see
dextersql/core/prompt/prompt_template.py); <result> holds a JSON object
instead of bare SQL so the applied-rule names survive alongside the SQL.
"""

from typing import List
from .example_rules import CorrectionRule, RULES

RULE_CORRECTION_SYSTEM_PROMPT = """\
# Task:
You are an SQL database expert reviewing a candidate SQLite query against a
fixed set of known error patterns. Each pattern was distilled from real
mistakes made on this exact benchmark. Your job is to check whether the
candidate SQL exhibits any of these patterns and, if so, correct it.

# Instructions:
1. Review the database schema, question, and evidence to understand what the
   query should return.
2. Check the candidate SQL against EACH rule below. A rule applies ONLY if
   the SQL clearly exhibits the described bad pattern for the specific
   question at hand -- do not apply a rule speculatively.
3. If one or more rules apply, rewrite the SQL to fix every applicable issue
   in a single corrected query. Preserve everything about the original SQL
   that is not implicated by an applied rule.
4. If no rule applies, return the SQL unchanged and an empty applied_rules
   list.

[IMPORTANT]
You are NOT ALLOWED to make any change that is not justified by one of the
listed rules. Do not rewrite working, correct SQL for style reasons. This
applies even when you are confident you've spotted a real, different bug
elsewhere in the query (wrong join, wrong filter scope, missing sort, wrong
table) -- if it doesn't match one of the listed bad patterns, leave it alone
and do not "fix" it. Every token you change must trace back to a rule you
listed as applied. A correction that changes more than what its rule's
correct_pattern describes (e.g. adding a NULL guard, an ORDER BY, or
restructuring a subquery while only a DISTINCT keyword was in scope) is
wrong even if the result looks like an improvement.

# Known Error Patterns:
{RULES_BLOCK}

# Output Format:
Please respond with XML code structured as follows.
<reasoning>
    For each rule, briefly state whether it applies to this SQL and why.
</reasoning>
<result>
    A single JSON object with exactly two keys:
    {{"applied_rules": ["<rule_name>", ...], "corrected_sql": "<full corrected SQLite query>"}}
    "applied_rules" is the list of rule_name values (from the patterns above)
    that you judged to apply and fixed; empty if none applied.
    "corrected_sql" is the final SQL -- the original SQL unchanged if
    applied_rules is empty, otherwise the rewritten query. It must be valid
    SQLite with no comments and no explanation text. Write comparison
    operators as literal characters (`<`, `>`, `<=`, `>=`) exactly as SQLite
    requires them -- do NOT XML-escape them as &lt;/&gt;/&amp;; those escaped
    forms are not valid SQL and will break execution.
</result>

# Input:
## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Evidence:
{EVIDENCE}

## Candidate SQL:
{SQL}

Check the candidate SQL against every rule above, then only output the XML
code (<reasoning>...</reasoning> and <result>...</result>) as your response.

# Output:
"""


def _format_rule(idx: int, rule: CorrectionRule) -> str:
    return (
        f"[{idx}] {rule['rule_name']}\n"
        f"  Gist            : {rule['gist']}\n"
        f"  Bad pattern     : {rule['bad_pattern']}\n"
        f"  Correct pattern : {rule['correct_pattern']}\n"
        f"  Fix             : {rule['fix']}"
    )


def format_rules_block(rules: List[CorrectionRule] = RULES) -> str:
    return "\n\n".join(_format_rule(i, r) for i, r in enumerate(rules, start=1))


def format_rule_correction_prompt(
    database_schema: str,
    question: str,
    evidence: str,
    sql: str,
    rules: List[CorrectionRule] = RULES,
) -> str:
    return RULE_CORRECTION_SYSTEM_PROMPT.format(
        RULES_BLOCK=format_rules_block(rules),
        DATABASE_SCHEMA=database_schema,
        QUESTION=question,
        EVIDENCE=evidence or "(none)",
        SQL=sql,
    )
