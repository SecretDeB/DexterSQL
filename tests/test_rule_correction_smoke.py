#!/usr/bin/env python3
"""
Self-contained smoke test for the sql_rule_correction stage. No network, no
LLM server, no dataset -- exercises exactly the logic that is new in this
stage (prompt construction, and parsing/validating the LLM's <result> JSON),
the same way tests/test_smoke.py stub-tests the dep_tree generator.

The stage's plumbing (ThreadPoolExecutor fan-out, ArtifactStore checkpoint-
ing, dedup-by-normalized-SQL) is copied verbatim from sql_revision's already
-proven pattern and isn't re-tested here.

Run:  python tests/test_rule_correction_smoke.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dextersql.rule_correction.example_rules import RULES, RULE_NAMES, DISABLED_RULES  # noqa: E402
from dextersql.rule_correction.prompts import format_rule_correction_prompt  # noqa: E402

ok = True


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"   {'OK  ' if cond else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))


def main():
    print("== 1. rule set ==")
    # 2 example rules ship by default; three more sit in DISABLED_RULES as
    # documented negative results -- see the module docstring in
    # example_rules.py. Update this count when the active set changes.
    check("2 rules active", len(RULES) == 2, f"n={len(RULES)}")
    check("every active rule has all fields", all(
        {"rule_name", "gist", "bad_pattern", "correct_pattern", "fix"} <= set(r.keys())
        for r in RULES
    ))
    check("rule names unique", len(set(RULE_NAMES)) == len(RULE_NAMES))
    check("disabled rules kept out of the active set", not (set(r["rule_name"] for r in DISABLED_RULES) & set(RULE_NAMES)))

    print("\n== 2. prompt construction ==")
    schema = "CREATE TABLE t (id INTEGER, name TEXT);"
    sql = "SELECT id, name FROM t"
    p = format_rule_correction_prompt(schema, "Which ids exist?", "", sql)
    check("prompt built", len(p) > 500, f"chars={len(p)}")
    check("no unfilled placeholders", "{RULES_BLOCK}" not in p and "{QUESTION}" not in p and "{SQL}" not in p)
    check("all active rule names rendered", all(name in p for name in RULE_NAMES))
    check("schema/question/sql rendered", schema in p and "Which ids exist?" in p and sql in p)
    check("json result shape documented", '"applied_rules"' in p and '"corrected_sql"' in p)

    print("\n== 3. response parsing (via a bare, unconstructed runner instance) ==")
    # A __new__'d instance (no __init__, no LLM, no dataset) is enough to
    # exercise the parser in isolation -- but the active rule set is now an
    # instance attribute (so a mined set can replace the examples), and the
    # parser filters against it, so set it here as __init__ would.
    from dextersql.rule_correction.sql_rule_correction import SQLRuleCorrectionRunner
    runner = SQLRuleCorrectionRunner.__new__(SQLRuleCorrectionRunner)
    runner._rules = RULES
    runner._rule_names = RULE_NAMES

    well_formed = (
        "<reasoning>RC-PROJECTION-EXTRA-COLUMN applies.</reasoning>\n"
        '<result>{"applied_rules": ["RC-PROJECTION-EXTRA-COLUMN"], '
        '"corrected_sql": "SELECT id FROM t"}</result>'
    )
    r = runner._parse_llm_response(well_formed)
    check("well-formed response parsed", r == {
        "applied_rules": ["RC-PROJECTION-EXTRA-COLUMN"],
        "corrected_sql": "SELECT id FROM t",
    }, f"{r}")

    no_rule_applies = '<result>{"applied_rules": [], "corrected_sql": "SELECT id FROM t"}</result>'
    r = runner._parse_llm_response(no_rule_applies)
    check("no-rule-applies passthrough parsed", r == {"applied_rules": [], "corrected_sql": "SELECT id FROM t"}, f"{r}")

    fenced = (
        '<result>```json\n{"applied_rules": ["RC-COLUMN-ORDER-MISMATCH"], '
        '"corrected_sql": "SELECT x FROM t ORDER BY x LIMIT 1"}\n```</result>'
    )
    r = runner._parse_llm_response(fenced)
    check("```json fence stripped", r is not None and r["applied_rules"] == ["RC-COLUMN-ORDER-MISMATCH"], f"{r}")

    unknown_rule = '<result>{"applied_rules": ["RC-NOT-A-REAL-RULE", "RC-PROJECTION-EXTRA-COLUMN"], "corrected_sql": "SELECT 1"}</result>'
    r = runner._parse_llm_response(unknown_rule)
    check("unknown rule name filtered out", r is not None and r["applied_rules"] == ["RC-PROJECTION-EXTRA-COLUMN"], f"{r}")

    no_tag = "the model just chatted instead of following the format"
    check("missing <result> tag -> None", runner._parse_llm_response(no_tag) is None)

    empty_sql = '<result>{"applied_rules": [], "corrected_sql": ""}</result>'
    check("empty corrected_sql -> None (invalid)", runner._parse_llm_response(empty_sql) is None)

    not_json = "<result>SELECT id FROM t</result>"
    check("bare SQL (not JSON) -> None", runner._parse_llm_response(not_json) is None)

    print("\n" + ("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
