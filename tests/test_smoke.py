#!/usr/bin/env python3
"""
Self-contained smoke test for DexterSQL. No network, no LLM server.

Covers the whole `dep_tree` path with a stub backend:
  deterministic decomposition -> prompt build -> response parsing ->
  SQL/IR validation -> candidate collection -> tracing -> fail-closed behaviour

Run:  python tests/test_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dextersql import (DepTreeGenerator, StubBackend, TraceRecorder,  # noqa: E402
                       parse_question, extract_units, load_schema, Grounder,
                       build_ir, validate_sql, build_prompt, nlp_error)

SCHEMA = {
    "tables": {
        "Patient": {"table_name": "Patient", "columns": {
            "ID": {"column_name": "ID", "column_type": "INTEGER", "primary_key": True,
                   "foreign_keys": [], "description": "patient id",
                   "value_examples": ["2110", "11408"],
                   "value_statistics": {"total_count": 1238, "distinct_count": 1238}},
            "SEX": {"column_name": "SEX", "column_type": "TEXT", "primary_key": False,
                    "foreign_keys": [], "description": "Sex | F: female; M: male",
                    "value_examples": ["F", "M"],
                    "value_statistics": {"total_count": 1238, "distinct_count": 2}},
            "Diagnosis": {"column_name": "Diagnosis", "column_type": "TEXT",
                          "primary_key": False, "foreign_keys": [],
                          "description": "disease name",
                          "value_examples": ["SLE", "RA", "PSS"],
                          "value_statistics": {"total_count": 1238, "distinct_count": 220}},
        }},
        "Examination": {"table_name": "Examination", "columns": {
            "ID": {"column_name": "ID", "column_type": "INTEGER", "primary_key": False,
                   "foreign_keys": ["Patient.ID"], "description": "patient id",
                   "value_examples": ["2110"],
                   "value_statistics": {"total_count": 806, "distinct_count": 763}},
            "Diagnosis": {"column_name": "Diagnosis", "column_type": "TEXT",
                          "primary_key": False, "foreign_keys": [],
                          "description": "diagnosis at examination",
                          "value_examples": ["SLE", "SjS"],
                          "value_statistics": {"total_count": 806, "distinct_count": 182}},
        }},
    },
}
RETRIEVED = {"Patient": {"Diagnosis": [{"value": "SLE", "distance": 0.05}]}}
QUESTION = "How many patients were diagnosed with 'SLE'?"

ok = True


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"   {'OK  ' if cond else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))


def main():
    print("== 1. spaCy availability ==")
    tree = parse_question(QUESTION)
    if tree is None:
        print(f"   FAIL spaCy unavailable ({nlp_error()}).")
        print("   Install: pip install spacy && python -m spacy download en_core_web_sm")
        return 1
    check("dependency parse produced", tree is not None)

    print("\n== 2. deterministic decomposition ==")
    units = extract_units(tree)
    schema = load_schema(SCHEMA)
    grounder = Grounder(schema, RETRIEVED, [])
    ir = build_ir(tree, units, grounder)
    check("semantic units extracted", len(units) > 0, f"n={len(units)}")
    check("schema parsed", len(schema.columns) == 5, f"cols={len(schema.columns)}")
    check("FK discovered", len(schema.fks) >= 1, f"fks={schema.fks}")
    print(f"   IR select : {[s.expression for s in ir.select]}")
    print(f"   IR filters: {[(f.column, f.operator, f.value) for f in ir.filters]}")

    print("\n== 3. prompt construction ==")
    p = build_prompt(QUESTION, "", SCHEMA, schema, fewshot_block="", fewshot_instruction="")
    check("prompt built", len(p) > 500, f"chars={len(p)}")
    check("no unfilled placeholders", "{" + "DATABASE_SCHEMA" + "}" not in p)
    check("schema rendered into prompt", "Patient" in p and "Diagnosis" in p)

    p_fs = build_prompt(QUESTION, "", SCHEMA, schema,
                        fewshot_block="- Example 1:\nQuestion: q\nSQL: SELECT 1",
                        fewshot_instruction="Study the examples.")
    check("few-shot block injected", "## Few-Shot Examples:" in p_fs and len(p_fs) > len(p))

    print("\n== 4. SQL-vs-IR validation ==")
    good = "SELECT COUNT(*) FROM Patient WHERE Diagnosis = 'SLE'"
    dropped = "SELECT COUNT(*) FROM Patient"
    cartesian = "SELECT p.ID FROM Patient p, Examination e"
    v_good = validate_sql(good, ir, schema)
    v_drop = validate_sql(dropped, ir, schema)
    v_cart = validate_sql(cartesian, ir, schema)
    check("correct SQL passes cleanly", len(v_good) == 0, f"{v_good}")
    check("dropped predicate detected", len(v_drop) > len(v_good))
    cart_hits = [x for x in v_cart if "cartesian" in x.lower()]
    check("cartesian product detected", bool(cart_hits), f"{cart_hits[:1]}")

    print("\n== 5. end-to-end with stub LLM ==")
    with tempfile.TemporaryDirectory() as td:
        gen = DepTreeGenerator(trace_recorder=TraceRecorder(trace_dir=td, per_db=5))
        stub = StubBackend([
            f"<reasoning>plan</reasoning><result>{good}</result>",
            "<result>```sql\nSELECT COUNT(*) FROM Patient WHERE Diagnosis = 'SLE'\n```</result>",
        ])
        cands, usage, report = gen.generate_for(
            question=QUESTION, database_schema=SCHEMA, retrieved_values=RETRIEVED,
            db_id="thrombosis", question_id=1, sampling_budget=2, llm_call=stub)
        check("candidates produced", len(cands) >= 1, f"{cands}")
        check("exactly one LLM call", report["n_calls"] == 1)
        check("duplicates collapsed", len(cands) == 1, f"n={len(cands)}")

        tp = report.get("trace_path")
        check("trace written", tp and os.path.exists(tp))
        if tp and os.path.exists(tp):
            t = json.load(open(tp))
            steps = [s["step"] for s in t["steps"]]
            check("trace has A-Z steps", "A_dependency_tree" in steps, f"{steps}")
            check("trace stores prompt + response",
                  all("prompt_messages" in c and "raw_response" in c for c in t["llm_calls"]))

    print("\n== 6. fail-closed behaviour ==")
    with tempfile.TemporaryDirectory() as td:
        gen2 = DepTreeGenerator(trace_recorder=TraceRecorder(trace_dir=td))
        junk, _, _ = gen2.generate_for(
            question=QUESTION, database_schema=SCHEMA, question_id=2,
            sampling_budget=2, llm_call=StubBackend(["not sql at all", "still not sql"]))
        check("unparseable output -> no candidates", junk == [], f"{junk}")

        none_call, _, rep3 = gen2.generate_for(
            question=QUESTION, database_schema=SCHEMA, question_id=3, llm_call=None)
        check("no llm_call -> no candidates, no crash", none_call == [] and rep3["n_calls"] == 0)

    print("\n" + ("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
