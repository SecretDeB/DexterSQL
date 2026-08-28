#!/usr/bin/env python3
"""
DexterSQL quickstart — generate SQL for one question.

Two modes:
  * default: stub LLM, so it runs anywhere with no server
  * --live:  real generation against an OpenAI-compatible endpoint

    python examples/quickstart.py
    python examples/quickstart.py --live --model my-model \
        --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dextersql import DepTreeGenerator, StubBackend, make_llm_call  # noqa: E402

# A FOCUSED schema — i.e. what a schema-linking step would hand you, not the whole DB.
SCHEMA = {
    "tables": {
        "schools": {"table_name": "schools", "columns": {
            "CDSCode": {"column_name": "CDSCode", "column_type": "TEXT",
                        "primary_key": True, "foreign_keys": [],
                        "description": "school code", "value_examples": ["01100170109835"]},
            "School": {"column_name": "School", "column_type": "TEXT",
                       "primary_key": False, "foreign_keys": [],
                       "description": "school name", "value_examples": ["Envision Academy"]},
            "County": {"column_name": "County", "column_type": "TEXT",
                       "primary_key": False, "foreign_keys": [],
                       "description": "county name", "value_examples": ["Alameda", "Fresno"]},
            "Charter": {"column_name": "Charter", "column_type": "INTEGER",
                        "primary_key": False, "foreign_keys": [],
                        "description": "charter school (1) or not (0)",
                        "value_examples": ["1", "0"]},
        }},
    },
}

QUESTION = "How many charter schools are there in Fresno county?"
EVIDENCE = "charter schools refers to Charter = 1"
# Value retrieval output: which literals in the question match which columns.
RETRIEVED = {"schools": {"County": [{"value": "Fresno", "distance": 0.02}]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use a real LLM endpoint")
    ap.add_argument("--model", default=os.environ.get("DEXTERSQL_MODEL", ""))
    ap.add_argument("--base-url", default=os.environ.get("DEXTERSQL_BASE_URL", ""))
    ap.add_argument("--api-key", default=os.environ.get("DEXTERSQL_API_KEY", "EMPTY"))
    ap.add_argument("--n", type=int, default=2, help="candidates to sample")
    ap.add_argument("--show-prompt", action="store_true")
    a = ap.parse_args()

    if a.live:
        if not a.model:
            sys.exit("--live requires --model")
        llm = make_llm_call(model=a.model, base_url=a.base_url or None,
                            api_key=a.api_key, temperature=0.7)
    else:
        llm = StubBackend([
            "<reasoning>Plan: COUNT over schools, filter county and charter flag."
            "</reasoning><result>SELECT COUNT(*) FROM schools "
            "WHERE County = 'Fresno' AND Charter = 1</result>",
            "<result>SELECT COUNT(CDSCode) FROM schools "
            "WHERE County = 'Fresno' AND Charter = 1</result>",
        ])
        print("[quickstart] using StubBackend (pass --live for a real model)\n")

    gen = DepTreeGenerator()          # no few-shot store -> zero-shot
    sqls, usage, report = gen.generate_for(
        question=QUESTION, database_schema=SCHEMA, evidence=EVIDENCE,
        retrieved_values=RETRIEVED, db_id="california_schools", question_id=1,
        sampling_budget=a.n, llm_call=llm,
    )

    print(f"Question : {QUESTION}")
    print(f"Hint     : {EVIDENCE}")
    print(f"\nQuestion elements found by the parser ({len(report['elements'])}):")
    for e in report["elements"]:
        print(f"  - {e}")
    print(f"\nSQL candidates ({len(sqls)}), from {report['n_calls']} LLM call:")
    for i, s in enumerate(sqls):
        print(f"  [{i}] {' '.join(s.split())}")
    if report["violations"]:
        print(f"\nValidation violations: {report['violations']}")
    else:
        print("\nValidation: all candidates consistent with the deterministic IR")
    if a.live:
        print(f"\nTokens: {json.dumps(usage)}")

    if a.show_prompt:
        from dextersql import build_prompt, load_schema
        print("\n" + "=" * 70 + "\nPROMPT\n" + "=" * 70)
        print(build_prompt(QUESTION, EVIDENCE, SCHEMA, load_schema(SCHEMA)))


if __name__ == "__main__":
    main()
