#!/usr/bin/env python3
"""
Batch-run DexterSQL's `dep_tree` generator over a dataset file.

Input format (JSONL, one question per line). Only `question` and `schema` are
required; everything else is optional:

  {
    "question_id": 42,
    "db_id": "california_schools",
    "question": "How many charter schools are in Fresno?",
    "evidence": "charter refers to Charter = 1",          # optional hint
    "schema": { "tables": { "<table>": { "columns": {...} } } },
    "retrieved_values": { "<table>": { "<col>": [ {"value": "...", "distance": 0.1} ] } },
    "gold_sql": "SELECT ..."                              # optional, for eval
  }

`schema` should be the FOCUSED schema (after schema linking), not the whole DB.

Examples
--------
  # dry run: deterministic stages only, no LLM, no server needed
  python scripts/run_dextersql.py --data dev.jsonl --dry-run

  # full generation against an OpenAI-compatible endpoint
  python scripts/run_dextersql.py --data dev.jsonl \
      --model my-model --base-url http://localhost:8000/v1 \
      --few-shot few_shots.json --out preds.jsonl --trace-dir traces
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dextersql import (DepTreeGenerator, FewShotStore, TraceRecorder,  # noqa: E402
                       make_llm_call, parse_question, nlp_error)


def parse_args():
    ap = argparse.ArgumentParser(description="Run DexterSQL dep_tree generation.")
    ap.add_argument("--data", required=True, help="input JSONL")
    ap.add_argument("--out", default=None, help="output JSONL with sql_candidates")
    ap.add_argument("--db-filter", default=None, help="only run this db_id")
    ap.add_argument("--limit", type=int, default=0, help="max questions (0 = all)")

    ap.add_argument("--dry-run", action="store_true",
                    help="deterministic stages only; no LLM calls")
    ap.add_argument("--model", default=os.environ.get("DEXTERSQL_MODEL", ""))
    ap.add_argument("--base-url", default=os.environ.get("DEXTERSQL_BASE_URL", ""))
    ap.add_argument("--api-key", default=os.environ.get("DEXTERSQL_API_KEY", "EMPTY"))
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature; >0 so the n samples differ")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--sampling-budget", type=int, default=4,
                    help="candidates sampled from the single prompt")

    ap.add_argument("--few-shot", default=os.environ.get("DEXTERSQL_FEWSHOT_PATH", ""),
                    help="few-shot pool JSON (omit to run zero-shot)")
    ap.add_argument("--masked", default=os.environ.get("DEXTERSQL_MASKED_PATH", ""),
                    help="masked-example JSON (only used by non-default styles)")
    ap.add_argument("--example-style", default="plain", choices=["plain", "skeleton", "masked"])

    ap.add_argument("--trace-dir", default=None, help="A-Z traces land here")
    ap.add_argument("--trace-per-db", type=int, default=5,
                    help="trace the first N questions per database")
    ap.add_argument("--trace-all", action="store_true")
    ap.add_argument("--repair", action="store_true",
                    help="spend extra LLM calls fixing validation violations")
    return ap.parse_args()


def main():
    a = parse_args()
    if not os.path.exists(a.data):
        sys.exit(f"[dextersql] input not found: {a.data}")
    if not a.dry_run and not a.model:
        sys.exit("[dextersql] --model is required unless --dry-run")

    if parse_question("smoke test the parser") is None:
        sys.exit(f"[dextersql] spaCy unavailable ({nlp_error()}). "
                 f"Install: pip install spacy && python -m spacy download en_core_web_sm")

    store = None
    if a.few_shot:
        store = FewShotStore(fewshot_path=a.few_shot, masked_path=a.masked or None)
        if not store.pool:
            sys.exit(f"[dextersql] few-shot pool at {a.few_shot} is empty — "
                     f"refusing to run (it would silently become zero-shot).")
        print(f"[dextersql] few-shot pool: {len(store.pool)} questions "
              f"| masked index: {len(store.masked_by_sql)}")
    else:
        print("[dextersql] no --few-shot given: running ZERO-SHOT")

    tracer = TraceRecorder(trace_dir=a.trace_dir, per_db=a.trace_per_db,
                           trace_all=a.trace_all)
    gen = DepTreeGenerator(fewshot_store=store, example_style=a.example_style,
                           trace_recorder=tracer, repair=a.repair)

    llm_call = None
    if not a.dry_run:
        llm_call = make_llm_call(model=a.model, base_url=a.base_url or None,
                                 api_key=a.api_key, temperature=a.temperature,
                                 max_tokens=a.max_tokens)
        print(f"[dextersql] model={a.model} base_url={a.base_url or '(default)'} "
              f"temp={a.temperature} n={a.sampling_budget}")

    out_f = open(a.out, "w") if a.out else None
    n = n_sql = n_empty = 0
    agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    t0 = time.time()

    for line in open(a.data):
        line = line.strip()
        if not line:
            continue
        item: Dict[str, Any] = json.loads(line)
        db = item.get("db_id", "")
        if a.db_filter and db != a.db_filter:
            continue
        if a.limit and n >= a.limit:
            break
        n += 1

        cands, usage, report = gen.generate_for(
            question=item["question"],
            database_schema=item.get("schema") or {},
            evidence=item.get("evidence", "") or "",
            retrieved_values=item.get("retrieved_values"),
            notes=item.get("notes", "") or "",
            dialect=item.get("dialect", "sqlite"),
            db_id=db, question_id=item.get("question_id", n),
            sampling_budget=a.sampling_budget,
            llm_call=llm_call,
        )
        for k in agg:
            agg[k] += usage[k]
        n_sql += len(cands)
        if not cands:
            n_empty += 1

        if out_f:
            out_f.write(json.dumps({
                "question_id": item.get("question_id", n), "db_id": db,
                "question": item["question"], "gold_sql": item.get("gold_sql", ""),
                "sql_candidates": cands,
                "n_violations": len(report["violations"]),
                "trace_path": report.get("trace_path"),
            }) + "\n")

        if n % 25 == 0:
            print(f"  {n} questions | candidates={n_sql} empty={n_empty} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    if out_f:
        out_f.close()

    print(f"[dextersql] DONE {n} questions in {(time.time() - t0) / 60:.1f} min")
    print(f"[dextersql] candidates={n_sql}  questions with none={n_empty}")
    if not a.dry_run:
        print(f"[dextersql] tokens: {json.dumps(agg)}")
    if a.out:
        print(f"[dextersql] predictions -> {a.out}")
    if a.trace_dir:
        print(f"[dextersql] traces -> {a.trace_dir}")


if __name__ == "__main__":
    main()
