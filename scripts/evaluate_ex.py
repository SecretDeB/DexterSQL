#!/usr/bin/env python3
"""
Execution-accuracy (EX) evaluation for DexterSQL predictions.

Comparison semantics: a prediction is correct when its result set equals the
gold result set as an unordered set of rows — set(pred_rows) == set(gold_rows).
Questions whose GOLD query fails to execute are dropped, not counted as wrong.

Because the generator emits several candidates per question, a single number is
ambiguous, so both bounds are reported:

  oracle EX  any candidate matches gold   -> upper bound (generation quality)
  first EX   candidate[0] matches gold    -> lower bound (no candidate selection)

A downstream selection stage lands between these.

Usage
-----
  python scripts/evaluate_ex.py --pred preds.jsonl --db-root /path/to/databases
  # expects <db-root>/<db_id>/<db_id>.sqlite, or pass db_path per row in preds
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple


def run_sql(db_path: str, sql: str, timeout: float = 30.0) -> Tuple[Optional[list], Optional[str]]:
    if not (sql or "").strip():
        return None, "empty"
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        con.text_factory = lambda b: b.decode("utf-8", "ignore")
        try:
            return con.execute(sql).fetchall(), None
        finally:
            con.close()
    except Exception as e:
        return None, str(e)[:200]


def eval_one(task) -> Dict[str, Any]:
    qid, db, db_path, gold, preds, timeout = task
    grows, gerr = run_sql(db_path, gold, timeout)
    if gerr is not None:
        return {"qid": qid, "db": db, "skipped": True, "reason": f"gold failed: {gerr}"}
    gold_set = set(grows)
    hits = []
    for p in preds:
        rows, err = run_sql(db_path, p, timeout)
        hits.append(0 if err is not None else int(set(rows) == gold_set))
    return {"qid": qid, "db": db, "skipped": False, "n_cand": len(preds),
            "oracle": int(any(hits)), "first": int(hits[0]) if hits else 0}


def main():
    ap = argparse.ArgumentParser(description="Execution-accuracy eval for DexterSQL.")
    ap.add_argument("--pred", required=True, help="predictions JSONL from run_dextersql.py")
    ap.add_argument("--db-root", default="", help="dir containing <db_id>/<db_id>.sqlite")
    ap.add_argument("--out", default=None, help="write a JSON report here")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args()

    tasks, missing_db = [], 0
    for line in open(a.pred):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        gold = d.get("gold_sql", "")
        if not gold:
            continue                       # nothing to score against
        db = d.get("db_id", "")
        db_path = d.get("db_path") or (
            os.path.join(a.db_root, db, f"{db}.sqlite") if a.db_root else "")
        if not db_path or not os.path.exists(db_path):
            missing_db += 1
            continue
        tasks.append((d.get("question_id"), db, db_path, gold,
                      d.get("sql_candidates") or [], a.timeout))

    if not tasks:
        raise SystemExit("[eval] nothing to score — check --db-root and that "
                         "predictions carry gold_sql")
    if missing_db:
        print(f"[eval] WARNING: skipped {missing_db} rows with no readable database")

    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(eval_one, t) for t in tasks]):
            results.append(f.result())

    scored = [r for r in results if not r["skipped"]]
    skipped = len(results) - len(scored)
    if not scored:
        raise SystemExit("[eval] every gold query failed to execute")

    n = len(scored)
    oracle = sum(r["oracle"] for r in scored)
    first = sum(r["first"] for r in scored)
    no_cand = sum(1 for r in scored if r["n_cand"] == 0)

    by_db = defaultdict(lambda: [0, 0, 0])
    for r in scored:
        by_db[r["db"]][0] += r["oracle"]
        by_db[r["db"]][1] += r["first"]
        by_db[r["db"]][2] += 1

    print(f"==== DexterSQL EX ({a.pred}) ====")
    print(f"scored={n}  gold-fail dropped={skipped}  questions with 0 candidates={no_cand}")
    print(f"  oracle EX (any candidate) : {oracle}/{n} = {oracle / n:.2%}")
    print(f"  first-candidate EX        : {first}/{n} = {first / n:.2%}")
    print("  per-database (oracle / first / n):")
    for db, (o, f_, c) in sorted(by_db.items()):
        print(f"    {db:30s} {o:4d} / {f_:4d} / {c:4d}   "
              f"oracle={o / c:6.1%}  first={f_ / c:6.1%}")

    if a.out:
        report = {
            "pred_file": a.pred, "scored": n, "gold_fail_dropped": skipped,
            "questions_without_candidates": no_cand,
            "oracle_ex": oracle / n, "first_ex": first / n,
            "oracle_correct": oracle, "first_correct": first,
            "by_database": {db: {"oracle": o, "first": f_, "total": c,
                                 "oracle_ex": o / c, "first_ex": f_ / c}
                            for db, (o, f_, c) in sorted(by_db.items())},
        }
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  report -> {a.out}")


if __name__ == "__main__":
    main()
