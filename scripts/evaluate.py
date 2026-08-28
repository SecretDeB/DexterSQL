#!/usr/bin/env python3
"""
Score a finished DexterSQL run: overall, per-database, and per-difficulty EX.

Execution accuracy compares result sets as unordered sets of rows —
set(pred_rows) == set(gold_rows). Questions whose GOLD query fails to execute are
dropped rather than counted wrong, so a broken gold query cannot depress the score.

Usage
-----
    python scripts/evaluate.py \
        --snapshot results/snapshots/sql_selection/bird/dev.snapshot \
        --out results/perdb.json --label "DexterSQL (dep_tree)"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TIMEOUT = int(os.environ.get("DEXTERSQL_EVAL_TIMEOUT", "120"))


def _eval_one(args):
    pred_sql, gold_sql, db_path = args
    from dextersql.core.db_utils import execute_sql
    pr = execute_sql(db_path, pred_sql, timeout=_TIMEOUT)
    gr = execute_sql(db_path, gold_sql, timeout=_TIMEOUT)
    if gr.result_rows is None:
        return None                       # gold itself failed -> drop the question
    if pr.result_rows is None:
        return 0                          # prediction failed to execute
    return 1 if set(pr.result_rows) == set(gr.result_rows) else 0


def _attr(item, *names, default=None):
    """Read a field whether the snapshot exposes it flat or under `input`."""
    for n in names:
        if hasattr(item, n):
            return getattr(item, n)
    inp = getattr(item, "input", None)
    if isinstance(inp, dict):
        for n in names:
            if n in inp:
                return inp[n]
    return default


def main():
    ap = argparse.ArgumentParser(description="Execution-accuracy scoring for DexterSQL.")
    ap.add_argument("--snapshot", required=True, help="sql_selection stage snapshot")
    ap.add_argument("--out", default=None, help="write a JSON report here")
    ap.add_argument("--label", default="DexterSQL")
    ap.add_argument("--max-workers", type=int, default=16)
    a = ap.parse_args()

    from dextersql.core.dataset import load_dataset

    ds = load_dataset(a.snapshot)
    tasks, meta = [], []
    for it in ds:
        pred = getattr(it, "final_selected_sql", None) or ""
        gold = _attr(it, "gold_sql", default="") or ""
        db_path = _attr(it, "database_path", default="") or ""
        tasks.append((pred, gold, db_path))
        meta.append({
            "question_id": _attr(it, "question_id"),
            "database_id": _attr(it, "database_id", default=""),
            "difficulty": _attr(it, "difficulty", default="") or "",
            "predicted_sql": pred,
        })

    with ProcessPoolExecutor(max_workers=a.max_workers) as ex:
        scores = list(ex.map(_eval_one, tasks))

    by_db = defaultdict(lambda: [0, 0])
    by_diff = defaultdict(lambda: [0, 0])
    per_q, correct, scored, gold_fail = [], 0, 0, 0

    for m, s in zip(meta, scores):
        if s is None:
            gold_fail += 1
            m["score"] = None
            per_q.append(m)
            continue
        scored += 1
        correct += s
        by_db[m["database_id"]][0] += s
        by_db[m["database_id"]][1] += 1
        if m["difficulty"]:
            by_diff[m["difficulty"]][0] += s
            by_diff[m["difficulty"]][1] += 1
        m["score"] = s
        per_q.append(m)

    acc = correct / scored if scored else 0.0
    summary = {
        "label": a.label,
        "snapshot": a.snapshot,
        "total": len(meta),
        "scored": scored,
        "gold_fail": gold_fail,
        "correct": correct,
        "overall_accuracy": round(acc, 4),
        "by_database": {k: {"correct": v[0], "total": v[1],
                            "accuracy": round(v[0] / v[1], 4)}
                        for k, v in sorted(by_db.items())},
        "by_difficulty": {k: {"correct": v[0], "total": v[1],
                              "accuracy": round(v[0] / v[1], 4)}
                          for k, v in sorted(by_diff.items())},
    }

    print(f"\n==== {a.label} ====")
    print(f"overall: {correct}/{scored} = {acc:.2%}   (gold-fail dropped: {gold_fail})")
    print("By database:")
    for k, v in summary["by_database"].items():
        print(f"  {k:28s} {v['correct']:4d}/{v['total']:<4d} = {v['accuracy']:.1%}")
    if summary["by_difficulty"]:
        print("By difficulty:")
        for k, v in summary["by_difficulty"].items():
            print(f"  {k:28s} {v['correct']:4d}/{v['total']:<4d} = {v['accuracy']:.1%}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "per_question": per_q}, f, indent=2)
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
