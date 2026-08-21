#!/usr/bin/env python3
"""
Run the DexterSQL pipeline: dataset in, predictions out.

    value_retrieval -> sql_generation -> sql_revision -> sql_selection

Schema linking is NOT a stage here — run the ABC linker (dextersql/linking/)
separately first; see the README's "Run the whole pipeline" section. This script
only covers value_retrieval and the generation/revision/selection tail, and
sql_generation always uses dep_tree as its third generator.

Every stage checkpoints per item, so re-running resumes where it stopped instead of
recomputing finished work.

Examples
--------
    # value_retrieval, then generation/revision/selection
    python scripts/run_pipeline.py --config config/bird.toml \
        --few-shot data/few_shots.json --trace-dir results/traces

    # resume / re-run only the tail
    python scripts/run_pipeline.py --config config/bird.toml \
        --stages sql_revision sql_selection

Then score the result:
    python scripts/evaluate.py --snapshot <sql_selection snapshot> --out results/perdb.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dextersql.pipeline import Pipeline, STAGES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run the DexterSQL pipeline.")
    ap.add_argument("--config", required=True, help="pipeline TOML config")
    ap.add_argument("--stages", nargs="*", default=None,
                    help=f"subset to run, in any order (default all): {' '.join(STAGES)}")

    ap.add_argument("--few-shot", default=os.environ.get("DEXTERSQL_FEWSHOT_PATH", ""),
                    help="few-shot pool JSON for dep_tree (omit -> zero-shot)")
    ap.add_argument("--masked", default=os.environ.get("DEXTERSQL_MASKED_PATH", ""),
                    help="masked-example JSON (only for non-default example styles)")
    ap.add_argument("--example-style", default="plain",
                    choices=["plain", "skeleton", "masked"])

    ap.add_argument("--trace-dir", default=None, help="dep_tree A-Z traces land here")
    ap.add_argument("--trace-per-db", type=int, default=5,
                    help="trace the first N questions per database")
    ap.add_argument("--trace-all", action="store_true",
                    help="trace every question (expensive at dataset scale)")
    a = ap.parse_args()

    pipe = Pipeline(
        config_path=a.config,
        fewshot_path=a.few_shot or None,
        masked_path=a.masked or None,
        example_style=a.example_style,
        trace_dir=a.trace_dir,
        trace_per_db=a.trace_per_db,
        trace_all=a.trace_all,
    )
    timings = pipe.run(stages=a.stages)

    print("\n==== stage timings (min) ====")
    for k, v in timings.items():
        print(f"  {k:18s} {v:7.1f}")
    print(f"  {'TOTAL':18s} {sum(timings.values()):7.1f}")
    print("\nScore the run with:")
    print("  python scripts/evaluate.py --snapshot <sql_selection snapshot> --out perdb.json")


if __name__ == "__main__":
    main()
