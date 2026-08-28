"""
Rule Creator orchestrator and CLI.

Runs the four steps in order, each reading the previous step's output file
and writing its own -- so the pipeline can be resumed or re-run from any
step without repeating earlier (expensive) LLM work:

  1. sampling.sample_training_questions      -> 01_sampled_questions.json
  2. error_mining.run_error_mining           -> 02_fail_candidate.jsonl
  3. db_agnostic_filter.run_db_agnostic_filter -> 03_fail_final.jsonl
     clustering.run_clustering               -> 04_error_groups.json
  4. rule_synthesis.run_rule_synthesis       -> 05_created_rules.json

(Step numbers above match the paper's four-step methodology; file numbers
match pipeline order, where Step 3 alone produces two files -- filtering
then clustering both belong to "keep only database-agnostic errors, then
cluster them", matching config.py's DbAgnosticFilterConfig/ClusteringConfig
split.)

Usage:
    python -m dextersql.rule_creator.pipeline --config path/to/rule_creator.toml
    python -m dextersql.rule_creator.pipeline --config ... --start-step 3
    python -m dextersql.rule_creator.pipeline --config ... --stop-step 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dextersql.rule_creator.config import RuleCreatorConfig, load_rule_creator_config
from dextersql.rule_creator.db_agnostic_filter import run_db_agnostic_filter
from dextersql.rule_creator.clustering import run_clustering
from dextersql.rule_creator.error_mining import run_error_mining
from dextersql.rule_creator.rule_synthesis import run_rule_synthesis
from dextersql.rule_creator.sampling import sample_training_questions

STEP_NAMES = {1: "error_mining (sampling + generation + explanation)", 2: "db_agnostic_filter", 3: "clustering", 4: "rule_synthesis"}


def run_pipeline(config: RuleCreatorConfig, start_step: int = 1, stop_step: int = 4) -> None:
    print(f"{'=' * 70}\nRule Creator: steps {start_step}-{stop_step}\n{'=' * 70}")
    print(f"train_json_path:       {config.train_json_path}")
    print(f"train_databases_root:  {config.train_databases_root}")
    print(f"output_dir:            {config.output_dir}")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    samples = None
    fail_candidates = None
    fail_final = None
    error_groups = None

    if start_step <= 1:
        print(f"\n--- Step 1: {STEP_NAMES[1]} ---")
        samples = sample_training_questions(config)
        fail_candidates = run_error_mining(config, samples)
    if stop_step < 2:
        return

    if start_step <= 2:
        print(f"\n--- Step 2: {STEP_NAMES[2]} ---")
        fail_final = run_db_agnostic_filter(config, fail_candidates)
    if stop_step < 3:
        return

    if start_step <= 3:
        print(f"\n--- Step 3: {STEP_NAMES[3]} ---")
        error_groups = run_clustering(config, fail_final)
    if stop_step < 4:
        return

    if start_step <= 4:
        print(f"\n--- Step 4: {STEP_NAMES[4]} ---")
        run_rule_synthesis(config, error_groups)

    print(f"\n{'=' * 70}\nDONE. Candidate rules -> {config.created_rules_path}\n{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule Creator: mine database-agnostic SQL-generation error rules from BIRD training data.")
    parser.add_argument("--config", required=True, help="Path to a rule_creator.toml config (see config/rule_creator.toml.template)")
    parser.add_argument("--start-step", type=int, default=1, choices=[1, 2, 3, 4], help="Resume from this step, reusing earlier steps' saved output files")
    parser.add_argument("--stop-step", type=int, default=4, choices=[1, 2, 3, 4], help="Stop after this step")
    args = parser.parse_args()

    config = load_rule_creator_config(args.config)
    run_pipeline(config, start_step=args.start_step, stop_step=args.stop_step)


if __name__ == "__main__":
    main()
