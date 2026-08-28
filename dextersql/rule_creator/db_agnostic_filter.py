"""
Step 2 (Database-agnostic error isolation).

For every failure explanation from Step 1, ask an LLM whether it reflects a
reusable, database-agnostic SQL-formulation issue or a database-specific
artifact (wrong table/column pick, schema-linking mistake, value/enum
knowledge specific to one database). Only the former survive into Step 3.

Input: config.fail_candidate_path (from error_mining.run_error_mining).
Output: config.fail_final_path, same row shape as the input plus a
"db_agnostic_reasoning" field is dropped (kept out of the file on purpose --
the classification is binary and final by the time it's written).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from dextersql.core.llm import make_llm
from dextersql.core.llm_extractor import LLMExtractor
from dextersql.core.logger import logger
from dextersql.rule_creator.config import RuleCreatorConfig
from dextersql.rule_creator.parsing import parse_db_agnostic_classification
from dextersql.rule_creator.prompts import format_db_agnostic_classification_prompt


def _load_fail_candidates(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _classify_one(row: Dict[str, Any], llm) -> bool:
    """Returns True iff the explanation is database-agnostic (keep it)."""
    extractor = LLMExtractor(max_retry=3)
    prompt = format_db_agnostic_classification_prompt(
        row["question"], row["gold_sql"], row["candidate_sql"], row["explanation"]
    )
    results, _ = extractor.extract_with_retry(
        llm=llm,
        messages=[{"role": "user", "content": prompt}],
        rule_parser=parse_db_agnostic_classification,
        n=1,
    )
    if not results:
        logger.warning(f"[db_agnostic_filter] classification failed for question_id={row['question_id']}, discarding (fail closed)")
        return False
    return results[0] == "database_agnostic"


def run_db_agnostic_filter(config: RuleCreatorConfig, fail_candidates: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if fail_candidates is None:
        fail_candidates = _load_fail_candidates(config.fail_candidate_path)
    print(f"[db_agnostic_filter] classifying {len(fail_candidates)} explanations")

    llm = make_llm(config.db_agnostic_filter.llm)
    kept: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=config.db_agnostic_filter.n_parallel) as pool:
        futures = {pool.submit(_classify_one, row, llm): row for row in fail_candidates}
        completed = 0
        for future in as_completed(futures):
            row = futures[future]
            try:
                is_agnostic = future.result()
            except Exception as e:
                logger.warning(f"[db_agnostic_filter] question_id={row['question_id']} failed: {e}")
                is_agnostic = False
            if is_agnostic:
                kept.append(row)
            completed += 1
            if completed % 25 == 0 or completed == len(fail_candidates):
                print(f"[db_agnostic_filter] {completed}/{len(fail_candidates)} classified, {len(kept)} kept as database-agnostic")

    out_path = Path(config.fail_final_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    print(f"[db_agnostic_filter] {len(kept)}/{len(fail_candidates)} kept ({len(fail_candidates) - len(kept)} discarded as database-specific) -> {out_path}")
    return kept
