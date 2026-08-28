"""
Step 1 (Error Mining).

For each sampled training question: generate multiple candidate SQLs,
execute each against the training database, discard candidates whose
result matches gold, and for every remaining incorrect candidate ask an
LLM to explain -- possibly as multiple separate explanations -- why it's
wrong compared to gold.

Input: the sampled question list from sampling.sample_training_questions
(or config.sampled_path). Output: config.fail_candidate_path, one JSON
object per line, one line per (candidate, explanation) pair:
  {question_id, db_id, question, evidence, gold_sql, candidate_sql, explanation}
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

from dextersql.core.db_utils import execute_sql
from dextersql.core.llm import make_llm
from dextersql.core.llm_extractor import LLMExtractor
from dextersql.core.logger import logger
from dextersql.core.services import configure_schema_service, get_schema_service
from dextersql.rule_creator.config import RuleCreatorConfig
from dextersql.rule_creator.prompts import format_failure_explanation_prompt, format_sql_generation_prompt
from dextersql.rule_creator.parsing import parse_explanation_list, parse_sql_result


def _normalize_sql(sql: str) -> str:
    return " ".join((sql or "").split()).strip().lower()


def _db_path(config: RuleCreatorConfig, db_id: str) -> str:
    return str(Path(config.train_databases_root) / db_id / f"{db_id}.sqlite")


def _load_already_done(path: str) -> Set[int]:
    done: Set[int] = set()
    if not Path(path).exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _mine_one_question(config: RuleCreatorConfig, item: Dict[str, Any], generation_llm, explanation_llm) -> List[Dict[str, Any]]:
    extractor = LLMExtractor(max_retry=3)
    db_path = _db_path(config, item["db_id"])
    schema = get_schema_service().load_sqlite_schema(db_path)

    gold_sql = item["gold_sql"]
    gold_result = execute_sql(db_path, gold_sql, timeout=config.error_mining.execution_timeout)
    if gold_result.result_rows is None:
        logger.warning(f"[error_mining] gold SQL failed to execute for question_id={item['question_id']} db={item['db_id']}, skipping")
        return []
    gold_rows = set(gold_result.result_rows)

    gen_prompt = format_sql_generation_prompt(schema, item["question"], item["evidence"])
    candidates, _ = extractor.extract_with_retry(
        llm=generation_llm,
        messages=[{"role": "user", "content": gen_prompt}],
        rule_parser=parse_sql_result,
        n=config.error_mining.candidates_per_question,
    )

    wrong_unique: Dict[str, str] = {}  # normalized -> original candidate SQL
    for cand_sql in candidates:
        norm = _normalize_sql(cand_sql)
        if not norm or norm in wrong_unique:
            continue
        cand_result = execute_sql(db_path, cand_sql, timeout=config.error_mining.execution_timeout)
        if cand_result.result_rows is not None and set(cand_result.result_rows) == gold_rows:
            continue  # correct, discard per the methodology
        wrong_unique[norm] = cand_sql

    if not wrong_unique:
        return []

    rows: List[Dict[str, Any]] = []
    for cand_sql in wrong_unique.values():
        explain_prompt = format_failure_explanation_prompt(schema, item["question"], item["evidence"], gold_sql, cand_sql)
        results, _ = extractor.extract_with_retry(
            llm=explanation_llm,
            messages=[{"role": "user", "content": explain_prompt}],
            rule_parser=parse_explanation_list,
            n=1,
        )
        explanations = results[0] if results else []
        for explanation in explanations:
            rows.append({
                "question_id": item["question_id"],
                "db_id": item["db_id"],
                "question": item["question"],
                "evidence": item["evidence"],
                "gold_sql": gold_sql,
                "candidate_sql": cand_sql,
                "explanation": explanation,
            })
    return rows


def run_error_mining(config: RuleCreatorConfig, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    configure_schema_service()
    generation_llm = make_llm(config.error_mining.generation_llm)
    explanation_llm = make_llm(config.error_mining.explanation_llm)

    out_path = Path(config.fail_candidate_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = _load_already_done(config.fail_candidate_path)
    todo = [s for s in samples if s["question_id"] not in already_done]
    print(f"[error_mining] {len(samples)} sampled, {len(already_done)} already done, {len(todo)} to process")

    all_rows: List[Dict[str, Any]] = []
    if already_done:
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))

    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=config.error_mining.n_parallel) as pool:
        futures = {
            pool.submit(_mine_one_question, config, item, generation_llm, explanation_llm): item
            for item in todo
        }
        completed = 0
        for future in as_completed(futures):
            item = futures[future]
            try:
                rows = future.result()
            except Exception as e:
                logger.warning(f"[error_mining] question_id={item['question_id']} failed: {e}")
                rows = []
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            all_rows.extend(rows)
            completed += 1
            if completed % 10 == 0 or completed == len(todo):
                print(f"[error_mining] {completed}/{len(todo)} questions processed, {len(all_rows)} failure explanations so far")

    print(f"[error_mining] done: {len(all_rows)} failure explanations -> {out_path}")
    return all_rows
