import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from dextersql.core.dataset import BaseDataset, load_dataset, save_dataset, DataItem
from dextersql.core.llm import LLM, make_llm
from dextersql.core.llm_extractor import LLMExtractor
from dextersql.core.logger import logger
from dextersql.core.pipeline.validation import validate_pipeline_step
from dextersql.core.services import (
    ArtifactStore,
    STAGE_ARTIFACT_FIELDS,
    configure_schema_service,
    get_schema_service,
    load_stage_dataset,
    reset_schema_service,
)
from tqdm import tqdm

from .prompts import format_rule_correction_prompt
from .example_rules import RULES, load_rules


class SQLRuleCorrectionRunner:
    """Applies the fixed correction rule set (see rules.RULES) to every sql_revision candidate.

    Sits between sql_revision and sql_selection. Unlike sql_revision (which
    fails an entire item to None on any candidate error, forcing a retry on
    resume), a single candidate's LLM call failing or returning unparseable
    output here falls back to that candidate's pre-correction SQL rather than
    failing the item: this stage is a best-effort semantic cleanup layer, and
    losing one candidate to a flaky response should not block the item from
    reaching sql_selection with its other (still-valid) candidates.
    """

    _llm: LLM = None
    _dataset: BaseDataset = None
    _thread_pool_executor: ThreadPoolExecutor = None
    _inner_thread_pool_executor: ThreadPoolExecutor = None
    _extractor: LLMExtractor = None
    _artifact_store: ArtifactStore = None
    _extractor_max_retry: int = 3
    _stage_config = None
    _input_save_path: str = ""
    _dataset_config = None

    def __init__(self, stage_config, dataset_config, input_save_path: str, extractor_max_retry: int,
                 rules=None):
        self._stage_config = stage_config
        self._dataset_config = dataset_config
        self._input_save_path = input_save_path
        self._extractor_max_retry = extractor_max_retry
        self._artifact_store = ArtifactStore(
            self._stage_config.save_path,
            "sql_rule_correction",
            STAGE_ARTIFACT_FIELDS["sql_rule_correction"],
        )
        self._dataset, checkpoint_source = load_stage_dataset(
            load_dataset_fn=load_dataset,
            current_save_path=self._stage_config.save_path,
            fallback_load_path=self._input_save_path,
            artifact_store=self._artifact_store,
            stage_name="sql_rule_correction",
        )
        logger.info(f"Initialized SQL rule correction dataset from {checkpoint_source}")
        configure_schema_service(max_value_example_length=self._dataset_config.max_value_example_length)
        self._llm = make_llm(self._stage_config.llm)
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=self._stage_config.n_parallel)
        self._inner_thread_pool_executor = ThreadPoolExecutor(max_workers=max(1, self._stage_config.n_internal_parallel))
        self._extractor = LLMExtractor(max_retry=extractor_max_retry)
        self._rules = RULES if rules is None else rules
        self._rule_names = [r["rule_name"] for r in self._rules]
        logger.info(f"Using {len(self._rules)} correction rules: {self._rule_names}")

    @classmethod
    def from_config(cls, app_config=None) -> "SQLRuleCorrectionRunner":
        if app_config is None:
            from dextersql.core.config import get_config

            app_config = get_config()
        return cls(
            stage_config=app_config.sql_rule_correction_config,
            dataset_config=app_config.dataset_config,
            input_save_path=app_config.sql_revision_config.save_path,
            extractor_max_retry=app_config.llm_extractor_config.max_retry,
            rules=load_rules(getattr(app_config.sql_rule_correction_config, "rules_path", "")),
        )

    def _normalize_sql(self, sql: str) -> str:
        """Simple normalization to handle whitespace and case differences."""
        if not sql:
            return ""
        return " ".join(sql.split()).strip().lower()

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        try:
            match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
            if not match:
                logger.warning("[sql_rule_correction] No <result> tag found in LLM response")
                return None
            content = match.group(1).strip()
            fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", content, re.IGNORECASE)
            if fence:
                content = fence.group(1).strip()
            if not content:
                return None

            parsed = json.loads(content)
            applied_rules = parsed.get("applied_rules", [])
            corrected_sql = parsed.get("corrected_sql", "")

            if not isinstance(applied_rules, list):
                return None
            applied_rules = [r for r in applied_rules if r in self._rule_names]
            if not isinstance(corrected_sql, str) or not corrected_sql.strip():
                return None

            return {"applied_rules": applied_rules, "corrected_sql": corrected_sql.strip()}
        except Exception as e:
            logger.warning(f"[sql_rule_correction] Error parsing LLM response: {e}")
            logger.debug(f"[sql_rule_correction] Response content: {response}")
            return None

    def _build_prompt(self, sql: str, data_item: DataItem) -> Tuple[Optional[str], int]:
        max_prompt_len = self._llm.llm_config.max_model_len - self._llm.llm_config.max_tokens
        schema_service = get_schema_service()
        schema_to_use = getattr(data_item, "database_schema_after_schema_linking", None) or data_item.database_schema

        def prompt_format_func(schema_profile: str) -> str:
            return format_rule_correction_prompt(
                database_schema=schema_profile,
                question=data_item.question,
                evidence=data_item.evidence,
                sql=sql,
                rules=self._rules,
            )

        prompt, level_idx = schema_service.build_prompt_with_progressive_schema_stripping(
            schema_to_use,
            encoding_model_name=self._llm.llm_config.model,
            max_prompt_len=max_prompt_len,
            prompt_format_func=prompt_format_func,
            item_id=data_item.question_id,
            log_prefix="Rule Correction",
        )
        if prompt is not None and level_idx > 0:
            logger.warning(
                f"[sql_rule_correction] prompt for item {data_item.question_id} was too large. "
                f"Compressed using level {level_idx}"
            )
        return prompt, level_idx

    def _correct_one_candidate(self, sql: str, data_item: DataItem) -> Tuple[str, List[str], Dict[str, int]]:
        """Run the single rule-correction LLM call for one SQL candidate.

        Returns (corrected_or_original_sql, applied_rule_names, token_usage).
        Falls back to the original sql with no applied rules on any prompt-
        build, LLM, or parse failure -- see class docstring for why this
        stage prefers a soft per-candidate fallback over failing the item.
        """
        empty_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        prompt, _level = self._build_prompt(sql, data_item)
        if prompt is None:
            logger.error(
                f"[sql_rule_correction] Prompt for item {data_item.question_id} exceeds token "
                f"limit even at minimal schema. Leaving candidate unchanged."
            )
            return sql, [], empty_tokens

        try:
            results, token_usage = self._extractor.extract_with_retry(
                llm=self._llm,
                messages=[{"role": "user", "content": prompt}],
                rule_parser=self._parse_llm_response,
                fix_end_token=self._llm.llm_config.fix_end_token,
                end_token="</result>",
                n=1,
            )
        except Exception as e:
            logger.error(f"[sql_rule_correction] LLM call failed for item {data_item.question_id}: {e}")
            traceback.print_exc()
            return sql, [], empty_tokens

        if not results:
            logger.warning(
                f"[sql_rule_correction] No valid correction result for item {data_item.question_id}; "
                f"leaving candidate unchanged."
            )
            return sql, [], token_usage

        result = results[0]
        return result["corrected_sql"], result["applied_rules"], token_usage

    def _correct_sql(self, data_item: DataItem) -> None:
        start_time = time.time()
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        sql_candidates = data_item.sql_candidates_after_revision

        if not sql_candidates:
            logger.error(
                f"sql_candidates_after_revision is empty or None for item {data_item.question_id}, "
                f"setting sql_candidates_after_rule_correction to None"
            )
            data_item.sql_candidates_after_rule_correction = None
            data_item.rule_correction_applied_rules = {}
            data_item.rule_correction_time = time.time() - start_time
            data_item.rule_correction_llm_cost = total_token_usage
            data_item.total_time += data_item.rule_correction_time
            data_item.total_llm_cost = {
                "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.rule_correction_llm_cost["prompt_tokens"],
                "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.rule_correction_llm_cost["completion_tokens"],
                "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.rule_correction_llm_cost["total_tokens"],
            }
            return

        # Deduplicate candidates using normalized SQL as key (same approach as sql_revision).
        unique_candidates_map: Dict[str, str] = {}
        for sql in sql_candidates:
            norm_sql = self._normalize_sql(sql)
            if norm_sql not in unique_candidates_map:
                unique_candidates_map[norm_sql] = sql

        unique_norms = list(unique_candidates_map.keys())
        future_to_norm = {
            self._inner_thread_pool_executor.submit(
                self._correct_one_candidate, unique_candidates_map[norm], data_item
            ): norm
            for norm in unique_norms
        }

        norm_to_result: Dict[str, Tuple[str, List[str]]] = {}
        for future in as_completed(future_to_norm):
            norm = future_to_norm[future]
            corrected_sql, applied_rules, tokens = future.result()
            norm_to_result[norm] = (corrected_sql, applied_rules)
            total_token_usage["prompt_tokens"] += tokens["prompt_tokens"]
            total_token_usage["completion_tokens"] += tokens["completion_tokens"]
            total_token_usage["total_tokens"] += tokens["total_tokens"]

        final_candidates = []
        applied_rules_log: Dict[str, List[str]] = {}
        for sql in sql_candidates:
            norm = self._normalize_sql(sql)
            corrected_sql, applied_rules = norm_to_result[norm]
            final_candidates.append(corrected_sql)
            if applied_rules:
                applied_rules_log[norm] = applied_rules

        data_item.sql_candidates_after_rule_correction = final_candidates
        data_item.rule_correction_applied_rules = applied_rules_log
        data_item.rule_correction_time = time.time() - start_time
        data_item.rule_correction_llm_cost = total_token_usage
        data_item.total_time += data_item.rule_correction_time
        data_item.total_llm_cost = {
            "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.rule_correction_llm_cost["prompt_tokens"],
            "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.rule_correction_llm_cost["completion_tokens"],
            "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.rule_correction_llm_cost["total_tokens"],
        }

    def run(self):
        future_to_item = {}
        for data_item in self._dataset:
            if data_item.is_stage_complete("sql_rule_correction"):
                logger.info(f"Skipping data item {data_item.question_id} because it has already been rule-corrected")
                continue
            future = self._thread_pool_executor.submit(self._correct_sql, data_item)
            future_to_item[future] = data_item
        for idx, future in tqdm(enumerate(as_completed(future_to_item), start=1), total=len(future_to_item), desc="Applying Rule Correction"):
            future.result()
            self._artifact_store.record_item(future_to_item[future])
            if idx % 5 == 0:
                logger.info(f"Applying Rule Correction {idx} / {len(future_to_item)} completed")
                self.save_result()
        logger.info("Applying Rule Correction completed")

        self._artifact_store.flush()
        validate_pipeline_step(self._dataset, "sql_rule_correction")
        self.save_result(materialize_snapshot=True)

        self._clean_up()

    def save_result(self, materialize_snapshot: bool = False):
        self._artifact_store.flush()
        if materialize_snapshot:
            save_dataset(self._dataset, self._stage_config.save_path)
            self._artifact_store.cleanup()

    def _clean_up(self):
        if self._thread_pool_executor is not None:
            self._thread_pool_executor.shutdown(wait=True)
            self._thread_pool_executor = None
        if self._inner_thread_pool_executor is not None:
            self._inner_thread_pool_executor.shutdown(wait=True)
            self._inner_thread_pool_executor = None
        if self._artifact_store is not None:
            self._artifact_store.close()
        reset_schema_service()
        self._llm = None
        self._extractor = None
        self._dataset = None
