"""
Configuration for the Rule Creator pipeline.

Deliberately independent of dextersql.core.config.AppConfig, which is built
around DatasetConfig(split="dev"|"test") for the inference-time pipeline.
Rule Creator only ever reads BIRD *training* data, and giving it its own
config -- with explicit train_json_path / train_databases_root fields and no
concept of "split" at all -- makes that isolation structural rather than a
matter of remembering to pass the right flag.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from dextersql.core.config.config import LLMConfig


class SamplingConfig(BaseModel):
    """Step 1a: stratified sampling of training questions by difficulty."""

    target_size: int = Field(default=300, description="Total number of training questions to sample")
    seed: int = Field(default=42)
    simple_fraction: float = Field(default=0.40)
    moderate_fraction: float = Field(default=0.35)
    challenging_fraction: float = Field(default=0.25)

    @model_validator(mode="after")
    def _check_fractions(self):
        total = self.simple_fraction + self.moderate_fraction + self.challenging_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"sampling fractions must sum to 1.0, got {total}")
        return self


class ErrorMiningConfig(BaseModel):
    """Step 1: candidate generation, execution comparison, failure explanation."""

    candidates_per_question: int = Field(default=4, description="Candidate SQLs sampled per question")
    n_parallel: int = Field(default=4, description="Questions processed concurrently")
    execution_timeout: int = Field(default=30, description="Seconds before a SQL execution against a train DB is aborted")
    generation_llm: LLMConfig
    explanation_llm: LLMConfig


class DbAgnosticFilterConfig(BaseModel):
    """Step 2: keep only reusable, database-agnostic failure explanations."""

    n_parallel: int = Field(default=8)
    llm: LLMConfig


class ClusteringConfig(BaseModel):
    """Step 3: hierarchical clustering of database-agnostic explanations."""

    batch_size: int = Field(default=15, description="Explanations per batch-level clustering call, within one database")
    cross_db_merge_chunk_size: int = Field(
        default=20,
        description="Groups per merge call when merging across databases -- keeps the merge call's token budget from being overwhelmed (see README.md, lesson learned the hard way on a different clustering pass)",
    )
    min_group_support: int = Field(default=3, description="Groups with fewer supporting explanations than this are discarded as non-dominant")
    llm: LLMConfig


class RuleSynthesisConfig(BaseModel):
    """Step 4: convert each dominant error group into a canonical rule."""

    max_examples_per_rule: int = Field(default=6, description="Representative member explanations shown to the LLM per group, to keep the synthesis prompt bounded")
    llm: LLMConfig


class RuleCreatorConfig(BaseModel):
    # -- The only paths this pipeline ever reads. There is deliberately no
    # "split" concept and no path here that could resolve to dev/test data. --
    train_json_path: str = Field(..., description="Path to BIRD train.json (question/evidence/SQL/db_id per row)")
    train_databases_root: str = Field(..., description="Path to train_databases/ (one subfolder per db_id, each holding <db_id>.sqlite)")

    output_dir: str = Field(..., description="Where every step writes its checkpoint/output files")

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    error_mining: ErrorMiningConfig
    db_agnostic_filter: DbAgnosticFilterConfig
    clustering: ClusteringConfig
    rule_synthesis: RuleSynthesisConfig

    @model_validator(mode="after")
    def _check_train_paths_exist(self):
        if not Path(self.train_json_path).is_file():
            raise ValueError(f"train_json_path does not exist: {self.train_json_path}")
        if not Path(self.train_databases_root).is_dir():
            raise ValueError(f"train_databases_root is not a directory: {self.train_databases_root}")
        return self

    # -- output file paths, one per pipeline step, all under output_dir --
    @property
    def sampled_path(self) -> str:
        return str(Path(self.output_dir) / "01_sampled_questions.json")

    @property
    def fail_candidate_path(self) -> str:
        return str(Path(self.output_dir) / "02_fail_candidate.jsonl")

    @property
    def fail_final_path(self) -> str:
        return str(Path(self.output_dir) / "03_fail_final.jsonl")

    @property
    def error_groups_path(self) -> str:
        return str(Path(self.output_dir) / "04_error_groups.json")

    @property
    def created_rules_path(self) -> str:
        return str(Path(self.output_dir) / "05_created_rules.json")


def load_rule_creator_config(path: str) -> RuleCreatorConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    def llm(section_name: str, sub_key: str) -> LLMConfig:
        return LLMConfig(**raw[section_name][sub_key])

    sampling = SamplingConfig(**raw.get("sampling", {}))

    error_mining_raw = raw["error_mining"]
    error_mining = ErrorMiningConfig(
        candidates_per_question=error_mining_raw.get("candidates_per_question", 4),
        n_parallel=error_mining_raw.get("n_parallel", 4),
        execution_timeout=error_mining_raw.get("execution_timeout", 30),
        generation_llm=LLMConfig(**error_mining_raw["generation_llm"]),
        explanation_llm=LLMConfig(**error_mining_raw["explanation_llm"]),
    )

    db_agnostic_raw = raw["db_agnostic_filter"]
    db_agnostic_filter = DbAgnosticFilterConfig(
        n_parallel=db_agnostic_raw.get("n_parallel", 8),
        llm=LLMConfig(**db_agnostic_raw["llm"]),
    )

    clustering_raw = raw["clustering"]
    clustering = ClusteringConfig(
        batch_size=clustering_raw.get("batch_size", 15),
        cross_db_merge_chunk_size=clustering_raw.get("cross_db_merge_chunk_size", 20),
        min_group_support=clustering_raw.get("min_group_support", 3),
        llm=LLMConfig(**clustering_raw["llm"]),
    )

    rule_synthesis_raw = raw["rule_synthesis"]
    rule_synthesis = RuleSynthesisConfig(
        max_examples_per_rule=rule_synthesis_raw.get("max_examples_per_rule", 6),
        llm=LLMConfig(**rule_synthesis_raw["llm"]),
    )

    return RuleCreatorConfig(
        train_json_path=raw["data"]["train_json_path"],
        train_databases_root=raw["data"]["train_databases_root"],
        output_dir=raw["data"]["output_dir"],
        sampling=sampling,
        error_mining=error_mining,
        db_agnostic_filter=db_agnostic_filter,
        clustering=clustering,
        rule_synthesis=rule_synthesis,
    )
