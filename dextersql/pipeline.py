"""
DexterSQL end-to-end pipeline.

Runs value_retrieval, then generation/revision/selection, from a raw dataset to a
scored result:

    1. value_retrieval   annotate the schema with example values matching the question
    2. sql_generation    three methods propose candidates (dc | icl | dep_tree)
    3. sql_revision      repair candidates against the checker suite
    4. sql_selection     pick one answer by confidence-aware pairwise voting

Schema linking is NOT one of this pipeline's stages: the validated linker
(`dextersql/linking/`, the ABC 5-fix linker) is run as its own step outside this
class, producing a snapshot that `sql_generation` reads directly. See the README's
"Run the whole pipeline" section for the full sequence.

Every stage checkpoints per item, so a re-run resumes rather than recomputing.
That matters at dataset scale: a stage interrupted at item 900/1534 picks up at 901.

The `dep_tree` generator is installed via a runtime rebind in
`install_dep_tree_generator()` — no stage code is edited, and the swap is asserted
rather than assumed, so a silent failure to install cannot go unnoticed. It is the
only third-generator implementation in this package; there is no stock fallback.

Usage
-----
    from dextersql.pipeline import Pipeline
    Pipeline(config_path="config/bird.toml").run_all()

    # or a subset
    Pipeline(config_path="...").run(stages=["sql_generation", "sql_selection"])
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

STAGES = ["value_retrieval", "sql_generation", "sql_revision", "sql_selection"]


def install_dep_tree_generator(fewshot_path: Optional[str] = None,
                               masked_path: Optional[str] = None,
                               example_style: str = "plain",
                               trace_dir: Optional[str] = None,
                               trace_per_db: int = 5,
                               trace_all: bool = False) -> str:
    """Install `dep_tree` as the third SQL generator.

    The generation stage resolves its generator classes by name from its own module
    namespace, so binding "SkeletonGenerator" there installs the implementation
    without touching stage code. Returns the installed class name.

    Raises if spaCy is unavailable or the few-shot pool is empty — either would
    silently degrade every question rather than fail loudly.
    """
    from .core.pipeline.sql_generation.generators.base import BaseSQLGenerator
    from .core.pipeline.sql_generation import sql_generation as _stage
    from .core.logger import logger
    from .deptree import DepTreeGenerator, FewShotStore, TraceRecorder
    from .deptree.parser import parse_question, nlp_error

    if parse_question("smoke test the parser") is None:
        raise RuntimeError(
            f"dep_tree requires spaCy ({nlp_error()}). "
            f"pip install spacy && python -m spacy download en_core_web_sm")

    store = None
    if fewshot_path:
        store = FewShotStore(fewshot_path=fewshot_path, masked_path=masked_path or None)
        if not store.pool:
            raise RuntimeError(
                f"few-shot pool at {fewshot_path} is empty — dep_tree would silently "
                f"run zero-shot on every question; refusing to start.")
        logger.info(f"[dep_tree] few-shot pool: {len(store.pool)} questions")

    tracer = TraceRecorder(trace_dir=trace_dir, per_db=trace_per_db, trace_all=trace_all)

    class _DepTreeStageGenerator(BaseSQLGenerator):
        """Adapter: stage-facing signature over the dep_tree generator."""

        def __init__(self, extractor_max_retry: Optional[int] = None, **_ignored):
            super().__init__(extractor_max_retry=extractor_max_retry)
            self._impl = DepTreeGenerator(fewshot_store=store,
                                          example_style=example_style,
                                          trace_recorder=tracer)

        def generate(self, data_item, llm, sampling_budget: int = 1):
            if sampling_budget == 0:
                return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            extractor = self._get_extractor()

            def llm_call(messages, n=1):
                # rule_parser=identity: dep_tree does its own <result> extraction.
                out, usage = extractor.extract_with_retry(
                    llm=llm, messages=messages, rule_parser=lambda r: r,
                    fix_end_token=llm.llm_config.fix_end_token,
                    end_token="</result>", n=n)
                return list(out or []), usage

            schema = (getattr(data_item, "database_schema_after_schema_linking", None)
                      or data_item.database_schema)
            cands, usage, _report = self._impl.generate_for(
                question=data_item.question,
                database_schema=schema,
                evidence=getattr(data_item, "evidence", "") or "",
                retrieved_values=getattr(data_item, "retrieved_values", None),
                dialect=getattr(data_item, "db_type", None) or "sqlite",
                db_id=getattr(data_item, "database_id", ""),
                question_id=getattr(data_item, "question_id", ""),
                sampling_budget=sampling_budget,
                llm_call=llm_call)
            return cands, usage

    _stage.SkeletonGenerator = _DepTreeStageGenerator
    logger.info(f"[dep_tree] installed as the third SQL generator")
    return _DepTreeStageGenerator.__name__


class Pipeline:
    """Runs the DexterSQL stages against a config file."""

    def __init__(self, config_path: str,
                 fewshot_path: Optional[str] = None, masked_path: Optional[str] = None,
                 example_style: str = "plain", trace_dir: Optional[str] = None,
                 trace_per_db: int = 5, trace_all: bool = False):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config not found: {config_path}")
        # The vendored config layer reads this env var.
        os.environ["CONFIG_PATH"] = os.path.abspath(config_path)
        self.config_path = config_path
        self._dep_tree_kwargs = dict(
            fewshot_path=fewshot_path, masked_path=masked_path,
            example_style=example_style, trace_dir=trace_dir,
            trace_per_db=trace_per_db, trace_all=trace_all)
        self._installed = False

    # ── stage dispatch ───────────────────────────────────────────────────────
    def _runner(self, stage: str, app_config):
        from .core.pipeline import ValueRetrievalRunner, SQLGenerationRunner
        from .core.pipeline.sql_revision import SQLRevisionRunner
        from .core.pipeline.sql_selection import SQLSelectionRunner
        return {
            "value_retrieval": ValueRetrievalRunner,
            "sql_generation": SQLGenerationRunner,
            "sql_revision": SQLRevisionRunner,
            "sql_selection": SQLSelectionRunner,
        }[stage].from_config(app_config)

    def run(self, stages: Optional[List[str]] = None) -> Dict[str, float]:
        """Run the given stages (default: all) in pipeline order."""
        from .core.config import get_config
        from .core.logger import configure_logger, logger

        stages = stages or list(STAGES)
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; valid: {STAGES}")
        stages = [s for s in STAGES if s in stages]      # enforce order

        app_config = get_config()
        configure_logger(app_config.logger_config.print_level)

        # Install before constructing the generation runner, which binds its
        # generators at construction time.
        if "sql_generation" in stages and not self._installed:
            install_dep_tree_generator(**self._dep_tree_kwargs)
            self._installed = True

        timings: Dict[str, float] = {}
        for stage in stages:
            logger.info(f"===== {stage} =====")
            t0 = time.time()
            runner = self._runner(stage, app_config)
            if stage == "sql_generation":
                actual = type(runner._skeleton_generator).__name__
                if actual != "_DepTreeStageGenerator":
                    raise RuntimeError(
                        f"dep_tree install did not take effect (runner holds {actual})")
                logger.info(f"[dep_tree] confirmed active: {actual}")
            runner.run()
            timings[stage] = (time.time() - t0) / 60.0
            logger.info(f"===== {stage} done in {timings[stage]:.1f} min =====")

        total = sum(timings.values())
        logger.info(f"pipeline complete in {total:.1f} min: "
                    + ", ".join(f"{k}={v:.1f}m" for k, v in timings.items()))
        return timings

    def run_all(self) -> Dict[str, float]:
        return self.run(stages=list(STAGES))
