"""
DexterSQL — an end-to-end text-to-SQL system.

Given a database and a natural-language question, DexterSQL runs the full pipeline:

    value retrieval -> schema linking -> SQL generation -> revision -> selection

SQL generation runs three complementary methods and pools their candidates:

    * divide-and-conquer   — decompose the question, solve the parts
    * in-context-learning  — adapt patterns from retrieved solved examples
    * dep_tree             — dependency-grounded generation (see dextersql.deptree)

Revision then repairs candidates against a checker suite, and selection picks a
single answer by confidence-aware pairwise voting.

Quick start
-----------
    # one-shot: dataset in, accuracy out
    python scripts/run_pipeline.py --config config/bird.toml

    # or use just the dep_tree generator as a library
    from dextersql.deptree import DepTreeGenerator, FewShotStore
    from dextersql import make_llm_call
"""

from .backends import OpenAICompatibleBackend, StubBackend, make_llm_call

# The dep_tree generator and its building blocks. Imported lazily-friendly: this
# subpackage needs spaCy, which the vendored pipeline stages do not.
from .deptree import (
    DepTreeGenerator, build_prompt, DEP_TREE_PROMPT,
    FewShotStore, RENDER_STYLES, STYLE_INSTRUCTIONS,
    TraceRecorder, QuestionTrace,
    load_schema, Grounder, GroundedSchema,
    parse_question, nlp_error, extract_units,
    SQLIR, build_ir, validate_ir, validate_sql,
)

__version__ = "1.0.0"

__all__ = [
    # backends
    "OpenAICompatibleBackend", "StubBackend", "make_llm_call",
    # dep_tree method
    "DepTreeGenerator", "build_prompt", "DEP_TREE_PROMPT",
    "FewShotStore", "RENDER_STYLES", "STYLE_INSTRUCTIONS",
    "TraceRecorder", "QuestionTrace",
    "load_schema", "Grounder", "GroundedSchema",
    "parse_question", "nlp_error", "extract_units",
    "SQLIR", "build_ir", "validate_ir", "validate_sql",
    "__version__",
]
