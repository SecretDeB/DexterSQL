"""The dep_tree generation method: dependency-grounded, single-call SQL generation."""

from .generator import DepTreeGenerator, build_prompt, DEP_TREE_PROMPT
from .fewshot import FewShotStore, RENDER_STYLES, STYLE_INSTRUCTIONS
from .trace import TraceRecorder, QuestionTrace
from .grounding import load_schema, Grounder, GroundedSchema
from .parser import parse_question, nlp_error
from .units import extract_units
from .ir import SQLIR
from .ir_builder import build_ir
from .validate import validate_ir, validate_sql

__all__ = [
    "DepTreeGenerator", "build_prompt", "DEP_TREE_PROMPT",
    "FewShotStore", "RENDER_STYLES", "STYLE_INSTRUCTIONS",
    "TraceRecorder", "QuestionTrace",
    "load_schema", "Grounder", "GroundedSchema",
    "parse_question", "nlp_error", "extract_units",
    "SQLIR", "build_ir", "validate_ir", "validate_sql",
]
