"""
Few-shot example loading and rendering.

Two sources, joined on normalized SQL text:
  * a few-shot pool JSON — cross-domain question->SQL pairs keyed by question id,
                           e.g. {"0": [{"question": ..., "sql": ...}, ...], ...}
                           Typically ~9 retrieved examples per target question.
  * a masked-example JSON (optional) — training examples where DB elements are
                           stripped: tables -> TB, columns -> CB_n, literals -> 0:
                             SELECT CB_2 FROM TB WHERE CB_3 = 0 ORDER BY CB_6 DESC
                           A pure STRUCTURAL skeleton, which lines up directly with
                           the Skeleton step of the generation prompt.

The two are joined on normalized SQL, so an example can carry its masked skeleton
alongside its concrete SQL.

Render styles
-------------
plain     Question -> SQL                    (DEFAULT; used by the dep_tree method)
skeleton  Question -> Skeleton -> SQL        (demonstrates the Skeleton -> Complete move)
masked    Question -> Skeleton only          (structure transfer with NO concrete
                                              columns to copy; forces adaptation)

`plain` is the configuration validated end-to-end; the other two are provided for
ablation and are not the documented default.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# Paths are deployment-specific: pass them explicitly, or set these env vars.
DEFAULT_FEWSHOT = os.environ.get("DEXTERSQL_FEWSHOT_PATH", "")
DEFAULT_MASKED = os.environ.get("DEXTERSQL_MASKED_PATH", "")

RENDER_STYLES = ("plain", "skeleton", "masked")


def _norm_sql(s: str) -> str:
    return " ".join((s or "").split()).lower()


class FewShotStore:
    """Loads the few-shot pool once and (optionally) indexes masked skeletons onto it."""

    def __init__(self, fewshot_path: str = DEFAULT_FEWSHOT,
                 masked_path: Optional[str] = DEFAULT_MASKED):
        self.pool: Dict[str, List[Dict[str, Any]]] = {}
        self.masked_by_sql: Dict[str, str] = {}
        self.n_joined = 0
        self.n_examples = 0

        if fewshot_path and os.path.exists(fewshot_path):
            with open(fewshot_path) as f:
                self.pool = json.load(f)

        if masked_path and os.path.exists(masked_path):
            with open(masked_path) as f:
                rows = json.load(f)
            for r in rows:
                sql = r.get("original_sql") or ""
                m = r.get("masked_sql") or ""
                if sql and m:
                    self.masked_by_sql.setdefault(_norm_sql(sql), m)

    def get(self, question_id: Any) -> List[Dict[str, Any]]:
        """The retrieved examples for this question id (empty list if none)."""
        return self.pool.get(str(question_id)) or []

    def masked_for(self, sql: str) -> Optional[str]:
        return self.masked_by_sql.get(_norm_sql(sql))

    def render(self, question_id: Any, style: str = "plain",
               limit: Optional[int] = None) -> str:
        """Render this question's examples in the requested style.

        Examples with no masked counterpart are SKIPPED in 'masked' style (showing
        nothing beats showing a half-masked example), but kept in 'skeleton' style
        with the skeleton line omitted, so the example is not wasted.
        """
        if style not in RENDER_STYLES:
            raise ValueError(f"unknown style {style!r}; choose from {RENDER_STYLES}")
        exs = self.get(question_id)
        if limit:
            exs = exs[:limit]
        out: List[str] = []
        i = 0
        for e in exs:
            q = (e.get("question") or "").strip()
            sql = (e.get("sql") or "").strip()
            if not q or not sql:
                continue
            masked = self.masked_for(sql)
            self.n_examples += 1
            if masked:
                self.n_joined += 1

            if style == "plain":
                i += 1
                out.append(f"- Example {i}:\nQuestion: {q}\nSQL: {sql}")
            elif style == "skeleton":
                i += 1
                if masked:
                    out.append(f"- Example {i}:\nQuestion: {q}\n"
                               f"Skeleton: {masked}\nSQL: {sql}")
                else:
                    out.append(f"- Example {i}:\nQuestion: {q}\nSQL: {sql}")
            else:  # masked
                if not masked:
                    continue
                i += 1
                out.append(f"- Example {i}:\nQuestion: {q}\nQuery structure: {masked}")
        return "\n".join(out) if out else "(none available)"


# Instruction text injected alongside each style, so the model knows what it is
# looking at and how to use it.
STYLE_INSTRUCTIONS = {
    "plain": (
        "You are also given several solved examples from OTHER databases. Study their\n"
        "SQL patterns (aggregations, JOIN structure, filtering, ordering) and adapt the\n"
        "patterns — never the literal table/column names — to the target schema."
    ),
    "skeleton": (
        "You are also given several solved examples from OTHER databases. Each shows the\n"
        "question, its SKELETON (a schema-independent structure where tables are TB,\n"
        "columns are CB_n, and literals are 0), and the final SQL. Use them as worked\n"
        "demonstrations of Step 2 -> Step 3: first fix the query STRUCTURE, then bind it\n"
        "to the target schema. Adapt patterns, never literal names."
    ),
    "masked": (
        "You are also given the QUERY STRUCTURES of several solved questions from OTHER\n"
        "databases, with all schema elements masked (tables are TB, columns are CB_n,\n"
        "literals are 0). They show only the shape of the SQL that such a question\n"
        "requires — which clauses appear, how they nest. Use them to choose your Step 2\n"
        "skeleton, then bind that skeleton to the target schema yourself."
    ),
}
