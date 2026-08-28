"""
A-Z trace: for the sampled questions, record EXACTLY what went to the LLM and what
was extracted from it, plus every deterministic intermediate (tree, units,
groundings, draft IR, validated IR, violations, repairs).

Default sampling: the first TRACE_PER_DB (5) questions PER DATABASE. Set
DEP_TREE_TRACE_ALL=1 to trace everything (expensive on 1534 questions).

One JSON file per question: <trace_dir>/<db_id>/q<qid>.json
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

TRACE_PER_DB_DEFAULT = 5


class TraceRecorder:
    """Thread-safe; batch runners drive items through a ThreadPoolExecutor."""

    def __init__(self, trace_dir: Optional[str] = None,
                 per_db: Optional[int] = None, trace_all: Optional[bool] = None):
        self.trace_dir = trace_dir or os.environ.get("DEP_TREE_TRACE_DIR", "")
        self.per_db = int(per_db if per_db is not None
                          else os.environ.get("DEP_TREE_TRACE_PER_DB", TRACE_PER_DB_DEFAULT))
        self.trace_all = bool(trace_all if trace_all is not None
                              else os.environ.get("DEP_TREE_TRACE_ALL") == "1")
        self._counts: Dict[str, int] = {}
        self._claimed: Dict[str, set] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.trace_dir)

    def should_trace(self, db_id: str, question_id: Any) -> bool:
        if not self.enabled:
            return False
        if self.trace_all:
            return True
        with self._lock:
            claimed = self._claimed.setdefault(db_id, set())
            if question_id in claimed:
                return True
            if self._counts.get(db_id, 0) < self.per_db:
                self._counts[db_id] = self._counts.get(db_id, 0) + 1
                claimed.add(question_id)
                return True
        return False

    def new(self, db_id: str, question_id: Any, question: str) -> "QuestionTrace":
        return QuestionTrace(self, db_id, question_id, question,
                             active=self.should_trace(db_id, question_id))


class QuestionTrace:
    def __init__(self, rec: TraceRecorder, db_id: str, question_id: Any,
                 question: str, active: bool):
        self._rec = rec
        self.active = active
        self.data: Dict[str, Any] = {
            "question_id": question_id, "database_id": db_id, "question": question,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": [], "llm_calls": [], "candidates": [], "stats": {},
        }

    # ── deterministic steps ──────────────────────────────────────────────────
    def step(self, name: str, payload: Any) -> None:
        """A-Z record of one deterministic stage (tree, units, grounding, IR, ...)."""
        if not self.active:
            return
        self.data["steps"].append({"step": name, "payload": _safe(payload)})

    # ── LLM I/O (exact prompt in, raw response out) ──────────────────────────
    def llm(self, role: str, messages: List[Dict[str, str]], raw_response: Any,
            parsed: Any = None, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.active:
            return
        self.data["llm_calls"].append({
            "role": role,
            "prompt_messages": messages,
            "raw_response": raw_response if isinstance(raw_response, (str, list)) else str(raw_response),
            "parsed": _safe(parsed),
            "meta": _safe(meta or {}),
        })

    def candidate(self, sql: str, variant: str, violations: List[str],
                  repaired: bool = False, accepted: bool = True) -> None:
        if not self.active:
            return
        self.data["candidates"].append({
            "sql": sql, "variant": variant, "violations": violations,
            "repaired": repaired, "accepted": accepted,
        })

    def stat(self, **kw) -> None:
        if not self.active:
            return
        self.data["stats"].update(_safe(kw))

    # ── flush ────────────────────────────────────────────────────────────────
    def save(self) -> Optional[str]:
        if not self.active or not self._rec.trace_dir:
            return None
        d = os.path.join(self._rec.trace_dir, str(self.data["database_id"]))
        try:
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f"q{self.data['question_id']}.json")
            self.data["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(p, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
            return p
        except Exception:
            return None                    # tracing must never break generation


def _safe(obj: Any) -> Any:
    """Best-effort JSON-able conversion (dataclasses, sets, objects)."""
    from dataclasses import is_dataclass, asdict
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe(x) for x in obj]
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return str(obj)
