"""
Online relevance gate: turns a database's offline-mined notes into the
short list that's actually worth showing for ONE question.

Direct injection (every note whose column is schema-linked, unconditionally
appended) tends to mislead about as often as it helps -- most questions
that touch a flagged column don't actually depend on the specific
distinction the note describes. The gate is a second, cheap LLM call that
looks at the actual question and keeps only the notes it judges essential,
and it fails CLOSED (keeps nothing) on any parse error, since over-
injection is the documented failure mode here -- an unparseable gate
response should never fall back to dumping every note.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def render_note(note: Dict[str, Any]) -> str:
    cols = note.get("columns", [])
    if len(cols) < 2:
        return note.get("sql_note", "")
    a, b = cols[0], cols[1]
    parts = [f"- {a} vs {b}:"]
    if note.get("col_a_purpose"):
        parts.append(f"{a} = {note['col_a_purpose']}")
    if note.get("col_b_purpose"):
        parts.append(f"{b} = {note['col_b_purpose']}")
    extra = []
    if note.get("use_a_when"):
        extra.append(f"use {a} when {note['use_a_when']}")
    if note.get("use_b_when"):
        extra.append(f"use {b} when {note['use_b_when']}")
    if note.get("sql_note"):
        extra.append(note["sql_note"])
    if note.get("caution"):
        extra.append(f"CAUTION: {note['caution']}")
    return " ".join(parts) + (" " + "; ".join(extra) if extra else "")


def load_notes_for_db(notes_dir: str, db_id: str) -> List[Dict[str, Any]]:
    path = Path(notes_dir) / f"ambiguity_notes_{db_id}.json"
    if not path.exists():
        return []
    try:
        return json.load(open(path)).get("notes", [])
    except Exception:
        return []


def candidates_for_question(notes: List[Dict[str, Any]], linked_tables_and_columns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """A note fires if EITHER of its columns is in the schema-linked set --
    a mechanical both-columns rule would suppress exactly the case where the
    linker silently kept one side of an ambiguous pair, which is the case
    most worth catching."""
    linked = {f"{t}.{c}".lower() for t, cs in linked_tables_and_columns.items() for c in cs}
    return [n for n in notes if any(c.lower() in linked for c in n.get("columns", []))]


def gate_relevant_notes(llm, question: str, evidence: str, candidates: List[Dict[str, Any]], max_keep: int = 0) -> List[int]:
    """Returns indices into `candidates` that are ESSENTIAL for writing the
    SQL for this question. Fails closed (returns []) on any parse error."""
    if not candidates:
        return []
    listing = "\n".join(f"[{i}] {render_note(c)}" for i, c in enumerate(candidates))
    prompt = (
        f"You are about to write a SQL query (SQLite) to answer this question:\n"
        f"QUESTION: {question}\n"
        + (f"HINT: {evidence}\n" if evidence else "")
        + "\nThe database has some columns that are easy to confuse. Below are notes "
        "about them. Decide which notes are ESSENTIAL for deciding how to write the "
        "SQL for THIS question -- i.e. a note you genuinely need so you do not pick "
        "the wrong column. Keep a note only if the question actually uses one of the "
        "columns it distinguishes AND a reasonable writer might otherwise choose the "
        "wrong one. Skip notes about columns this question does not use, and notes "
        "where the right column is already obvious. It is normal to keep none.\n\n"
        f"NOTES:\n{listing}\n\n"
        + (f"Keep AT MOST {max_keep} note(s) -- only the single most essential. " if max_keep else "")
        + "Respond with ONLY a JSON list of the indices of the notes to keep "
        "(for example [0,2]), or [] if none are needed. Output just the list."
    )
    try:
        responses, _usage = llm.ask(
            [{"role": "user", "content": prompt}],
            system_message={
                "role": "system",
                "content": "You select which disambiguation notes are essential for writing a SQL "
                "query. Reply with only a JSON list of integers, e.g. [0,2] or [].",
            },
            n=1,
            max_tokens=1024,
            temperature=0.0,
        )
        text = responses[0].content or ""
        matches = re.findall(r"\[[0-9,\s]*\]", text)
        if not matches:
            return []
        idx = [i for i in json.loads(matches[-1]) if isinstance(i, int) and 0 <= i < len(candidates)]
        if max_keep:
            idx = idx[:max_keep]
        return idx
    except Exception:
        return []
