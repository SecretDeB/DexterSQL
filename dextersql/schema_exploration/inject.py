"""
Called from dextersql.pipeline.Pipeline.run(), immediately after the
schema_linking stage, and only when app_config.schema_exploration.enabled
is true. Rewrites the schema_linking stage's OWN snapshot file in place --
appending gated disambiguation notes to each item's `evidence` field -- so
every downstream stage (sql_generation, sql_revision, ...) picks up the
augmented evidence exactly the way it already reads schema_linking's
output, with no changes to their code.

When disabled (the default), this module's only visible effect on a
pipeline run is that this function is never called -- see the `if` guard
at the single call site in pipeline.py.
"""

from __future__ import annotations

from typing import Any, Dict

from .gate import candidates_for_question, gate_relevant_notes, load_notes_for_db, render_note

_NOTES_HEADER = "\n\n[Disambiguation notes -- relevant column distinctions for this question:]\n"

# The newline-independent part of the header. Used for the idempotency check
# because an item with no prior evidence stores `block.strip()`, which drops the
# header's leading newlines -- matching on _NOTES_HEADER itself would miss
# exactly those items (common in BIRD, where many questions have no hint).
_NOTES_MARKER = _NOTES_HEADER.strip()


def inject_schema_exploration_notes(app_config) -> Dict[str, int]:
    from dextersql.core.dataset import load_dataset, save_dataset
    from dextersql.core.llm import make_llm
    from dextersql.core.logger import logger

    cfg = app_config.schema_exploration_config
    if cfg.gate_llm is None:
        raise ValueError("schema_exploration.enabled=true requires a [schema_exploration.gate_llm] config section")

    llm = make_llm(cfg.gate_llm)
    snapshot_path = app_config.schema_linking_config.save_path

    ds = load_dataset(snapshot_path)
    n_items = n_candidates = n_triggered = n_already = 0
    notes_cache: Dict[str, Any] = {}

    for item in ds:
        n_items += 1

        # Idempotency guard: this function rewrites the snapshot it reads, so a
        # second run would otherwise append a duplicate note block to evidence
        # that already has one -- silently changing the prompt sql_generation
        # sees. Re-running the step (after an interruption, or just to be sure
        # it ran) must be a no-op, so skip items already carrying notes.
        if _NOTES_MARKER in (item.evidence or ""):
            n_already += 1
            continue

        db_id = item.database_id
        if db_id not in notes_cache:
            notes_cache[db_id] = load_notes_for_db(cfg.notes_dir, db_id)
        notes = notes_cache[db_id]
        if not notes:
            continue

        linked = item.final_linked_tables_and_columns or {}
        candidates = candidates_for_question(notes, linked)
        if not candidates:
            continue
        n_candidates += len(candidates)

        keep_idx = gate_relevant_notes(llm, item.question, item.evidence, candidates, max_keep=cfg.max_notes_per_question)
        fired = [candidates[i] for i in keep_idx]
        if fired:
            n_triggered += 1
            block = _NOTES_HEADER + "\n".join(render_note(n) for n in fired)
            item.evidence = (item.evidence + block) if item.evidence else block.strip()

    save_dataset(ds, snapshot_path)
    stats = {"n_items": n_items, "n_candidates": n_candidates,
             "n_triggered": n_triggered, "n_already_injected": n_already}
    logger.info(
        f"[schema_exploration] {n_items} items | {n_candidates} schema-linked candidate notes "
        f"| {n_triggered} questions received a note after gating"
        + (f" | {n_already} already had notes (skipped, re-run is a no-op)" if n_already else "")
    )
    return stats
