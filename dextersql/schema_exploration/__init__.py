"""
Deep Schema Exploration -- offline column-ambiguity mining + an online,
gated note-injection step.

Two halves:
  - Offline (run once per database, see offline_pipeline.py): finds pairs of
    columns in different tables that share a name/values but a different
    real-world meaning, computes deterministic statistics about them (join
    coverage, fan-out, value agreement, ...), and has an LLM turn each
    genuinely ambiguous pair into a short disambiguation note. Output:
    one ambiguity_notes_<db_id>.json per database.
  - Online (inject.py, wired into dextersql.pipeline.Pipeline.run): right
    after the schema_linking stage, if enabled, filters each question's
    candidate notes down to the ones touching a linked column, asks an LLM
    which of those are actually needed for THIS question, and appends the
    survivors to the item's `evidence` field before generation reads it.

Fully opt-in: with schema_exploration.enabled unset or false (the default,
and the state of every config that produced a confirmed benchmark run so
far), none of this module's code executes and the pipeline behaves exactly
as it did before this module existed.
"""
