# Deep Schema Exploration

Offline column-ambiguity mining + a gated, opt-in online injection step.
Detects pairs of columns in different tables that share a name or values
but a different real-world meaning (e.g. `card.type` = card tier vs.
`disp.type` = account-holder role, both just called "type"), computes
deterministic statistics that expose the distinction, and has an LLM turn
each genuinely ambiguous pair into a short note. At inference time, if
enabled, a second LLM call decides -- per question -- which of a
database's notes are actually needed, and appends only those to the
question's `evidence` before generation reads it.

**Toggle guarantee.** `schema_exploration.enabled` defaults to `false`, and
every config that predates this module simply doesn't have the section at
all. In both cases `dextersql/pipeline.py`'s single call site is never
reached -- not "behaves the same", but structurally never executed. Turning
this on only ever adds text to `evidence`; it never touches any other
field, any other stage's code, or the STAGES list.

## Two halves

**Offline** (`offline_pipeline.py`, phases 0-4, run once per database,
independent of `config/bird.toml`):

1. **Phase 0 -- FK discovery** (`fk_discovery.py`): declared FKs
   (`PRAGMA foreign_key_list`) + name-matched FKs validated by a
   containment check (a child column's values must live inside its
   candidate parent's) + optionally an LLM proposing implicit FKs for any
   primary key still unreferenced, itself containment-checked before being
   trusted.
2. **Phase 1 -- candidate pairs** (`candidate_pairs.py`): exact
   column-name matches and fuzzy token-overlap matches across tables,
   with structurally-explained pairs (PK-vs-PK, a known FK edge,
   date/country columns, ID columns, sibling FKs) filtered out before any
   LLM sees them.
3. **Phase 2 -- triage** (`triage.py`): one LLM call per candidate,
   screening for *referential* ambiguity specifically -- would a realistic
   question plausibly map to either column? -- not mere relatedness (two
   different lab tests are related, not ambiguous).
4. **Phase 3 -- deep investigation** (`deep_investigation.py`): pure SQL,
   no LLM. Join-path discovery (direct / bridge / unconnected), value
   domains, value overlap, and for connected pairs: **coverage** (what
   fraction of parent entities have rows in the child table), **fan-out**
   (rows per key value, labeled with which table fans out), and
   **agreement** (how often the two columns' values actually match on the
   same entity after joining).
5. **Phase 4 -- verdict** (`verdict.py`): one LLM call per investigated
   pair, distilling the evidence bundle into a note: verdict, each
   column's purpose, when to use which, a caution for partial-coverage
   traps, and pitfalls. This is the *only* thing that survives into the
   compact output file -- raw statistics never leave the offline pipeline.

Run per database:

```bash
python -m dextersql.schema_exploration.offline_pipeline \
    --db-id thrombosis_prediction \
    --db-path /path/to/thrombosis_prediction.sqlite \
    --output-dir workspace/schema_exploration/notes \
    --model openai/gpt-oss-120b --base-url http://HOST:8000/v1 --api-key dummy
```

Writes `ambiguity_notes_<db_id>.json` (compact -- what the online gate
reads) and `column_ambiguity_deep_<db_id>.json` (notes + full evidence, for
a human to audit).

**Online** (`gate.py` + `inject.py`, wired into `pipeline.py` right after
`schema_linking`, only when enabled): for each question, candidate notes
are the ones where at least one column is in the question's schema-linked
set (both-columns-required would suppress exactly the case where the
linker silently kept one side of an ambiguous pair). A relevance-gate LLM
call then asks which candidates are actually essential for *this*
question, keeping none by default on any parse failure -- direct,
ungated injection tends to mislead about as often as it helps, since most
questions touching a flagged column don't depend on the specific
distinction the note describes.

## Enabling it

```toml
[schema_exploration]
enabled = true
notes_dir = "/path/to/workspace/schema_exploration/notes"   # ambiguity_notes_<db_id>.json per DB
max_notes_per_question = 0                                   # 0 = unlimited

[schema_exploration.gate_llm]
model = "openai/gpt-oss-120b"
api_type = "openai"
base_url = "http://HOST:8000/v1"
api_key = "dummy"
```

Omit the section, or set `enabled = false`, to run exactly as before.

## Deliberate simplifications vs. the reference this was built from

This module was built from an existing offline ambiguity-mining pipeline
(`column_ambiguity.py` / `column_ambiguity_deep.py`, elsewhere in this
workspace) as a template, not a spec to replicate exactly. Differences,
each made to cut scope/dependencies rather than for a correctness reason:

- **No profile-embedding candidate generation.** The reference pipeline
  had a third Phase-1 generator using OpenAI embeddings over column
  profiles, to catch differently-named but semantically related columns.
  Dropped here to keep this pipeline dependency-free; exact + fuzzy name
  matching already cover the flagship cases (same or near-same column
  name in two tables).
- **Single-prompt triage, not a 3-variant majority vote.** The reference
  asked three prompt variants (no schema / filtered schema / full schema)
  per candidate and required a 2-of-3 or 3-of-3 consensus for HARD/SOFT.
  This version asks once, with full schema context, and maps
  yes/no/maybe directly to HARD/SKIP/SOFT -- a 3x cheaper triage pass, at
  the cost of the extra precision a consensus vote buys.
- **No cluster-type notes.** The reference also supported grouping several
  related columns into one "family" note (e.g. a dozen position-code
  columns). This version only produces pairwise notes.

If triage precision or candidate recall becomes a bottleneck, these are
the natural next additions -- the deterministic core (Phases 0, 1, 3) and
the online gate are unchanged in spirit from the reference.
