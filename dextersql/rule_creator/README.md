# Rule Creator

The offline pipeline that mines recurring LLM SQL-generation errors from
**BIRD training data only** and converts each dominant recurring error into
a canonical correction rule: `{rule_name, gist, bad_pattern,
correct_pattern, fix}`.

Rules are database-agnostic and created **once, offline** -- no fine-tuning,
no access to dev/test data at any point.

## Data isolation

`RuleCreatorConfig` (`config.py`) has exactly two data-path fields:
`train_json_path` and `train_databases_root`. There is no `split` concept
anywhere in this package's config, no fallback to a shared dataset root, and
every module resolves database paths directly under
`train_databases_root/<db_id>/<db_id>.sqlite`. It is structurally impossible
to point this pipeline at dev or test data without editing the config file
to name a different training-data path explicitly.

## The four steps

### Step 1 -- Error mining (`sampling.py`, `error_mining.py`)

1. **Sample** a subset of training questions, stratified by difficulty so
   the sample covers a range of SQL complexity. BIRD's official `train.json`
   carries no difficulty label (unlike `dev.json`), so `difficulty.py`
   estimates one from the gold SQL's own structure -- joins, nesting,
   aggregation, grouping, ordering, set operations -- the same signal the
   paper's methodology specifies as the fallback.
2. For each sampled question, **generate multiple candidate SQLs** from the
   underlying LLM (schema + question + evidence in, several sampled SQL
   completions out).
3. **Execute** every candidate against the real training database and
   compare to gold's execution result. Discard candidates that match gold.
4. For every remaining incorrect candidate, ask the LLM to **explain** why
   it's wrong compared to gold -- specific and precise enough that the
   explanation could become a rule. If one candidate has multiple
   independent problems, they come back as separate explanations rather
   than one blended one.

Output: `02_fail_candidate.jsonl`, one JSON object per
`(candidate, explanation)` pair.

### Step 2 -- Database-agnostic error isolation (`db_agnostic_filter.py`)

Not every mismatch generalizes. A wrong column pick caused by two
confusingly-named columns *in this one schema*, or a domain value mapping
specific to *this one database*, won't recur anywhere else. An LLM
classifies each Step 1 explanation as a reusable SQL-formulation issue
("database-agnostic") or a schema/value artifact tied to one database
("database-specific"), and only the former survive.

Output: `03_fail_final.jsonl`.

### Step 3 -- Clustering errors (`clustering.py`)

Hierarchical LLM clustering, in the same "batch, then merge" shape at every
level:

1. Within each training database, split its surviving explanations into
   batches and ask the LLM to group same-error explanations within each
   batch.
2. Merge similar batch-level groups into one set of groups for that
   database.
3. Merge across databases, repeating the same comparison-and-merge
   operation, until groups describing the same underlying error end up
   together regardless of which database they were mined from.
4. Discard groups with too few supporting explanations
   (`min_group_support`) as non-dominant / noise.

Both merge levels (step 2 and step 3 above) call the *same*
`_merge_groups_chunked` function -- it's the same operation either way:
compare group summaries, merge the ones describing the same error. Merge
calls are chunked (`cross_db_merge_chunk_size`) because handing the model
too many groups in one call risks silently eating the whole token budget
on the model's own `<reasoning>` before it ever reaches `<result>` --
chunking plus a generous `max_tokens` on the clustering LLM avoids that
failure mode.

Output: `04_error_groups.json` -- the dominant `ErrorGroup` list, each with
its full member rows retained (for Step 4, and for a human skimming the
output to sanity-check a group before trusting a rule synthesized from it).

### Step 4 -- Rule synthesis (`rule_synthesis.py`)

For each dominant group, the LLM synthesizes one canonical rule:
`{rule_name, gist, bad_pattern, correct_pattern, fix}` -- told explicitly
that the rule must describe the error and fix *generically*, since it will
later be applied to brand-new questions against databases it has never seen
a mined example from.

Output: `05_created_rules.json` -- a JSON list of rule dicts, each carrying
a `note` field with provenance (support count, databases covered, example
training question ids, synthesis date).

## Usage

```bash
cp config/rule_creator.toml.template config/rule_creator.toml
# fill in train_json_path, train_databases_root, output_dir, and every
# <MODEL_ID> / <LLM_BASE_URL> / <API_KEY> placeholder

python -m dextersql.rule_creator.pipeline --config config/rule_creator.toml

# resume from a specific step (reuses earlier steps' saved output files):
python -m dextersql.rule_creator.pipeline --config config/rule_creator.toml --start-step 3

# run only steps 1-2, stop before clustering:
python -m dextersql.rule_creator.pipeline --config config/rule_creator.toml --stop-step 2
```

Every step's output is a plain JSON/JSONL file under `output_dir`
(`01_sampled_questions.json` ... `05_created_rules.json`), so intermediate
results can be inspected, and a run can be resumed after a crash or an HPC
job time limit without repeating the expensive LLM-heavy steps.

## Module map

| File | Step | Role |
|---|---|---|
| `config.py` | -- | `RuleCreatorConfig`; the only place train data paths are named |
| `difficulty.py` | 1a | gold-SQL-structure difficulty estimation |
| `sampling.py` | 1a | stratified sampling of training questions |
| `error_mining.py` | 1 | candidate generation, execution comparison, failure explanation |
| `db_agnostic_filter.py` | 2 | database-agnostic vs database-specific classification |
| `clustering.py` | 3 | hierarchical batch-then-merge clustering |
| `rule_synthesis.py` | 4 | dominant group -> canonical rule |
| `prompts.py` | 1-4 | every LLM prompt template |
| `parsing.py` | 1-4 | `<result>` parsing/validation for every LLM call |
| `pipeline.py` | -- | orchestrator + CLI (`python -m dextersql.rule_creator.pipeline`) |
