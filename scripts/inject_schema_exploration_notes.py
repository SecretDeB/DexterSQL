#!/usr/bin/env python3
"""
Optional step: inject gated column-disambiguation notes into the schema-linking
snapshot's `evidence` field, for questions whose schema-linked columns are covered
by an offline-mined ambiguity_notes_<db_id>.json (see
dextersql/schema_exploration/README.md for how those are produced).

Run this AFTER building the schema_linking snapshot (step 5 in the README's "Run
the whole pipeline" sequence) and BEFORE `scripts/run_pipeline.py --stages
sql_generation ...`, so sql_generation reads the augmented evidence.

Requires [schema_exploration] enabled = true in the config file, with a
[schema_exploration.gate_llm] section and notes_dir pointing at a directory of
ambiguity_notes_<db_id>.json files. If enabled = false (the default), this script
is a no-op by design -- the snapshot is left untouched.

Usage:
    python scripts/inject_schema_exploration_notes.py --config config/bird.toml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="pipeline TOML config")
    args = ap.parse_args()

    os.environ["CONFIG_PATH"] = os.path.abspath(args.config)
    from dextersql.core.config import get_config
    from dextersql.schema_exploration.inject import inject_schema_exploration_notes

    app_config = get_config()
    if not app_config.schema_exploration_config.enabled:
        print("[schema_exploration] enabled=false in this config -- nothing to do.")
        return

    stats = inject_schema_exploration_notes(app_config)
    print(
        f"[schema_exploration] {stats['n_items']} items | "
        f"{stats['n_candidates']} schema-linked candidate notes | "
        f"{stats['n_triggered']} questions received a note after gating"
    )
    already = stats.get("n_already_injected", 0)
    if already:
        print(
            f"[schema_exploration] {already} item(s) already carried notes and were "
            f"left untouched -- re-running this step is a no-op, not a double-injection."
        )


if __name__ == "__main__":
    main()
