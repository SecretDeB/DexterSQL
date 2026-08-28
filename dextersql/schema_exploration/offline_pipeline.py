"""
Offline, once-per-database pipeline: runs Phases 0-4 against one SQLite
database and writes two files:

  column_ambiguity_deep_<db_id>.json  -- notes + full evidence bundle,
                                          for a human to audit.
  ambiguity_notes_<db_id>.json        -- compact notes only (no
                                          statistics). This is the file
                                          gate.py reads at inference time.

Independent of dextersql.core.config.AppConfig on purpose: this tool runs
offline, once, against whichever database you point it at (dev or train --
schema/column semantics don't depend on the split), well before any
inference-time pipeline run, so it takes its own LLM settings on the
command line rather than sharing config/bird.toml's stage sections.

Usage:
    python -m dextersql.schema_exploration.offline_pipeline \\
        --db-id thrombosis_prediction \\
        --db-path /path/to/thrombosis_prediction.sqlite \\
        --output-dir workspace/schema_exploration/notes \\
        --model openai/gpt-oss-120b --base-url http://HOST:8000/v1 --api-key dummy
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict

from dextersql.core.config.config import LLMConfig
from dextersql.core.llm import make_llm

from .candidate_pairs import generate_candidate_pairs
from .db_stats import load_create_statements, load_schema
from .deep_investigation import investigate_pair
from .fk_discovery import discover_foreign_keys
from .triage import triage_pairs
from .verdict import synthesize_verdict


def load_dev_docs(db_dir: Path) -> Dict[str, Dict[str, str]]:
    """BIRD's standard per-database database_description/<table>.csv files,
    if present. Not required -- Phase 4 works from statistics alone -- but
    official column descriptions help the verdict name the distinction
    correctly rather than just describe it."""
    desc_dir = db_dir / "database_description"
    if not desc_dir.exists():
        return {}
    docs: Dict[str, Dict[str, str]] = {}
    for csv_path in sorted(desc_dir.glob("*.csv")):
        table_name = csv_path.stem
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    col_name = (row.get("original_column_name") or "").strip()
                    if not col_name:
                        continue
                    docs[f"{table_name}.{col_name}"] = {
                        "column_description": (row.get("column_description") or "").strip(),
                        "value_description": (row.get("value_description") or "").strip(),
                    }
        except Exception:
            continue
    return docs


def run_for_database(
    db_id: str,
    db_path: str,
    output_dir: str,
    llm,
    include_soft: bool = False,
    use_llm_implicit_fk: bool = True,
) -> Dict[str, str]:
    conn = sqlite3.connect(db_path)
    try:
        db_dir = Path(db_path).parent
        dev_docs = load_dev_docs(db_dir)

        print(f"[{db_id}] Phase 0 -- FK discovery")
        schema = load_schema(conn)
        create_statements = load_create_statements(conn)
        fk_map = discover_foreign_keys(conn, schema, create_statements, llm=llm if use_llm_implicit_fk else None)
        print(f"[{db_id}]   fk_map: {len(fk_map)} edges")

        print(f"[{db_id}] Phase 1 -- candidate pair generation")
        candidates = generate_candidate_pairs(schema, fk_map)
        print(f"[{db_id}]   {len(candidates)} candidate pairs")

        print(f"[{db_id}] Phase 2 -- LLM triage")
        flagged = triage_pairs(candidates, conn, schema, dev_docs, create_statements, fk_map, llm)
        hard = [p for p in flagged if p["flag_strength"] == "hard"]
        soft = [p for p in flagged if p["flag_strength"] == "soft"]
        print(f"[{db_id}]   {len(hard)} hard, {len(soft)} soft")
        to_investigate = flagged if include_soft else hard

        print(f"[{db_id}] Phase 3 -- deep investigation ({len(to_investigate)} pairs)")
        investigations = [investigate_pair(conn, pair, fk_map, schema) for pair in to_investigate]

        print(f"[{db_id}] Phase 4 -- verdict synthesis")
        notes = []
        full_records = []
        for inv in investigations:
            note = synthesize_verdict(inv, llm, dev_docs)
            full_records.append({"investigation": inv, "note": note})
            if note is not None:
                notes.append(note)
        print(f"[{db_id}]   {len(notes)}/{len(investigations)} pairs synthesized into notes")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        compact_path = out_dir / f"ambiguity_notes_{db_id}.json"
        with open(compact_path, "w") as f:
            json.dump({"db_id": db_id, "notes": notes}, f, indent=2)

        full_path = out_dir / f"column_ambiguity_deep_{db_id}.json"
        with open(full_path, "w") as f:
            json.dump(
                {
                    "db_id": db_id,
                    "fk_map": fk_map,
                    "candidate_pairs": candidates,
                    "flagged_pairs": flagged,
                    "records": full_records,
                },
                f,
                indent=2,
            )

        return {"compact": str(compact_path), "full": str(full_path)}
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-id", required=True)
    ap.add_argument("--db-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--api-type", default="openai")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--include-soft", action="store_true", help="also deep-investigate SOFT-flagged pairs (default: HARD only)")
    ap.add_argument("--no-llm-implicit-fk", action="store_true", help="skip Phase 0c (LLM-proposed implicit FKs)")
    args = ap.parse_args()

    llm = make_llm(
        LLMConfig(
            model=args.model, base_url=args.base_url, api_key=args.api_key,
            api_type=args.api_type, max_tokens=args.max_tokens, temperature=0.0,
        )
    )

    paths = run_for_database(
        db_id=args.db_id, db_path=args.db_path, output_dir=args.output_dir, llm=llm,
        include_soft=args.include_soft, use_llm_implicit_fk=not args.no_llm_implicit_fk,
    )
    print(f"\n-> {paths['compact']}")
    print(f"-> {paths['full']}")


if __name__ == "__main__":
    main()
