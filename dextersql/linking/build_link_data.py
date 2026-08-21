#!/usr/bin/env python3
"""
Build the schema-linker's per-database data, from scratch, on canonical BIRD.

For each database this produces, into <out-root>/<db_id>/:
    <db_id>.sqlite                     symlink to the canonical BIRD database
    database_description/              symlink to the canonical BIRD descriptions
    <db_id>.long_profiles.jsonl        deterministic  (from the SQLite values)
    <db_id>.full_profiles.jsonl        deterministic  (long profiles + descriptions)
    <db_id>.short_profile_prompts.jsonl deterministic (from full profiles)
    <db_id>.short_profiles.jsonl       LLM           (one call per column)
    <db_id>.faiss  + .faiss_meta.json  OpenAI embeddings over long profiles
    <db_id>.faiss_full + meta          OpenAI embeddings over full profiles
    <db_id>.lsh.pkl + .lsh_index.json  local MinHash-LSH over the SQLite values

The result directory is what the linker reads via DEXTERSQL_LINK_DATA_ROOT, so the
package is standalone: SQLite + descriptions are referenced from the canonical BIRD
tree, everything else is regenerated here — nothing is reused from prior projects.

Usage
-----
    export OPENAI_API_KEY=...            # FAISS index embeddings
    export DEXTERSQL_LINK_DATA_ROOT=<out-root>
    python -m dextersql.linking.build_link_data \
        --bird-db-root /path/to/dev_databases \
        --out-root     $DEXTERSQL_LINK_DATA_ROOT \
        --backend vllm_server            # for the short-profile LLM step
        [--db debit_card_specializing]   # one DB (default: all 11)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # build_indexes, llm_backends_local
sys.path.insert(0, str(_HERE / "profiling"))

BIRD_DEV_DBS = [
    "california_schools", "card_games", "codebase_community",
    "debit_card_specializing", "european_football_2", "financial",
    "formula_1", "student_club", "superhero", "thrombosis_prediction",
    "toxicology",
]


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        return
    if not src.exists():
        raise FileNotFoundError(f"missing canonical source: {src}")
    dst.symlink_to(src)


def build_one(db_id: str, bird_db_root: Path, out_root: Path, backend) -> None:
    from profiles_sqlite_local import build_and_write_long_profiles
    from profiles_full_local import build_and_write_full_profiles
    from short_profiles_local import (build_short_profile_prompts,
                                      generate_short_profiles)
    from build_indexes import build_faiss_index, build_lsh_index
    from build_indexes_abc import build_faiss_index_full

    db_dir = out_root / db_id
    db_dir.mkdir(parents=True, exist_ok=True)
    flag = db_dir / "link_data_success_flag"
    if flag.exists():
        print(f"[{db_id}] already built (flag present), skipping", flush=True)
        return

    t0 = time.time()
    # canonical SQLite + descriptions, referenced not copied
    _symlink(bird_db_root / db_id / f"{db_id}.sqlite", db_dir / f"{db_id}.sqlite")
    _symlink(bird_db_root / db_id / "database_description", db_dir / "database_description")

    # explicit output paths (avoid Path.with_suffix multi-dot ambiguity)
    long_p = db_dir / f"{db_id}.long_profiles.jsonl"
    full_p = db_dir / f"{db_id}.full_profiles.jsonl"
    prompts_p = db_dir / f"{db_id}.short_profile_prompts.jsonl"
    short_p = db_dir / f"{db_id}.short_profiles.jsonl"

    print(f"[{db_id}] 1/5 long profiles (deterministic)", flush=True)
    build_and_write_long_profiles(db_dir / f"{db_id}.sqlite",
                                  output_jsonl=long_p, db_id=db_id)

    print(f"[{db_id}] 2/5 full profiles (deterministic)", flush=True)
    build_and_write_full_profiles(long_p, db_dir / "database_description",
                                  output_jsonl=full_p)

    print(f"[{db_id}] 3/5 short-profile prompts (deterministic)", flush=True)
    build_short_profile_prompts(full_p, output_jsonl=prompts_p)

    print(f"[{db_id}] 4/5 short profiles (LLM)", flush=True)
    generate_short_profiles(prompts_p, backend, output_jsonl=short_p)

    print(f"[{db_id}] 5/5 indices: FAISS (long), FAISS_full, LSH", flush=True)
    build_faiss_index(db_id, db_dir)          # OpenAI embeddings over long profiles
    build_faiss_index_full(db_id, db_dir)     # OpenAI embeddings over full profiles
    build_lsh_index(db_id, db_dir)            # local, from the SQLite values

    flag.touch()
    print(f"[{db_id}] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build linker data from canonical BIRD.")
    ap.add_argument("--bird-db-root", required=True,
                    help="dir containing <db_id>/<db_id>.sqlite + database_description/")
    ap.add_argument("--out-root", default=os.environ.get("DEXTERSQL_LINK_DATA_ROOT", ""),
                    help="output dev_databases root (also read from DEXTERSQL_LINK_DATA_ROOT)")
    ap.add_argument("--db", default=None, help="one DB (default: all 11 BIRD dev DBs)")
    ap.add_argument("--backend", default="vllm_server",
                    help="LLM backend for the short-profile step")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    if not a.out_root:
        sys.exit("--out-root (or DEXTERSQL_LINK_DATA_ROOT) is required")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY must be set (FAISS index embeddings use OpenAI)")

    bird_db_root = Path(a.bird_db_root)
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dbs = [a.db] if a.db else BIRD_DEV_DBS

    from llm_backends_local import make_backend
    backend = make_backend(a.backend, model_id=a.model, cache=True)

    print(f"[build_link_data] {len(dbs)} database(s) -> {out_root}", flush=True)
    t0 = time.time()
    for i, db in enumerate(dbs, 1):
        print(f"\n===== [{i}/{len(dbs)}] {db} =====", flush=True)
        build_one(db, bird_db_root, out_root, backend)
    print(f"\n[build_link_data] ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
