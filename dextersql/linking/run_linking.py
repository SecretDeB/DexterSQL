"""
run_pipeline.py
---------------
Main runner for the Schema Linking pipeline.

Runs the full Phase 3 pipeline for:
  - All 30 questions in debit_card_specializing  (default)
  - Or 3-4 questions per database for all 11 databases  (--all)

For each question:
  1. Schema Linking: FAISS + LSH → focused schema
  2. Build 5 schema + profile combinations
  3. Generate SQL via LLM for each → 5 SQL queries
  4. Extract referenced columns from each SQL
  5. Union of columns = schema links

Default backend: GPT-OSS (local HuggingFace pipeline, for HPC/GPU).
Override to OpenAI API with: --backend openai
Override to OpenRouter API with: --backend openrouter  (no GPU required)

Output:
  results/schema_links_<db_id>_<timestamp>.json

Usage:
  # Run all 30 debit_card questions (default: gptoss backend)
  python run_pipeline.py --db debit_card_specializing

  # Run 3 questions per database for all 11 databases
  python run_pipeline.py --all --questions_per_db 3

  # Use OpenAI API instead of local GPU
  python run_pipeline.py --db debit_card_specializing --backend openai --model gpt-5.2

  # Use OpenRouter gpt-oss-120b (API, no GPU — needs OPENROUTER_API_KEY)
  python run_pipeline.py --db debit_card_specializing --backend openrouter
  python run_pipeline.py --db debit_card_specializing --backend openrouter --model openai/gpt-oss-120b

  # Use a specific local model / dtype
  python run_pipeline.py --all --backend gptoss --model openai/gpt-oss-20b --dtype bfloat16

  # Dry run (no LLM call, just show schemas)
  python run_pipeline.py --db debit_card_specializing --no_llm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ABC_SQL lives one level below the project root; use root infra/data/results.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))         # build_indexes, llm_backends_local, etc.
sys.path.insert(0, str(Path(__file__).parent)) # schema_linking_abc

MINIDEV_ROOT = (Path(os.environ["DEXTERSQL_LINK_DATA_ROOT"])
                if os.environ.get("DEXTERSQL_LINK_DATA_ROOT")
                else _PROJECT_ROOT / "MINIDEV" / "dev_databases")
MINIDEV_JSON = _PROJECT_ROOT / "MINIDEV" / "mini_dev_sqlite.json"
RESULTS_DIR  = _PROJECT_ROOT / "results"


# ─────────────────────────────────────────────────────────────
# Load Questions
# ─────────────────────────────────────────────────────────────

def load_questions(db_id: Optional[str] = None, limit: Optional[int] = None,
                   questions_file: Optional[Path] = None) -> List[Dict]:
    """
    Load questions from a BIRD-format JSON file.
    Defaults to MINIDEV (mini_dev_sqlite.json, 500 q). Pass questions_file to
    load the full BIRD dev set (dev.json, 1534 q) — it uses the SAME 11 dbs and
    the same question_id numbering, so the existing indexes are reused as-is.
    If db_id is given, filter to that database only.
    If limit is given, take only the first N questions per database.
    """
    src = Path(questions_file) if questions_file else MINIDEV_JSON
    with open(src) as f:
        all_qs = json.load(f)

    if db_id:
        qs = [q for q in all_qs if q["db_id"] == db_id]
    else:
        qs = all_qs

    if limit:
        # If filtering by db, just take first N
        # If all dbs, take first N per db
        if db_id:
            qs = qs[:limit]
        else:
            by_db: Dict[str, List] = {}
            for q in qs:
                by_db.setdefault(q["db_id"], []).append(q)
            qs = []
            for db, db_qs in sorted(by_db.items()):
                qs.extend(db_qs[:limit])

    return qs


# ─────────────────────────────────────────────────────────────
# Check Indexes Exist
# ─────────────────────────────────────────────────────────────

def check_indexes(db_id: str) -> bool:
    """Return True if FAISS + LSH indexes exist for this database."""
    db_dir = MINIDEV_ROOT / db_id
    faiss_ok = (db_dir / f"{db_id}.faiss").exists()
    lsh_ok   = (db_dir / f"{db_id}.lsh.pkl").exists()
    if not faiss_ok or not lsh_ok:
        print(f"  [WARN] Indexes missing for {db_id} — run build_indexes.py first")
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Main Pipeline Runner
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    questions: List[Dict],
    backend=None,
    faiss_top_k: int = 10,
    output_path: Optional[Path] = None,
    few_shot_retriever=None,
    resume_results: Optional[List[Dict]] = None,
    resume_errors:  Optional[List[Dict]] = None,
    resume_done_ids: Optional[set] = None,
) -> List[Dict]:
    """
    Run the schema linking pipeline for a list of questions.
    few_shot_retriever: FewShotRetriever instance (built once in main, shared across all questions).
    resume_results / resume_errors: pre-loaded from a partial run (via --resume).
    resume_done_ids: set of question_ids to skip (already processed in a prior run).
    Returns list of result dicts (including any resumed results).
    """
    from schema_linker import SchemaLinker   # ABC: modified linker (5 fixes)

    # Cache one SchemaLinker per db_id
    linkers: Dict[str, SchemaLinker] = {}

    # Seed from prior partial run if resuming, otherwise start fresh
    results: List[Dict] = list(resume_results or [])
    errors:  List[Dict] = list(resume_errors  or [])
    done_ids: set        = resume_done_ids or set()

    total = len(questions)
    n_skip = sum(1 for q in questions if q.get("question_id", None) in done_ids)
    print(f"\nProcessing {total} questions"
          + (f" ({n_skip} already done, skipping)" if n_skip else "") + "...")

    for i, q in enumerate(questions):
        db_id    = q["db_id"]
        question = q["question"]
        evidence = q.get("evidence", "")
        qid      = q.get("question_id", i)

        # Skip questions already completed in a previous run
        if qid in done_ids:
            continue

        print(f"\n[{i+1}/{total}] Q#{qid} | db={db_id}")

        # Check indexes
        if not check_indexes(db_id):
            errors.append({"question_id": qid, "db_id": db_id, "error": "missing indexes"})
            continue

        # Get or create linker
        if db_id not in linkers:
            try:
                linkers[db_id] = SchemaLinker(db_id)
                print(f"  Loaded SchemaLinker for: {db_id}")
            except Exception as e:
                print(f"  [ERROR] Failed to load {db_id}: {e}")
                errors.append({"question_id": qid, "db_id": db_id, "error": str(e)})
                continue

        linker = linkers[db_id]

        try:
            result = linker.run(
                question=question,
                evidence=evidence,
                backend=backend,
                faiss_top_k=faiss_top_k,
                question_id=qid,
                few_shot_retriever=few_shot_retriever,
            )
            # Attach gold SQL for reference/evaluation later
            result["gold_sql"] = q.get("SQL", "")
            result["difficulty"] = q.get("difficulty", "")
            results.append(result)

        except Exception as e:
            print(f"  [ERROR] Q#{qid}: {e}")
            errors.append({"question_id": qid, "db_id": db_id, "error": str(e)})

        # Save incrementally every 5 questions
        if output_path and (i + 1) % 5 == 0:
            _save(results, errors, output_path)
            print(f"  [Saved] {i+1}/{total} done → {output_path.name}")

    # Final save
    if output_path:
        _save(results, errors, output_path)

    print(f"\n{'='*60}")
    print(f"Done: {len(results)} succeeded, {len(errors)} errors")
    return results


def _save(results: List[Dict], errors: List[Dict], path: Path):
    """Save results + errors to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2)


def _load_resume(path: Path):
    """
    Load existing results + errors from a previous partial run.
    Returns (results, errors, done_ids) where done_ids is the set of
    question_ids already processed (both succeeded and errored).
    """
    with open(path) as f:
        data = json.load(f)
    results  = data.get("results", [])
    errors   = data.get("errors",  [])
    done_ids = (
        {r["question_id"] for r in results if "question_id" in r} |
        {e["question_id"] for e in errors  if "question_id" in e}
    )
    return results, errors, done_ids


# ─────────────────────────────────────────────────────────────
# Summary Printer
# ─────────────────────────────────────────────────────────────

def print_summary(results: List[Dict]):
    """Print a human-readable summary of schema linking results."""
    print(f"\n{'='*70}")
    print("SCHEMA LINKING SUMMARY")
    print(f"{'='*70}")

    for r in results:
        qid = r.get("question_id", "?")
        db  = r.get("db_id", "?")
        q   = r.get("question", "")[:60]
        links = r.get("schema_links", [])
        n_links = len(links)
        link_str = ", ".join(f"{l['table']}.{l['column']}" for l in links[:5])
        if len(links) > 5:
            link_str += f" ... (+{len(links)-5} more)"

        print(f"\nQ#{qid} [{db}]")
        print(f"  {q}...")
        print(f"  Literals: {r.get('literals', [])}")
        print(f"  Schema links ({n_links}): {link_str or '(none)'}")

    print(f"\nTotal questions: {len(results)}")
    avg_links = sum(len(r.get("schema_links", [])) for r in results) / max(len(results), 1)
    print(f"Avg schema links per question: {avg_links:.1f}")


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Schema Linking Pipeline")
    parser.add_argument("--db",  type=str, default="debit_card_specializing",
                        help="Single database to process")
    parser.add_argument("--all", action="store_true",
                        help="Process all 11 databases (3-4 questions each)")
    parser.add_argument("--questions_per_db", type=int, default=4,
                        help="Questions per DB when using --all (default: 4). "
                             "Pass 0 for NO per-db limit (all questions in the file).")
    parser.add_argument("--questions_file", type=str, default=None,
                        help="BIRD-format questions JSON to load instead of MINIDEV. "
                             "Use the full dev.json (1534 q) here — same 11 dbs and "
                             "question_ids as MINIDEV, so indexes are reused as-is.")
    parser.add_argument("--top_k",  type=int, default=10,
                        help="FAISS top-k columns to retrieve (default: 10)")
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM calls (dry run: just schema linking)")
    parser.add_argument("--out",    type=str, default=None,
                        help="Output JSON path (default: results/schema_links_<db>_<ts>.json)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a partial results JSON from a previous run. "
                             "Already-processed question_ids are skipped and new "
                             "results are appended to that same file.")
    # ----------------------------------------------------------------
    # Backend selection: default = gptoss (local GPU on HPC).
    # Pass --backend openai to use the OpenAI API instead.
    # ----------------------------------------------------------------
    parser.add_argument("--backend", type=str, default="gptoss",
                        choices=["gptoss", "gptoss120b", "qwen", "openai", "openrouter", "vllm_server"],
                        help="LLM backend to use for schema linking. "
                             "gptoss/gptoss120b/qwen: local GPU (HuggingFace). "
                             "openai: OpenAI API (needs OPENAI_API_KEY). "
                             "openrouter: OpenRouter API (needs OPENROUTER_API_KEY, no GPU). "
                             "vllm_server: HTTP to a live vLLM server (CPU-only; URL from "
                             "VLLM_SERVER_URL or the handshake file).")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID override. "
                             "gptoss default: openai/gpt-oss-20b | "
                             "gptoss120b default: openai/gpt-oss-120b | "
                             "qwen default: Qwen/Qwen2.5-7B-Instruct | "
                             "openai default: gpt-5.2 | "
                             "openrouter default: openai/gpt-oss-120b")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="HuggingFace device_map (default: auto). Ignored for openai/openrouter.")
    parser.add_argument("--dtype", type=str, default=None,
                        help="Torch dtype: bfloat16 | float16 | float32 | auto "
                             "(default: bfloat16 on GPU, ignored for openai/openrouter)")
    args = parser.parse_args()

    # Load questions
    qfile = Path(args.questions_file) if args.questions_file else None
    qsrc_tag = "dev1534" if qfile else "minidev"
    if args.all:
        # questions_per_db == 0 → no per-db limit (full file)
        limit = args.questions_per_db or None
        questions = load_questions(db_id=None, limit=limit, questions_file=qfile)
        tag = f"{qsrc_tag}_all_{args.questions_per_db}per" if limit else f"{qsrc_tag}_all"
    else:
        questions = load_questions(db_id=args.db, questions_file=qfile)
        tag = f"{qsrc_tag}_{args.db}"

    print(f"Loaded {len(questions)} questions")

    # ----------------------------------------------------------------
    # Output path + optional resume
    # ----------------------------------------------------------------
    resume_results:  List[Dict] = []
    resume_errors:   List[Dict] = []
    resume_done_ids: set        = set()

    if args.resume:
        resume_path = Path(args.resume)
        resume_results, resume_errors, resume_done_ids = _load_resume(resume_path)
        out_path = resume_path   # append into the same file
        print(f"Resuming from: {resume_path}  ({len(resume_done_ids)} questions already done, skipping)")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.out) if args.out else (RESULTS_DIR / f"schema_links_{tag}_{ts}.json")
    print(f"Output → {out_path}")

    # ----------------------------------------------------------------
    # Initialise backend
    # ----------------------------------------------------------------
    backend = None
    if not args.no_llm:
        from llm_backends_local import make_backend
        print(f"Initializing backend: {args.backend!r}  model: {args.model or '(default)'} ...")
        _api_backends = {"openai", "openrouter", "vllm_server"}
        backend = make_backend(
            args.backend,
            model_id=args.model,
            # device_map / dtype only apply to local GPU backends
            **({"device_map": args.device_map, "dtype": args.dtype}
               if args.backend not in _api_backends else {}),
            cache=True,
        )
    else:
        print("Dry run mode — no LLM calls")

    # Run pipeline
    t0 = time.time()
    results = run_pipeline(
        questions,
        backend=backend,
        faiss_top_k=args.top_k,
        output_path=out_path,
        few_shot_retriever=None,
        resume_results=resume_results,
        resume_errors=resume_errors,
        resume_done_ids=resume_done_ids,
    )
    elapsed = time.time() - t0

    print_summary(results)
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
