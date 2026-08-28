"""
Step 3 (Clustering errors).

Hierarchical LLM clustering of database-agnostic failure explanations:
  1. Within each training database, split its explanations into batches and
     ask an LLM to group same-error explanations within each batch.
  2. Compare and merge similar batch-level groups to form one set of groups
     for that database.
  3. Repeat the same comparison-and-merging process across databases,
     merging groups that describe the same underlying error.
  4. Discard groups with too few supporting explanations.

The within-database merge (step 2) and the cross-database merge (step 3)
are the exact same operation -- "compare group summaries, merge same-error
groups" -- so both are implemented by the one _merge_groups_chunked
function. Merge calls are chunked (config.clustering.cross_db_merge_chunk_size)
because a single call over too many groups at once risks overwhelming the
model's token budget for its own reasoning before it reaches <result>.

Input: config.fail_final_path (from db_agnostic_filter.run_db_agnostic_filter).
Output: config.error_groups_path -- the dominant ErrorGroup list, each with
its full member rows retained for Step 4 and for human review.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dextersql.core.llm import make_llm
from dextersql.core.llm_extractor import LLMExtractor
from dextersql.core.logger import logger
from dextersql.rule_creator.config import RuleCreatorConfig
from dextersql.rule_creator.parsing import parse_batch_clustering, parse_group_merge
from dextersql.rule_creator.prompts import format_batch_clustering_prompt, format_group_merge_prompt

Group = Dict[str, Any]  # {"label": str, "support": int, "example_explanations": [str], "member_rows": [dict]}


def _load_fail_final(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_from_rows(label: str, rows: List[Dict[str, Any]]) -> Group:
    return {
        "label": label,
        "support": len(rows),
        "example_explanations": [r["explanation"] for r in rows[:2]],
        "member_rows": rows,
    }


def _batch_cluster(rows: List[Dict[str, Any]], llm) -> List[Group]:
    """One batch (<= config.clustering.batch_size rows) -> list of Groups."""
    extractor = LLMExtractor(max_retry=3)
    prompt = format_batch_clustering_prompt([r["explanation"] for r in rows])
    results, _ = extractor.extract_with_retry(
        llm=llm,
        messages=[{"role": "user", "content": prompt}],
        rule_parser=parse_batch_clustering,
        parser_kwargs={"n": len(rows)},
        n=1,
    )
    if not results:
        logger.warning(f"[clustering] batch clustering failed on a batch of {len(rows)} -- treating each as its own singleton group")
        return [_group_from_rows(r["explanation"][:80], [r]) for r in rows]

    clusters = results[0]
    return [_group_from_rows(c["label"], [rows[i - 1] for i in c["member_ids"]]) for c in clusters]


def _merge_once(groups: List[Group], llm) -> List[Group]:
    """One merge call over <= chunk_size groups -> merged Groups."""
    extractor = LLMExtractor(max_retry=3)
    prompt = format_group_merge_prompt(groups)
    results, _ = extractor.extract_with_retry(
        llm=llm,
        messages=[{"role": "user", "content": prompt}],
        rule_parser=parse_group_merge,
        parser_kwargs={"n": len(groups)},
        n=1,
    )
    if not results:
        logger.warning(f"[clustering] merge failed on {len(groups)} groups -- leaving them unmerged")
        return groups

    merges = results[0]
    merged: List[Group] = []
    for m in merges:
        source_rows: List[Dict[str, Any]] = []
        for gid in m["source_group_ids"]:
            source_rows.extend(groups[gid - 1]["member_rows"])
        merged.append(_group_from_rows(m["merged_label"], source_rows))
    return merged


def _merge_groups_chunked(groups: List[Group], llm, chunk_size: int, max_passes: int = 6) -> List[Group]:
    """
    Reduce `groups` by repeatedly merging in chunks until a pass over all
    current groups doesn't reduce the count any further (nothing left to
    merge) or max_passes is hit (safety valve against oscillation).
    """
    current = groups
    for _ in range(max_passes):
        if len(current) <= 1:
            break
        chunks = [current[i:i + chunk_size] for i in range(0, len(current), chunk_size)]
        next_round: List[Group] = []
        for chunk in chunks:
            next_round.extend(_merge_once(chunk, llm) if len(chunk) > 1 else chunk)
        if len(next_round) == len(current):
            current = next_round
            break
        current = next_round
    return current


def run_clustering(config: RuleCreatorConfig, fail_final: List[Dict[str, Any]] = None) -> List[Group]:
    if fail_final is None:
        fail_final = _load_fail_final(config.fail_final_path)
    print(f"[clustering] clustering {len(fail_final)} database-agnostic explanations")

    llm = make_llm(config.clustering.llm)
    cfg = config.clustering

    by_db: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in fail_final:
        by_db[row["db_id"]].append(row)

    db_level_groups: List[Group] = []
    for db_id, rows in by_db.items():
        batches = [rows[i:i + cfg.batch_size] for i in range(0, len(rows), cfg.batch_size)]
        batch_groups: List[Group] = []
        for batch in batches:
            batch_groups.extend(_batch_cluster(batch, llm))
        print(f"[clustering]   {db_id}: {len(rows)} explanations -> {len(batches)} batches -> {len(batch_groups)} batch-level groups")

        merged = _merge_groups_chunked(batch_groups, llm, cfg.cross_db_merge_chunk_size)
        print(f"[clustering]   {db_id}: merged to {len(merged)} db-level groups")
        db_level_groups.extend(merged)

    print(f"[clustering] {len(db_level_groups)} total db-level groups across {len(by_db)} databases; merging across databases")
    final_groups = _merge_groups_chunked(db_level_groups, llm, cfg.cross_db_merge_chunk_size)
    print(f"[clustering] {len(final_groups)} groups after cross-database merge")

    dominant = [g for g in final_groups if g["support"] >= cfg.min_group_support]
    dominant.sort(key=lambda g: -g["support"])
    print(f"[clustering] {len(dominant)}/{len(final_groups)} groups kept as dominant (support >= {cfg.min_group_support})")

    # Enrich with group_id/db_ids up front so the in-memory list returned to
    # an in-process caller (pipeline.py chaining steps together) has the
    # exact same shape as what gets written to disk and re-loaded by
    # _load_error_groups when resuming from --start-step 4 -- Step 4 depends
    # on both fields being present either way.
    enriched = [
        {
            "group_id": i,
            "label": g["label"],
            "support": g["support"],
            "db_ids": sorted(set(r["db_id"] for r in g["member_rows"])),
            "member_rows": g["member_rows"],
        }
        for i, g in enumerate(dominant)
    ]

    out_path = Path(config.error_groups_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "n_explanations_in": len(fail_final),
                "n_databases": len(by_db),
                "n_final_groups_before_support_filter": len(final_groups),
                "min_group_support": cfg.min_group_support,
            },
            "groups": enriched,
        }, f, indent=2)
    print(f"[clustering] -> {out_path}")

    return enriched
