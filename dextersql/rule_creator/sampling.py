"""
Step 1a (part of Error Mining): sample a subset of training questions,
stratified by difficulty so the sampled set "covers a range of SQL
complexities" as the methodology requires.

Uses train.json's own "difficulty" field when present; BIRD's official
release doesn't carry one for train, so in practice this always falls back
to difficulty.estimate_difficulty(gold_sql).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from dextersql.rule_creator.config import RuleCreatorConfig
from dextersql.rule_creator.difficulty import Difficulty, estimate_difficulty


def _load_train_rows(train_json_path: str) -> List[Dict[str, Any]]:
    with open(train_json_path) as f:
        rows = json.load(f)
    return rows


def _stratified_pick(
    pool_by_tier: Dict[Difficulty, List[int]],
    tier_targets: Dict[Difficulty, int],
    rng: random.Random,
) -> List[int]:
    selected: List[int] = []
    for tier, target in tier_targets.items():
        pool = list(pool_by_tier.get(tier, []))
        rng.shuffle(pool)
        selected.extend(pool[:target])
    return selected


def sample_training_questions(config: RuleCreatorConfig) -> List[Dict[str, Any]]:
    """
    Returns a list of sampled question dicts:
      {question_id (index into train.json), db_id, question, evidence,
       gold_sql, difficulty}

    Writes the same list to config.sampled_path and returns it.
    """
    rows = _load_train_rows(config.train_json_path)
    print(f"[sampling] loaded {len(rows)} rows from {config.train_json_path}")

    pool_by_tier: Dict[Difficulty, List[int]] = {"simple": [], "moderate": [], "challenging": []}
    difficulty_of: Dict[int, Difficulty] = {}
    for idx, row in enumerate(rows):
        gold_sql = row.get("SQL") or row.get("query") or ""
        difficulty = row.get("difficulty") or estimate_difficulty(gold_sql)
        if difficulty not in pool_by_tier:
            difficulty = estimate_difficulty(gold_sql)
        pool_by_tier[difficulty].append(idx)
        difficulty_of[idx] = difficulty

    for tier, pool in pool_by_tier.items():
        print(f"[sampling]   {tier:12s} pool={len(pool)}")

    cfg = config.sampling
    tier_targets: Dict[Difficulty, int] = {
        "simple": round(cfg.target_size * cfg.simple_fraction),
        "moderate": round(cfg.target_size * cfg.moderate_fraction),
        "challenging": round(cfg.target_size * cfg.challenging_fraction),
    }
    drift = cfg.target_size - sum(tier_targets.values())
    if drift:
        tier_targets["moderate"] += drift

    rng = random.Random(cfg.seed)
    selected_indices = _stratified_pick(pool_by_tier, tier_targets, rng)
    selected_indices.sort()

    samples: List[Dict[str, Any]] = []
    for idx in selected_indices:
        row = rows[idx]
        samples.append({
            "question_id": idx,
            "db_id": row.get("db_id"),
            "question": row.get("question"),
            "evidence": row.get("evidence", ""),
            "gold_sql": row.get("SQL") or row.get("query"),
            "difficulty": difficulty_of[idx],
        })

    actual_counts = {t: sum(1 for s in samples if s["difficulty"] == t) for t in tier_targets}
    print(f"[sampling] sampled {len(samples)}/{cfg.target_size} target: {actual_counts}")

    out_path = Path(config.sampled_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "target_size": cfg.target_size,
                "seed": cfg.seed,
                "tier_targets": tier_targets,
                "actual_counts": actual_counts,
                "source": config.train_json_path,
            },
            "samples": samples,
        }, f, indent=2)
    print(f"[sampling] -> {out_path}")

    return samples
