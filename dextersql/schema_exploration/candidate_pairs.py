"""
Phase 1 -- Candidate pair generation.

Cheaply enumerates column pairs that MIGHT be ambiguous, then prunes the
obvious non-cases -- all without an LLM.

Two generators (simplified from the reference this module is based on,
which added a third, profile-embedding generator requiring an external
embedding API; dropped here to keep this pipeline dependency-free -- exact
and fuzzy name matching already cover the flagship cases, like
Patient.Diagnosis vs Examination.Diagnosis):

  1A. Exact name match   -- same (case-insensitive) column name in two
                             different tables.
  1B. Fuzzy token match  -- tokenized names overlap >= 50% of the smaller
                             token set (e.g. "aCL IgA" vs "IGA").

Then passes_noise_filter (db_stats.py) drops PK-vs-PK, known-FK, date/
country, ID-column, and sibling-FK pairs -- structurally explained
"look-alikes" that would otherwise flood the triage LLM.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Tuple

from .db_stats import build_noise_filter_sets, passes_noise_filter, tokenize_column_name


def generate_candidate_pairs(schema: Dict[str, List[Dict[str, Any]]], fk_map: Dict[str, str]) -> List[Dict[str, Any]]:
    all_columns = [(table, c["name"], c["type"]) for table, cols in schema.items() for c in cols]

    seen: set = set()
    candidates: List[Dict[str, Any]] = []

    def add(ta, ca, tb, cb, method, score=1.0):
        key_a, key_b = f"{ta}.{ca}", f"{tb}.{cb}"
        pair_key = tuple(sorted([key_a, key_b]))
        if pair_key not in seen:
            seen.add(pair_key)
            candidates.append({"col_a": key_a, "col_b": key_b, "method": method, "score": score})

    # 1A -- exact name match.
    name_to_locations: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for table, col_name, _ in all_columns:
        name_to_locations[col_name.lower()].append((table, col_name))
    for locations in name_to_locations.values():
        if len(locations) >= 2:
            for (ta, ca), (tb, cb) in combinations(locations, 2):
                add(ta, ca, tb, cb, "exact")

    # 1B -- fuzzy token match.
    col_tokens = {f"{table}.{col_name}": tokenize_column_name(col_name) for table, col_name, _ in all_columns}
    for i, (ta, ca, _) in enumerate(all_columns):
        for j, (tb, cb, _) in enumerate(all_columns):
            if j <= i or ta == tb:
                continue
            tokens_a, tokens_b = col_tokens[f"{ta}.{ca}"], col_tokens[f"{tb}.{cb}"]
            if not tokens_a or not tokens_b:
                continue
            overlap = len(tokens_a & tokens_b)
            min_len = min(len(tokens_a), len(tokens_b))
            if min_len > 0 and overlap / min_len >= 0.5:
                add(ta, ca, tb, cb, "fuzzy", round(overlap / min_len, 2))

    pk_set, skip_col_set, id_col_set = build_noise_filter_sets(schema)
    return [
        c for c in candidates
        if passes_noise_filter(c["col_a"], c["col_b"], pk_set, skip_col_set, id_col_set, fk_map)
    ]
