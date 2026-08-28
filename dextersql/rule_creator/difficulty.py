"""
Step 1a (part of Error Mining): difficulty estimation.

BIRD's official train.json does not carry a "difficulty" field (unlike
dev.json, which has simple/moderate/challenging labels) -- so sampling has
to estimate it. Per the methodology: "If such labels are unavailable, this
step estimates difficulty based on the gold SQL structure, such as whether
the query contains joins, nesting, aggregation, grouping, ordering, or set
operations, so that the sampled questions cover a range of SQL complexities."

This module does exactly that via lightweight structural checks on the gold
SQL text -- no full SQL parser, just the same signal a human skimming the
query would use.
"""

from __future__ import annotations

import re
from typing import Literal

Difficulty = Literal["simple", "moderate", "challenging"]

_AGGREGATE_FUNCS = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_JOIN = re.compile(r"\bJOIN\b", re.IGNORECASE)
_GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_SET_OP = re.compile(r"\b(UNION|INTERSECT|EXCEPT)\b", re.IGNORECASE)
_HAVING = re.compile(r"\bHAVING\b", re.IGNORECASE)
_CASE = re.compile(r"\bCASE\b", re.IGNORECASE)
_SELECT = re.compile(r"\bSELECT\b", re.IGNORECASE)


def _count_nested_selects(sql: str) -> int:
    """
    Number of SELECT keywords beyond the first -- a cheap proxy for
    subquery/nesting depth without a real SQL parser. A CTE or a
    WHERE-clause subquery both show up as an extra SELECT.
    """
    return max(0, len(_SELECT.findall(sql)) - 1)


def estimate_difficulty(gold_sql: str) -> Difficulty:
    """
    Structural-complexity score over: joins, nesting, aggregation, grouping,
    ordering, set operations. Each present feature contributes to the score,
    and the tier boundaries simply split that score into three bands. They
    don't need to be exact -- only consistent enough to stratify sampling
    across a real range of SQL complexity. Used only when the training data
    carries no difficulty label of its own.
    """
    sql = gold_sql or ""

    score = 0
    score += 1 if _JOIN.search(sql) else 0
    score += 1 if _AGGREGATE_FUNCS.search(sql) else 0
    score += 1 if _GROUP_BY.search(sql) else 0
    score += 1 if _ORDER_BY.search(sql) else 0
    score += 2 if _SET_OP.search(sql) else 0
    score += 2 if _HAVING.search(sql) else 0
    score += 1 if _CASE.search(sql) else 0

    nested = _count_nested_selects(sql)
    score += min(nested, 2) * 2  # each nested SELECT is a strong complexity signal, cap contribution

    if score <= 1:
        return "simple"
    if score <= 4:
        return "moderate"
    return "challenging"
