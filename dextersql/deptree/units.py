"""
Step 3: identify semantic units on the normalized dependency tree. DETERMINISTIC.

A "unit" is a typed, span-anchored piece of question meaning:
  entity | attribute | literal | filter | aggregate | comparison | negation |
  conjunction | ordering | group | limit | clause

These are INTERPRETATION cues, not token->SQL substitutions: a unit records what
the syntax suggests plus the tokens that support it. Schema grounding (step 4)
decides the actual table/column; the IR builder (step 5) decides the SQL slot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .parser import (DepTree, Node, AGG_PHRASES, CMP_PHRASES, ORDER_PHRASES,
                    GROUP_PHRASES, NEG_WORDS)

# dependency relations that carry each interpretation (see handoff §3)
ARG_DEPS = {"nsubj", "nsubjpass", "dobj", "obj", "pobj", "obl", "attr", "dative"}
MOD_DEPS = {"compound", "amod", "nummod", "poss", "nmod"}
CLAUSE_DEPS = {"acl", "relcl", "advcl", "ccomp", "xcomp", "acl:relcl"}

SUPERLATIVE_TAGS = {"JJS", "RBS"}
_STOP_HEADS = {"be", "do", "have", "list", "show", "give", "find", "tell", "state",
               "return, provide", "please", "what", "which", "who", "there"}


@dataclass
class Unit:
    kind: str                                 # see module docstring
    text: str                                 # surface mention
    node_ids: List[int] = field(default_factory=list)
    source_tokens: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)   # kind-specific detail

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "node_ids": self.node_ids,
                "source_tokens": self.source_tokens, "payload": self.payload}


def _toks(tree: DepTree, ids: List[int]) -> List[str]:
    return [tree.nodes[i].text for i in ids if 0 <= i < len(tree.nodes)]


def _mention_span(tree: DepTree, n: Node) -> (str, List[int]):
    """Grow a schema-mention span from a head noun through compound/amod children.
    'patient ID' / 'total cholesterol' become one mention."""
    ids = [n.i]
    for c in n.children:
        ch = tree.nodes[c]
        if ch.dep in {"compound", "amod", "nmod", "poss"} and ch.pos in {"NOUN", "PROPN", "ADJ", "NUM"}:
            ids.append(c)
    ids = sorted(set(ids), key=lambda i: tree.nodes[i].start_char)
    return " ".join(tree.nodes[i].text for i in ids), ids


def extract_units(tree: DepTree) -> List[Unit]:
    if tree is None:
        return []
    units: List[Unit] = []
    units.extend(_aggregates(tree))
    units.extend(_comparisons(tree))
    units.extend(_orderings(tree))
    units.extend(_groupings(tree))
    units.extend(_negations(tree))
    units.extend(_conjunctions(tree))
    units.extend(_clauses(tree))
    units.extend(_literals(tree))
    units.extend(_mentions(tree))
    units.extend(_limits(tree))
    return units


# ── individual extractors ────────────────────────────────────────────────────
def _aggregates(tree: DepTree) -> List[Unit]:
    out = []
    for n in tree.nodes:
        agg = None
        if n.phrase_kind == "agg":
            agg = n.phrase_value
        elif n.lemma.lower() in AGG_PHRASES:
            agg = AGG_PHRASES[n.lemma.lower()]
        elif n.text.lower() in AGG_PHRASES:
            agg = AGG_PHRASES[n.text.lower()]
        if agg:
            out.append(Unit("aggregate", n.text, [n.i], _toks(tree, [n.i]),
                            {"function": agg}))
    return out


def _comparisons(tree: DepTree) -> List[Unit]:
    out = []
    for n in tree.nodes:
        op = None
        if n.phrase_kind == "cmp":
            op = n.phrase_value
        elif n.lemma.lower() in CMP_PHRASES:
            op = CMP_PHRASES[n.lemma.lower()]
        elif n.text.lower() in CMP_PHRASES:
            op = CMP_PHRASES[n.text.lower()]
        if op:
            # the compared value is usually a nummod/pobj in the local subtree
            val, vids = _nearby_number(tree, n)
            out.append(Unit("comparison", n.text, [n.i] + vids,
                            _toks(tree, [n.i] + vids),
                            {"operator": op, "value": val}))
    return out


def _nearby_number(tree: DepTree, n: Node) -> (Optional[str], List[int]):
    """Find the numeric/quoted literal a comparison refers to."""
    cand: List[int] = list(n.children)
    if n.head != n.i:
        cand.extend(tree.nodes[n.head].children)
        cand.append(n.head)
    for i in cand:
        if i == n.i or not (0 <= i < len(tree.nodes)):
            continue
        c = tree.nodes[i]
        if c.like_num or c.pos == "NUM":
            return c.text, [i]
    return None, []


def _orderings(tree: DepTree) -> List[Unit]:
    out = []
    for n in tree.nodes:
        key = n.text.lower()
        lem = n.lemma.lower()
        direction = ORDER_PHRASES.get(key) or ORDER_PHRASES.get(lem)
        if direction is None and n.tag in SUPERLATIVE_TAGS:
            direction = "DESC" if lem in {"most", "high", "large", "great", "late"} else "ASC"
        if direction:
            out.append(Unit("ordering", n.text, [n.i], _toks(tree, [n.i]),
                            {"direction": direction, "superlative": n.tag in SUPERLATIVE_TAGS}))
    return out


def _groupings(tree: DepTree) -> List[Unit]:
    out = []
    for n in tree.nodes:
        low = n.text.lower()
        if n.phrase_kind == "group" or low in GROUP_PHRASES or (
                low == "by" and n.dep in {"prep", "agent"}):
            target_ids = [c for c in n.children if tree.nodes[c].pos in {"NOUN", "PROPN"}]
            if not target_ids and n.head != n.i:
                target_ids = [n.head]
            text = " ".join(_toks(tree, target_ids)) or n.text
            out.append(Unit("group", text, [n.i] + target_ids,
                            _toks(tree, [n.i] + target_ids), {}))
    return out


def _negations(tree: DepTree) -> List[Unit]:
    out = []
    for n in tree.nodes:
        if n.dep == "neg" or n.text.lower() in NEG_WORDS:
            scope = n.head if n.head != n.i else n.i
            out.append(Unit("negation", n.text, [n.i, scope], _toks(tree, [n.i, scope]),
                            {"scope_node": scope}))
    return out


def _conjunctions(tree: DepTree) -> List[Unit]:
    out = []
    for n in tree.nodes:
        if n.dep == "cc" or n.text.lower() in {"and", "or"}:
            conn = "OR" if n.text.lower() == "or" else "AND"
            out.append(Unit("conjunction", n.text, [n.i], _toks(tree, [n.i]),
                            {"connector": conn}))
        elif n.dep == "conj":
            out.append(Unit("conjunction", n.text, [n.i], _toks(tree, [n.i]),
                            {"connector": "AND", "conj_node": n.i}))
    return out


def _clauses(tree: DepTree) -> List[Unit]:
    out = []
    for n in tree.nodes:
        if n.dep in CLAUSE_DEPS:
            out.append(Unit("clause", tree.subtree_text(n.i), [n.i],
                            _toks(tree, [n.i]),
                            {"dep": n.dep, "subquery_candidate": n.dep in {"relcl", "acl", "ccomp"}}))
    return out


_DATE_RE = re.compile(r"^(19|20)\d{2}(-\d{1,2}(-\d{1,2})?)?$")


def _is_literalish(text: str) -> bool:
    """Reject punctuation and bare symbols as literal values.

    '?' in "...diagnosed with 'SLE'?" sits adjacent to a closing quote, so the
    quote-proximity heuristic marks it quoted; without this guard it became a real
    filter value (observed: Patient.SEX = '?').
    """
    s = (text or "").strip()
    return bool(s) and any(ch.isalnum() for ch in s)


def _literals(tree: DepTree) -> List[Unit]:
    out = []
    for lit in tree.quoted_literals:
        if _is_literalish(lit):
            out.append(Unit("literal", lit, [], [lit], {"quoted": True, "value": lit}))
    for n in tree.nodes:
        if n.pos in {"PUNCT", "SYM", "SPACE"} or not _is_literalish(n.text):
            continue
        if n.like_num or n.pos == "NUM":
            kind = "date" if _DATE_RE.match(n.text.strip()) else "number"
            out.append(Unit("literal", n.text, [n.i], _toks(tree, [n.i]),
                            {"quoted": False, "value": n.text, "literal_type": kind}))
        elif n.is_quoted and n.text not in tree.quoted_literals:
            out.append(Unit("literal", n.text, [n.i], _toks(tree, [n.i]),
                            {"quoted": True, "value": n.text}))
        elif n.ent_type in {"DATE", "CARDINAL", "MONEY", "PERCENT", "GPE", "ORG", "PERSON"}:
            out.append(Unit("literal", n.text, [n.i], _toks(tree, [n.i]),
                            {"quoted": False, "value": n.text, "ent_type": n.ent_type}))
    return out


def _mentions(tree: DepTree) -> List[Unit]:
    """Entity/attribute mentions = candidate schema references (nouns in argument
    or modifier positions). Grounding decides table vs column vs value."""
    out, seen = [], set()
    for n in tree.nodes:
        if n.pos not in {"NOUN", "PROPN"}:
            continue
        if n.lemma.lower() in _STOP_HEADS:
            continue
        text, ids = _mention_span(tree, n)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        kind = "entity" if n.dep in {"nsubj", "nsubjpass"} else "attribute"
        out.append(Unit(kind, text, ids, _toks(tree, ids),
                        {"dep": n.dep, "head_lemma": n.lemma,
                         "in_prep": _under_prep(tree, n)}))
    return out


def _under_prep(tree: DepTree, n: Node) -> bool:
    """prep attachment => filter/relationship/temporal condition (handoff §3)."""
    cur, guard = n, 0
    while cur.head != cur.i and guard < 8:
        cur = tree.nodes[cur.head]
        guard += 1
        if cur.dep == "prep" or cur.pos == "ADP":
            return True
    return False


_LIMIT_RE = re.compile(r"\btop\s+(\d+)\b|\bfirst\s+(\d+)\b", re.I)


def _limits(tree: DepTree) -> List[Unit]:
    m = _LIMIT_RE.search(tree.question or "")
    if not m:
        return []
    val = next((g for g in m.groups() if g), None)
    if not val:
        return []
    return [Unit("limit", m.group(0), [], [m.group(0)], {"value": int(val)})]


def coverage_mentions(units: List[Unit]) -> List[str]:
    """Question elements a coverage report must account for."""
    out, seen = [], set()
    for u in units:
        if u.kind in {"entity", "attribute", "literal", "aggregate", "comparison",
                      "ordering", "group", "limit"}:
            k = u.text.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(u.text.strip())
    return out
