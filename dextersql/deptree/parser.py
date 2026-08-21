"""
Step 1-2: dependency parse + tree normalization.  DETERMINISTIC — no LLM.

Uses spaCy `en_core_web_sm`, whose English label set is exactly the one the
dep->SQL rules reference (nsubj/nsubjpass/dobj/pobj/prep/amod/nummod/conj/cc/
acl/relcl/advcl/ccomp/neg/advmod).

Normalization collapses multi-token units into single nodes so that
"patient ID", "how many", "at least", "New York", "most recent" behave as one
semantic unit downstream. Collapsing is span-based: the merged node keeps the
HEAD token's dependency relation, so the tree stays well-formed.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Host pipelines commonly drive items through a thread pool, and spaCy's
# Language object is not guaranteed thread-safe for concurrent __call__. Parsing is
# sub-millisecond per question, so serializing it costs almost nothing next to the
# LLM calls — and avoids corrupt/garbled parses under load.
_NLP_LOCK = threading.Lock()

_NLP = None
_NLP_ERR: Optional[str] = None


def get_nlp():
    """Load spaCy once per process. Parser+tagger only (NER kept for literals)."""
    global _NLP, _NLP_ERR
    if _NLP is not None or _NLP_ERR is not None:
        return _NLP
    with _NLP_LOCK:                       # double-checked: 16 threads may race here
        if _NLP is not None or _NLP_ERR is not None:
            return _NLP
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception as e:  # fail-closed: caller falls back to LLM-only path
            _NLP_ERR = str(e)
            _NLP = None
    return _NLP


def nlp_error() -> Optional[str]:
    return _NLP_ERR


# ── phrase inventory for normalization ───────────────────────────────────────
# Ordered longest-first so "at least" wins over "least".
AGG_PHRASES = {
    "how many": "COUNT", "how much": "SUM", "number of": "COUNT",
    "count of": "COUNT", "total number": "COUNT", "total of": "SUM",
    "sum of": "SUM", "average of": "AVG", "average": "AVG", "mean of": "AVG",
    "maximum": "MAX", "minimum": "MIN", "highest": "MAX", "lowest": "MIN",
    "largest": "MAX", "smallest": "MIN", "oldest": "MIN", "youngest": "MAX",
    "most recent": "MAX", "earliest": "MIN", "latest": "MAX",
    "percentage": "PCT", "percent": "PCT", "ratio": "RATIO",
}

CMP_PHRASES = {
    "at least": ">=", "at most": "<=", "no less than": ">=", "no more than": "<=",
    "greater than or equal to": ">=", "less than or equal to": "<=",
    "greater than": ">", "less than": "<", "more than": ">", "fewer than": "<",
    "larger than": ">", "smaller than": "<", "higher than": ">", "lower than": "<",
    "not equal to": "!=", "equal to": "=", "equals": "=", "beyond": ">",
    "above": ">", "below": "<", "under": "<", "over": ">",
    # temporal comparisons — without these, "born after 1930" silently degrades to
    # Birthday = '1930' (the grounding is right, the operator is lost).
    "after": ">", "before": "<", "since": ">=", "until": "<=",
    "later than": ">", "earlier than": "<", "prior to": "<", "no later than": "<=",
    "on or after": ">=", "on or before": "<=", "starting from": ">=",
}

ORDER_PHRASES = {
    "highest": "DESC", "largest": "DESC", "most": "DESC", "maximum": "DESC",
    "top": "DESC", "greatest": "DESC", "latest": "DESC", "most recent": "DESC",
    "lowest": "ASC", "smallest": "ASC", "least": "ASC", "minimum": "ASC",
    "earliest": "ASC", "oldest": "ASC", "bottom": "ASC",
}

GROUP_PHRASES = ("for each", "per", "by each", "grouped by", "for every")

NEG_WORDS = {"not", "no", "never", "without", "n't", "none", "neither", "except"}

# multi-token phrases we collapse into one node (superset of the above keys)
_COLLAPSE_PHRASES = sorted(
    set(list(AGG_PHRASES) + list(CMP_PHRASES) + list(GROUP_PHRASES) +
        ["primary key", "foreign key", "date of birth", "full name", "first name",
         "last name", "zip code", "phone number", "in total", "as well as"]),
    key=len, reverse=True,
)


@dataclass
class Node:
    """One node of the normalized dependency tree."""
    i: int                       # node index (position in `nodes`)
    text: str                    # surface text (possibly multi-token)
    lemma: str
    pos: str
    tag: str
    dep: str                     # dependency relation to head
    head: int                    # index into `nodes` (self == root)
    start_char: int
    end_char: int
    children: List[int] = field(default_factory=list)
    like_num: bool = False
    ent_type: str = ""
    is_quoted: bool = False
    phrase_kind: str = ""        # agg | cmp | group | "" — set when collapsed
    phrase_value: str = ""       # COUNT / >= / DESC ...

    def to_dict(self) -> Dict[str, Any]:
        return {"i": self.i, "text": self.text, "lemma": self.lemma, "pos": self.pos,
                "dep": self.dep, "head": self.head, "children": self.children,
                "phrase_kind": self.phrase_kind, "phrase_value": self.phrase_value,
                "like_num": self.like_num, "ent_type": self.ent_type,
                "is_quoted": self.is_quoted}


@dataclass
class DepTree:
    question: str
    nodes: List[Node] = field(default_factory=list)
    root: int = 0
    quoted_literals: List[str] = field(default_factory=list)

    def node(self, i: int) -> Optional[Node]:
        return self.nodes[i] if 0 <= i < len(self.nodes) else None

    def subtree_text(self, i: int) -> str:
        """Surface text of the subtree rooted at i, in original order."""
        idxs = self._subtree_idxs(i)
        toks = sorted((self.nodes[j] for j in idxs), key=lambda n: n.start_char)
        return " ".join(t.text for t in toks)

    def _subtree_idxs(self, i: int, seen: Optional[set] = None) -> List[int]:
        seen = seen if seen is not None else set()
        if i in seen or not (0 <= i < len(self.nodes)):
            return []
        seen.add(i)
        out = [i]
        for c in self.nodes[i].children:
            out.extend(self._subtree_idxs(c, seen))
        return out

    def find(self, **kw) -> List[Node]:
        """find(dep='nsubj'), find(phrase_kind='agg'), ..."""
        out = []
        for n in self.nodes:
            if all(getattr(n, k, None) == v for k, v in kw.items()):
                out.append(n)
        return out

    def render(self) -> str:
        """Compact text rendering handed to the LLM as fixed structural input."""
        lines = []
        for n in self.nodes:
            head = "ROOT" if n.head == n.i else self.nodes[n.head].text
            extra = f" [{n.phrase_kind}:{n.phrase_value}]" if n.phrase_kind else ""
            lines.append(f"  {n.i}: {n.text!r} pos={n.pos} dep={n.dep} head={head!r}{extra}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "root": self.root,
                "quoted_literals": self.quoted_literals,
                "nodes": [n.to_dict() for n in self.nodes]}


def _find_phrase_spans(doc) -> List[Tuple[int, int, str, str]]:
    """Locate collapsible phrases as (start_tok, end_tok_exclusive, kind, value)."""
    lower = [t.lower_ for t in doc]
    spans: List[Tuple[int, int, str, str]] = []
    taken = set()
    for phrase in _COLLAPSE_PHRASES:
        parts = phrase.split()
        n = len(parts)
        if n < 2:
            continue
        for s in range(len(lower) - n + 1):
            if any(k in taken for k in range(s, s + n)):
                continue
            if lower[s:s + n] == parts:
                kind, val = _phrase_kind(phrase)
                spans.append((s, s + n, kind, val))
                taken.update(range(s, s + n))
    # entities (New York) and quoted strings collapse too
    for ent in getattr(doc, "ents", []):
        if len(ent) > 1 and not any(k in taken for k in range(ent.start, ent.end)):
            spans.append((ent.start, ent.end, "", ""))
            taken.update(range(ent.start, ent.end))
    return sorted(spans)


def _phrase_kind(phrase: str) -> Tuple[str, str]:
    if phrase in AGG_PHRASES:
        return "agg", AGG_PHRASES[phrase]
    if phrase in CMP_PHRASES:
        return "cmp", CMP_PHRASES[phrase]
    if phrase in GROUP_PHRASES:
        return "group", "GROUP"
    return "", ""


def parse_question(question: str) -> Optional[DepTree]:
    """Parse + normalize. Returns None if spaCy is unavailable (caller degrades)."""
    nlp = get_nlp()
    if nlp is None or not (question or "").strip():
        return None
    with _NLP_LOCK:          # see _NLP_LOCK: runner is multi-threaded
        doc = nlp(question)

    # collapse spans -> map original token idx -> node idx
    spans = _find_phrase_spans(doc)
    span_of = {}
    for (s, e, kind, val) in spans:
        for k in range(s, e):
            span_of[k] = (s, e, kind, val)

    node_of_tok: Dict[int, int] = {}
    nodes: List[Node] = []
    tok_i = 0
    while tok_i < len(doc):
        if tok_i in span_of:
            s, e, kind, val = span_of[tok_i]
            span = doc[s:e]
            head_tok = span.root
            idx = len(nodes)
            nodes.append(Node(
                i=idx, text=span.text, lemma=" ".join(t.lemma_ for t in span),
                pos=head_tok.pos_, tag=head_tok.tag_, dep=head_tok.dep_, head=-1,
                start_char=span.start_char, end_char=span.end_char,
                like_num=any(t.like_num for t in span),
                ent_type=head_tok.ent_type_, phrase_kind=kind, phrase_value=val))
            for k in range(s, e):
                node_of_tok[k] = idx
            tok_i = e
        else:
            t = doc[tok_i]
            if t.is_space:
                tok_i += 1
                continue
            idx = len(nodes)
            nodes.append(Node(
                i=idx, text=t.text, lemma=t.lemma_, pos=t.pos_, tag=t.tag_,
                dep=t.dep_, head=-1, start_char=t.idx, end_char=t.idx + len(t.text),
                like_num=t.like_num, ent_type=t.ent_type_))
            node_of_tok[tok_i] = idx
            tok_i += 1

    # rebuild head links through the collapse map
    root = 0
    for tok in doc:
        if tok.is_space or tok.i not in node_of_tok:
            continue
        ni = node_of_tok[tok.i]
        hi = node_of_tok.get(tok.head.i, ni)
        n = nodes[ni]
        if n.head != -1:
            continue                       # head already set by the span's root
        if hi == ni or tok.dep_ == "ROOT":
            n.head = ni
            if tok.dep_ == "ROOT":
                root = ni
        else:
            n.head = hi
    for n in nodes:
        if n.head == -1:
            n.head = n.i
        if n.head != n.i:
            nodes[n.head].children.append(n.i)

    tree = DepTree(question=question, nodes=nodes, root=root)
    tree.quoted_literals = extract_quoted(question)
    _mark_quoted(tree)
    return tree


def extract_quoted(question: str) -> List[str]:
    """Literals in single/double quotes — high-precision value mentions."""
    out = []
    for m in re.finditer(r"['\"]([^'\"]{1,80})['\"]", question or ""):
        v = m.group(1).strip()
        if v:
            out.append(v)
    return out


def _mark_quoted(tree: DepTree) -> None:
    q = tree.question
    for n in tree.nodes:
        left = q[max(0, n.start_char - 1):n.start_char]
        right = q[n.end_char:n.end_char + 1]
        if left in ("'", '"') or right in ("'", '"'):
            n.is_quoted = True
        for lit in tree.quoted_literals:
            if n.text and n.text in lit:
                n.is_quoted = True
