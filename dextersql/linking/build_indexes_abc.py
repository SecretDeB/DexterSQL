"""
build_indexes_abc.py  (ABC fix 6)
---------------------------------
ABC variant of the FAISS semantic index: built over the **FULL** column profile
(`profile_full_en` = dev-doc + long stats) instead of just the long profile.
This is what the question is searched against during retrieval.

Writes SEPARATE index files so the original `<db>.faiss` (long-profile) is left
untouched:
    <db>.faiss_full           ← FAISS binary index over full profiles
    <db>.faiss_full_meta.json ← column metadata

`query_faiss_full` lazily builds the index on first use, then queries it.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

# build_indexes is on sys.path (inserted by schema_linking_abc); reuse its embedder.
from build_indexes import get_embeddings


def build_faiss_index_full(db_id: str, db_dir: Path) -> Path:
    """Build a FAISS index from the FULL profile text of every column."""
    import faiss
    import numpy as np

    full_path = db_dir / f"{db_id}.full_profiles.jsonl"
    if not full_path.exists():
        raise FileNotFoundError(f"Full profiles not found: {full_path}")

    with open(full_path) as f:
        profiles = [json.loads(l) for l in f if l.strip()]

    # full profile text; fall back to long if a row's full text is empty
    texts = [(r.get("profile_full_en") or r.get("profile_long_en") or
              f"{r['table']}.{r['column']}") for r in profiles]
    meta = [{"table": r["table"], "column": r["column"],
             "profile_full_en": r.get("profile_full_en") or r.get("profile_long_en") or ""}
            for r in profiles]

    embeddings = get_embeddings(texts)
    vecs = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss_path = db_dir / f"{db_id}.faiss_full"
    faiss.write_index(index, str(faiss_path))
    with open(db_dir / f"{db_id}.faiss_full_meta.json", "w") as f:
        json.dump(meta, f)
    print(f"  [ABC] built full-profile FAISS index: {faiss_path.name} "
          f"({len(profiles)} columns)")
    return faiss_path


def query_faiss_full(question: str, db_id: str, db_dir: Path,
                     top_k: int = 10) -> List[Dict[str, Any]]:
    """Top-k columns by question↔FULL-profile semantic similarity.

    Returns dicts shaped like build_indexes.query_faiss: {table, column,
    faiss_score, profile_full_en}. Lazily builds the index if absent.
    """
    import faiss
    import numpy as np

    faiss_path = db_dir / f"{db_id}.faiss_full"
    meta_path = db_dir / f"{db_id}.faiss_full_meta.json"
    if not faiss_path.exists() or not meta_path.exists():
        build_faiss_index_full(db_id, db_dir)

    index = faiss.read_index(str(faiss_path))
    with open(meta_path) as f:
        meta = json.load(f)

    q_emb = get_embeddings([question])[0]
    q_vec = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q_vec)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(q_vec, k)
    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        r = dict(meta[idx])
        r["faiss_score"] = float(score)
        results.append(r)
    return results
