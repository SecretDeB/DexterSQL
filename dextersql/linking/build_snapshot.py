"""
Merge schema-linking results into the pipeline snapshot.

The value-retrieval stage produces a snapshot in which every item carries
`database_schema_after_value_retrieval` — the full schema annotated with example
values drawn from the database. The schema linker separately produces, per
question, the set of tables/columns it believes the query needs.

This module joins the two: it replaces each item's linked tables/columns with the
linker's output and rebuilds the focused schema that SQL generation will see,
carrying the retrieved value examples through. Primary and foreign keys are always
force-included so joins remain expressible.

Anything the linker names that does not exist in the schema is dropped (and
counted); a question whose links come back empty keeps whatever links the snapshot
already had, so an item is never left with no schema at all.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from ..core.db_utils.schema import filter_used_database_schema


def links_by_question_id(links_path: str) -> Dict[Any, Dict[str, List[str]]]:
    """Load linker output into {question_id: {table: [columns]}}.

    Expected shape:
        {"results": [{"question_id": 0,
                      "schema_links": [{"table": "T", "column": "C"}, ...]}, ...]}
    """
    with open(links_path) as f:
        payload = json.load(f)
    out: Dict[Any, Dict[str, List[str]]] = {}
    for row in payload.get("results", []):
        links: Dict[str, List[str]] = {}
        for x in row.get("schema_links", []):
            t, c = x.get("table"), x.get("column")
            if t and c:
                links.setdefault(t, []).append(c)
        out[row["question_id"]] = links
    return out


def remap_to_schema_case(links: Dict[str, List[str]],
                         schema: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Map linker table/column names onto the schema's exact casing.

    Linkers routinely emit `patient.id` where the schema says `Patient.ID`; a
    case-sensitive lookup downstream would silently lose the column. Names with no
    counterpart in the schema are dropped and returned for logging.
    """
    tmap = {t.lower(): t for t in schema["tables"]}
    out: Dict[str, List[str]] = {}
    dropped: List[str] = []
    for t, cols in links.items():
        real_t = tmap.get(t.lower())
        if real_t is None:
            dropped.append(t)
            continue
        cmap = {c.lower(): c for c in schema["tables"][real_t]["columns"]}
        kept: List[str] = []
        for c in cols:
            real_c = cmap.get(c.lower())
            if real_c is None:
                dropped.append(f"{t}.{c}")
            elif real_c not in kept:
                kept.append(real_c)
        if kept:
            out[real_t] = kept
    return out, dropped


def build_snapshot(src_items: str, src_meta: str, links_path: str, out_dir: str,
                   verbose: bool = True) -> Dict[str, int]:
    """Write a schema-linking snapshot combining value retrieval + linker output.

    Args:
        src_items: items.jsonl from the value-retrieval stage
        src_meta:  the matching .snapshot metadata file
        links_path: linker results JSON
        out_dir:   destination directory (dev.snapshot + dev.snapshot.data/ land here)

    Returns counts: {items, empty_links, dropped}
    """
    links_by_qid = links_by_question_id(links_path)
    if verbose:
        print(f"[snapshot] linker results for {len(links_by_qid)} questions")

    data_dir = os.path.join(out_dir, "dev.snapshot.data")
    os.makedirs(data_dir, exist_ok=True)

    n = n_dropped = n_empty = 0
    sample_dropped: List[str] = []

    with open(src_items) as fin, open(os.path.join(data_dir, "items.jsonl"), "w") as fout:
        for line in fin:
            item = json.loads(line)
            qid = item["input"]["question_id"]
            schema_vr = (item["pipeline_artifacts"]["value_retrieval"]
                             ["database_schema_after_value_retrieval"])

            links, dropped = remap_to_schema_case(links_by_qid.get(qid, {}), schema_vr)
            if dropped:
                n_dropped += len(dropped)
                if len(sample_dropped) < 10:
                    sample_dropped.extend(dropped[:2])

            if not links:
                # Fail-safe: keep the snapshot's existing links rather than emitting
                # an item with no schema, which would make generation impossible.
                n_empty += 1
                if verbose:
                    print(f"  [warn] qid={qid}: linker returned nothing — keeping existing links")
            else:
                sl = item["pipeline_artifacts"]["schema_linking"]
                sl["final_linked_tables_and_columns"] = links
                sl["database_schema_after_schema_linking"] = filter_used_database_schema(
                    schema_vr, links, force_include_pks_and_fks=True)
                # Downstream generation reads only database_schema_after_schema_linking,
                # but the schema_linking artifact carries auxiliary fields the stock
                # linker would have filled. Our source is the value-retrieval snapshot,
                # where they are None. Fill them from `links` (and zero the cost/time)
                # so the merged snapshot is self-consistent and is_stage_complete()
                # holds if anything re-validates it later.
                for aux in ("direct_linked_tables_and_columns",
                            "reversed_linked_tables_and_columns",
                            "value_linked_tables_and_columns"):
                    if sl.get(aux) is None:
                        sl[aux] = links
                if sl.get("schema_linking_time") is None:
                    sl["schema_linking_time"] = 0.0
                if sl.get("schema_linking_llm_cost") is None:
                    sl["schema_linking_llm_cost"] = {
                        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            fout.write(json.dumps(item) + "\n")
            n += 1
            if verbose and n % 300 == 0:
                print(f"  ...{n} items", flush=True)

    with open(src_meta) as f:
        meta = json.load(f)
    meta["num_items"] = n
    with open(os.path.join(out_dir, "dev.snapshot"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"[snapshot] wrote {n} items -> {out_dir}")
        print(f"[snapshot] empty-link fallbacks: {n_empty} | dropped (not in schema): {n_dropped}")
        if sample_dropped:
            print(f"[snapshot] sample dropped: {sample_dropped}")

    return {"items": n, "empty_links": n_empty, "dropped": n_dropped}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Merge linker output into a snapshot.")
    ap.add_argument("--src-items", required=True, help="value-retrieval items.jsonl")
    ap.add_argument("--src-meta", required=True, help="value-retrieval .snapshot file")
    ap.add_argument("--links", required=True, help="schema linker results JSON")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    build_snapshot(a.src_items, a.src_meta, a.links, a.out_dir)


if __name__ == "__main__":
    main()
