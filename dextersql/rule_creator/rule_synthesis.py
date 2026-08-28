"""
Step 4 (Rule synthesis).

For each dominant error group from Step 3, synthesize a canonical
correction rule: {rule_name, gist, bad_pattern, correct_pattern, fix} --
a directive indicating when the rule applies and how to fix the
corresponding SQL error, plus a provenance note.

Input: config.error_groups_path (from clustering.run_clustering).
Output: config.created_rules_path -- a JSON list of rule dicts.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from dextersql.core.llm import make_llm
from dextersql.core.llm_extractor import LLMExtractor
from dextersql.core.logger import logger
from dextersql.rule_creator.config import RuleCreatorConfig
from dextersql.rule_creator.parsing import parse_rule_synthesis
from dextersql.rule_creator.prompts import format_rule_synthesis_prompt


def _load_error_groups(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)["groups"]


def _synthesize_one(group: Dict[str, Any], llm, max_examples: int) -> Dict[str, str] | None:
    extractor = LLMExtractor(max_retry=3)
    examples = group["member_rows"][:max_examples]
    prompt = format_rule_synthesis_prompt(group["label"], examples)
    results, _ = extractor.extract_with_retry(
        llm=llm,
        messages=[{"role": "user", "content": prompt}],
        rule_parser=parse_rule_synthesis,
        n=1,
    )
    return results[0] if results else None


def _dedupe_rule_name(name: str, used: set) -> str:
    if name not in used:
        return name
    i = 2
    while f"{name}-{i}" in used:
        i += 1
    return f"{name}-{i}"


def run_rule_synthesis(config: RuleCreatorConfig, error_groups: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if error_groups is None:
        error_groups = _load_error_groups(config.error_groups_path)
    print(f"[rule_synthesis] synthesizing rules for {len(error_groups)} dominant error groups")

    llm = make_llm(config.rule_synthesis.llm)
    max_examples = config.rule_synthesis.max_examples_per_rule

    rules: List[Dict[str, Any]] = []
    used_names: set = set()
    today = date.today().isoformat()

    for group in error_groups:
        synthesized = _synthesize_one(group, llm, max_examples)
        if synthesized is None:
            logger.warning(f"[rule_synthesis] group_id={group['group_id']} ('{group['label']}') failed to synthesize a rule, skipping")
            continue

        rule_name = _dedupe_rule_name(synthesized["rule_name"], used_names)
        used_names.add(rule_name)

        example_qids = [r["question_id"] for r in group["member_rows"][:max_examples]]
        rule = {
            "rule_name": rule_name,
            "gist": synthesized["gist"],
            "bad_pattern": synthesized["bad_pattern"],
            "correct_pattern": synthesized["correct_pattern"],
            "fix": synthesized["fix"],
            "note": (
                f"Synthesized by Rule Creator on {today} from a dominant error group "
                f"('{group['label']}') mined from BIRD training data: {group['support']} "
                f"execution-verified failure explanations across {len(group['db_ids'])} "
                f"training databases ({', '.join(group['db_ids'])}). Representative "
                f"training question_ids used for synthesis: {example_qids}."
            ),
        }
        rules.append(rule)
        print(f"[rule_synthesis]   {rule_name}  (support={group['support']}, {len(group['db_ids'])} dbs)")

    out_path = Path(config.created_rules_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rules, f, indent=2)

    print(f"[rule_synthesis] {len(rules)}/{len(error_groups)} groups synthesized into rules -> {out_path}")
    return rules
