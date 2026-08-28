"""
Example correction rules for the sql_rule_correction stage.

These are a small, illustrative rule set -- enough to exercise the stage
end to end and to show the expected shape of a rule. They are NOT a tuned or
complete set: to build one for your own setting, mine it from your training
data with dextersql/rule_creator/, which automates that process end to end,
and point RULES here at the result.

Each rule targets one recurring, database-agnostic way a generated query can
diverge from the intended answer. These are semantic mismatches, not syntax or
execution errors -- the SQL runs fine, it just answers a subtly different
question -- so the sql_revision checker suite (syntax/execution diagnostics)
does not catch them. Each is stated as a pattern pair: a faulty formulation to
detect (bad_pattern) and the corresponding correct one (correct_pattern), plus
a fix directive.

2 rules active: RC-PROJECTION-EXTRA-COLUMN, RC-COLUMN-ORDER-MISMATCH.
3 kept in DISABLED_RULES below as documented negative results -- worth reading
before adding rules of your own, since each failed in an instructive way:
RC-DROPPED-FILTER-CONDITION (needs schema *value* knowledge the correction
prompt doesn't reliably carry -- a harder task than the other rules' purely
syntactic transformations), RC-MISSING-NULL-FILTER (net-negative: over-applies
IS NOT NULL guards to columns that should not be filtered), and
RC-PROJECTION-MISSING-COLUMN (the "too few columns" mirror of
RC-PROJECTION-EXTRA-COLUMN -- it passed every narrow-scope test, including a
held-out split with zero misfires, yet regressed by 10 questions at full
scale; a caution that narrow-scope validation of a correction rule can
mislead).

Rules carry only the five fields the model is shown (rule_name, gist,
bad_pattern, correct_pattern, fix).
"""

from typing import List, TypedDict


class CorrectionRule(TypedDict):
    rule_name: str
    gist: str
    bad_pattern: str
    correct_pattern: str
    fix: str


RULES: List[CorrectionRule] = [
    {
        "rule_name": "RC-PROJECTION-EXTRA-COLUMN",
        "gist": (
            "When a question asks to list or identify entities, return exactly the "
            "column(s) the question requests -- no more, no fewer, and never merged "
            "into one. Extra columns cause tuple-width mismatch and evaluation "
            "failure even when the data values are correct."
        ),
        "bad_pattern": (
            "TWO separate bad patterns, both about column COUNT, never about which "
            "specific column identifies the entity: "
            "(a) The SELECT list has MORE columns/expressions than the question asks "
            "for -- e.g. SELECT productName, SUM(qty) when gold returns only SELECT "
            "productName; SELECT user_id, stars when gold returns only SELECT stars; "
            "a numeric metric or aggregate tacked on next to the entity column the "
            "question actually wants. A compound question like \"What is <value>? "
            "Indicate/give/show <name/entity>.\" often has this shape too -- gold "
            "frequently returns ONLY the named entity/identifier clause and drops the "
            "value clause entirely (e.g. \"What is the height of the tallest player? "
            "Indicate his name.\" -> gold selects only `player_name`, not `height`). "
            "(b) The SELECT list MERGES two columns the question asks for into one "
            "via `||`, `CONCAT(...)`, `printf(...)`, or similar -- most commonly "
            "combining `forename`/`surname` (or similar first/last-name pairs) into a "
            "single \"full_name\" string because the question says \"give his full "
            "name.\" Gold virtually never does this: it returns forename and surname "
            "as two separate columns even when the question asks for a \"full name.\""
        ),
        "correct_pattern": (
            "(a) Drop only the columns that are clearly NOT what the question is "
            "asking about -- extra numeric metrics, aggregates, or unrelated "
            "attributes riding along with the actual answer column(s). "
            "(b) Never concatenate/merge separate requested columns -- keep "
            "`forename, surname` (or equivalent) as two columns, never `forename || "
            "' ' || surname AS full_name`."
        ),
        "fix": (
            "1. Re-read the question and list exactly which attribute(s) it asks for. "
            "2. Remove any SELECT column/expression that is not one of those "
            "attributes -- typically a stray metric, aggregate, or unrelated field. "
            "3. If any two of the columns you're keeping got merged into one via "
            "`||`/CONCAT/printf, split them back into separate columns. "
            "4. DO NOT choose between two columns that could each independently "
            "identify or name the requested entity (e.g. an `id` column vs a `name` "
            "column for the same entity) -- picking the 'right' one between two "
            "already-plausible identifier candidates is NOT this rule's job, and "
            "guessing wrong here has repeatedly turned already-correct SQL into "
            "wrong SQL. If the SELECT list's column COUNT already matches what the "
            "question needs, leave it alone even if you think a different column "
            "choice would read better -- this rule only removes genuine excess, it "
            "never substitutes one plausible column for another. "
            "5. Before removing a column, double-check it isn't simply MISPLACED: "
            "if the question asks for it too, just later in the sentence or via a "
            "trailing \"include/indicate\" clause, it belongs in the SELECT list "
            "somewhere -- removing it is wrong even if its current position looks "
            "backwards; reordering it into the right position is "
            "RC-COLUMN-ORDER-MISMATCH's job, not this rule's, so leave a "
            "misordered-but-wanted column in place rather than deleting it."
        ),
    },
    {
        "rule_name": "RC-COLUMN-ORDER-MISMATCH",
        "gist": (
            "When a question asks for multiple pieces of information, gold returns "
            "them in the SAME order the question asks for them -- the first "
            "attribute the question mentions comes first in SELECT, the second "
            "attribute second, and so on. Evaluation compares result rows as "
            "tuples, so (A, B) does NOT equal (B, A) even when both sides contain "
            "the exact same values -- column ORDER inside the SELECT list matters "
            "just as much as which columns are present."
        ),
        "bad_pattern": (
            "The candidate's SELECT list contains EXACTLY the same set of columns "
            "as the question asks for (nothing extra, nothing missing -- if a "
            "column is missing or extra, that's RC-PROJECTION-EXTRA-COLUMN's job, "
            "not this rule's), but in a different order than the question mentions "
            "them. E.g. \"What is the phone number and extension...Indicate the "
            "school's name\" (asks for phone, then extension, then school name) but "
            "the candidate selects School, Phone, Ext -- same three columns, wrong "
            "order."
        ),
        "correct_pattern": (
            "Reorder the SELECT list's items to match the order the question asks "
            "for them, changing nothing else -- same columns, same aliases, same "
            "expressions, only their POSITION in the list changes."
        ),
        "fix": (
            "1. Re-read the question left to right and list the attributes it asks "
            "for, in the order it asks for them -- including anything requested via "
            "a trailing \"indicate/include/also list ...\" clause, which counts as "
            "asked-for-last unless the question's own phrasing puts it earlier. "
            "2. Compare that order against the candidate's current SELECT list "
            "order. 3. If the columns are the same but the order differs, reorder "
            "the SELECT items to match -- do not add, remove, rename, or alter any "
            "column while doing this, it is a pure reordering. 4. If the column SET "
            "itself differs from what's asked (extra or missing columns), leave "
            "that alone -- it's out of scope for this rule."
        ),
    },
]

# Disabled: not sent to the LLM, not applied. Kept here for provenance and in
# case it's worth re-enabling later (e.g. rephrased more narrowly, or scoped
# to specific filter keywords) rather than re-deriving from scratch.
#
# Pulled from RULES after the first live test because its bad_pattern is too
# generic for the model to reliably pattern-match against ("a WHERE condition
# the question mentions is missing" covers too much ground without a much
# more specific trigger) -- user feedback after reviewing the rule set,
# 2026-08-21, before the first small HPC run.
DISABLED_RULES: List[CorrectionRule] = [
    {
        "rule_name": "RC-PROJECTION-MISSING-COLUMN",
        "gist": (
            "The mirror image of RC-PROJECTION-EXTRA-COLUMN: the question asks for "
            "multiple pieces of information (often a paired identifier + attribute, "
            "or two related values), but the candidate's SELECT list is missing one "
            "of them -- returning less information than the question asked for, not "
            "more. This is a distinct, common pattern in its own right, not just a "
            "rare inverse case."
        ),
        "bad_pattern": (
            "The candidate's SELECT list is a SUBSET of what the question asks for "
            "-- it's missing at least one specific column/attribute the question "
            "explicitly names or clearly implies via a pairing (e.g. \"list the "
            "product ID and description\" but only description is selected; \"the "
            "account ID and district code\" but only account ID is selected; \"the "
            "user who added it, and the post title\" but only the user is "
            "selected). If the SELECT list already has every column/attribute the "
            "question asks for, this rule does not apply, even if you'd have "
            "phrased the query differently."
        ),
        "correct_pattern": (
            "Add the missing column(s) to the SELECT list, using the exact column "
            "the question refers to, without altering anything else -- not the "
            "columns already present, not the FROM/JOIN structure, not any WHERE "
            "condition."
        ),
        "fix": (
            "1. Re-read the question and list every attribute it asks for. HIGH-"
            "CONFIDENCE triggers -- if you see one of these shapes, the second "
            "attribute is being asked for even though it's easy to read past: "
            "\"Who/which <person/entity> ... ? Show/give/indicate/what is <also X>\" "
            "-- \"who\"/\"which <entity>\" asks for that entity's name/identifier, "
            "and the trailing question asks for X too, so the answer needs BOTH; "
            "\"list/give <A> and <B>\"; \"the <A>, and the <B>\" -- explicit "
            "coordination naming two attributes. \"List/what is/what are the <attribute> "
            "of/for <the -- named/filtered -- entities>\" -- e.g. \"list the product "
            "description of the products bought in ...\", \"what are the labels for "
            "TR000, TR001, TR002\", \"list all the expenses incurred by X\": here the "
            "question names only ONE attribute, but because the query already joins to "
            "or filters the entity's own table to find those rows, BIRD's gold answer "
            "conventionally also includes that entity's own identifier column (e.g. "
            "ProductID, expense_id, molecule_id) alongside the requested attribute, so "
            "each row can be tied back to the specific entity it describes -- add that "
            "entity's identifier column too, PROVIDED it is not already implicitly "
            "unique/present (e.g. don't add it if the WHERE clause already filters to a "
            "single row, or if another selected column already uniquely names the row). "
            "PLACEMENT of any added column matters as much as adding it -- never just "
            "tack it onto the end by default. As a general rule, place the added "
            "column where its concept is first mentioned in the question, relative to "
            "the concepts already in the SELECT list (e.g. if the question asks about "
            "the missing value in its FIRST sentence/clause and the already-selected "
            "columns come from a LATER clause, the added column goes first). The one "
            "exception is an implicit entity identifier that the question never names "
            "at all (the case just above) -- BIRD conventionally lists that identifier "
            "BEFORE its descriptive attribute (id, then name/description) even though "
            "the question never mentions the id explicitly, so insert it FIRST, ahead "
            "of the attribute already in the SELECT list -- UNLESS the question "
            "explicitly asks to also/additionally include it as a trailing addendum "
            "(e.g. \"... please also include the set ID\"), in which case append it "
            "LAST, matching where the question itself mentions it. "
            "2. Compare against the "
            "candidate's current SELECT list -- if a high-confidence attribute has "
            "no corresponding column, add it. 3. Beyond those high-confidence "
            "shapes, only add a column when you can still confidently identify the "
            "SPECIFIC missing column from the question's own wording and the "
            "schema -- do not invent or guess at a column just to \"be safe\" if "
            "it isn't clearly implied (e.g. a question that names only ONE "
            "attribute, with no explicit second attribute or \"who\"/\"which\" "
            "framing, may still be a single-column answer -- don't add an "
            "identifier column just because the row has one). This is especially "
            "important when the missing piece would be a computed expression (a "
            "window function like RANK() OVER, or an aggregate like SUM(...)) "
            "rather than a plain column -- only add it if you can confidently "
            "reproduce the exact computation the question implies; if you're not "
            "sure, leave the candidate as-is rather than adding a guessed "
            "expression. 4. Don't touch anything else."
        ),
    },
    {
        "rule_name": "RC-DROPPED-FILTER-CONDITION",
        "gist": (
            "The question or evidence names a specific value, category, status, "
            "flag, or range that restricts which rows count -- and the COLUMN that "
            "would express it doesn't appear anywhere in the candidate's WHERE "
            "clause at all, so the query computes over a broader set of rows than "
            "the question actually asked about."
        ),
        "bad_pattern": (
            "A specific qualifier the question/evidence explicitly states has NO "
            "column representing it anywhere in the WHERE clause. If some column "
            "IS already being filtered in a way that plausibly represents this "
            "qualifier (even a wrong column, wrong type, or wrong literal), that "
            "is a DIFFERENT bug and out of scope -- only act when the qualifier "
            "has no representation in WHERE whatsoever."
        ),
        "correct_pattern": (
            "Add the missing condition to the WHERE clause, ANDed with the "
            "existing conditions, using the correct column and the exact literal "
            "value/range the question or evidence specifies."
        ),
        "fix": (
            "For each specific value/category/status/flag/range the question or "
            "evidence states, identify which schema column would express it and "
            "check whether that column appears anywhere in WHERE; if it's "
            "completely absent, add a condition for it. Don't touch anything else."
        ),
    },
    {
        "rule_name": "RC-MISSING-NULL-FILTER",
        "gist": (
            "Gold SQL adds an `IS NOT NULL` guard on a nullable column somewhere in "
            "the query -- most often a column the query actually RETURNS (the SELECT "
            "list), but also columns used for ordering/aggregating/extremum lookup. "
            "Generated SQL omits this guard, allowing NULL rows to leak into the "
            "result or affect ranking."
        ),
        "bad_pattern": (
            "A column that is nullable per the schema appears in the SELECT list, "
            "in ORDER BY ... LIMIT 1, or inside MIN(col)/MAX(col) -- and the query "
            "has no `col IS NOT NULL` condition guarding it."
        ),
        "correct_pattern": (
            "Add `AND col IS NOT NULL` (or `WHERE col IS NOT NULL`) to the query's "
            "filter conditions for the nullable column(s) being returned, ordered "
            "by, or aggregated on."
        ),
        "fix": (
            "Check every column in the SELECT list, ORDER BY, MIN(), or MAX(): if "
            "it's nullable per the schema and has no existing NULL guard, add "
            "`col IS NOT NULL`."
        ),
    },
]

RULE_NAMES: List[str] = [r["rule_name"] for r in RULES]


def load_rules(rules_path: str = "") -> List[CorrectionRule]:
    """Return the active rule set.

    With no `rules_path`, the example rules above are used. Pass a path to a
    JSON list produced by dextersql/rule_creator/ (its 05_created_rules.json)
    to use a mined rule set instead. Each entry must carry the five fields the
    model is shown; any extra keys (e.g. a provenance note) are ignored.
    """
    if not rules_path:
        return RULES

    import json
    from pathlib import Path

    path = Path(rules_path)
    if not path.exists():
        raise FileNotFoundError(
            f"sql_rule_correction.rules_path points at {path}, which does not exist. "
            f"Run dextersql/rule_creator/ to produce it, or unset rules_path to use "
            f"the shipped example rules."
        )
    loaded = json.load(open(path))
    if not isinstance(loaded, list) or not loaded:
        raise ValueError(f"{path} must contain a non-empty JSON list of rules.")

    required = ("rule_name", "gist", "bad_pattern", "correct_pattern", "fix")
    cleaned = []
    for i, r in enumerate(loaded):
        missing = [k for k in required if not r.get(k)]
        if missing:
            raise ValueError(f"{path}: rule #{i + 1} is missing {missing}.")
        cleaned.append({k: r[k] for k in required})
    return cleaned
