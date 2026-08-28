__all__ = [
    "ValueRetrievalRunner",
    "SQLGenerationRunner",
    "SQLRevisionRunner",
    "SQLRuleCorrectionRunner",
    "SQLSelectionRunner",
]


def __getattr__(name):
    if name == "ValueRetrievalRunner":
        from .value_retrieval import ValueRetrievalRunner
        return ValueRetrievalRunner
    if name == "SQLGenerationRunner":
        from .sql_generation import SQLGenerationRunner
        return SQLGenerationRunner
    if name == "SQLRevisionRunner":
        from .sql_revision import SQLRevisionRunner
        return SQLRevisionRunner
    if name == "SQLRuleCorrectionRunner":
        # Original DexterSQL code, deliberately kept out of this vendored
        # (MIT-licensed third-party) core/pipeline tree -- it lives in
        # dextersql/rule_correction/.
        from dextersql.rule_correction.sql_rule_correction import SQLRuleCorrectionRunner
        return SQLRuleCorrectionRunner
    if name == "SQLSelectionRunner":
        from .sql_selection import SQLSelectionRunner
        return SQLSelectionRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
