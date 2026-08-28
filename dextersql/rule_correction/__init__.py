__all__ = ["SQLRuleCorrectionRunner"]


def __getattr__(name):
    if name == "SQLRuleCorrectionRunner":
        from .sql_rule_correction import SQLRuleCorrectionRunner
        return SQLRuleCorrectionRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
