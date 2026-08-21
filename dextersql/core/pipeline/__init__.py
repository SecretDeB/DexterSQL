__all__ = [
    "ValueRetrievalRunner",
    "SQLGenerationRunner",
    "SQLRevisionRunner",
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
    if name == "SQLSelectionRunner":
        from .sql_selection import SQLSelectionRunner
        return SQLSelectionRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
