__all__ = ["LLM", "make_llm"]


def __getattr__(name):
    if name == "LLM":
        from .llm import LLM
        return LLM
    if name == "make_llm":
        from ._factory import make_llm
        return make_llm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
