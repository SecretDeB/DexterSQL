"""Factory that instantiates the correct LLM backend from LLMConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from dextersql.core.config.config import LLMConfig
    from .llm import LLM
    from .vllm_backend import VLLMLLM


def make_llm(llm_config: "LLMConfig") -> "Union[LLM, VLLMLLM]":
    """Return LLM or VLLMLLM depending on config.api_type."""
    if llm_config.api_type == "gpt5_oss":
        from .gpt5_oss_backend import GPT5OssLLM
        return GPT5OssLLM(llm_config)
    if llm_config.api_type == "gpt5":
        from .gpt5_backend import GPT5LLM
        return GPT5LLM(llm_config)
    if llm_config.api_type == "vllm":
        from .vllm_backend import VLLMLLM
        return VLLMLLM(llm_config)
    from .llm import LLM
    return LLM(llm_config)
