"""
GPT5OssLLM — reasoning-model backend, VERSION A ("from oss").

Derived directly from the pattern the project already uses for its reasoning
model (GPT-OSS-120B): keep the <reasoning>/<result> XML tags, DO NOT inject
###FINAL### markers (see vllm_backend.py note — markers fight the <result>
parsers), and when the OpenAI-compatible endpoint returns an empty `content`
because the model spent its budget in the hidden reasoning channel, fall back
to `reasoning_content` — the exact fallback used in
rule_alignment/rule_aligner.py:117-122.

This is a minimal, faithful port of that behaviour onto the OpenRouter chat
path so it can be compared head-to-head with VERSION B (gpt5_backend.py).

Wired via app/llm/_factory.py on config field `llm_backend_variant == "gpt5_oss"`.

NOTE: untested draft — intended to be scp'd to HPC and validated by the 20-Q
smoke test before any full batch.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import (
    OpenAI,
    OpenAIError,
    AuthenticationError,
    RateLimitError,
    BadRequestError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)
from dextersql.core.logger import logger

if TYPE_CHECKING:
    from dextersql.core.config.config import LLMConfig


class _Message:
    """Duck-typed ChatCompletionMessage so downstream parsers work unchanged."""
    __slots__ = ("content", "role")

    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"


class EmptyResponseError(Exception):
    pass


class GPT5OssLLM:
    """Drop-in ask()-compatible backend that mirrors the GPT-OSS reasoning path."""

    def __init__(self, llm_config: "LLMConfig") -> None:
        self._config = llm_config
        self._client: Optional[OpenAI] = None
        self._lock = threading.Lock()
        # 25K floor: OpenAI recommends reserving >=25k tokens for reasoning+output.
        # sel's stock 1500 would be spent entirely on hidden reasoning -> empty.
        self._budget = max(int(llm_config.max_tokens or 0), 25_000)
        logger.debug(
            f"GPT5OssLLM ready: model={llm_config.model}, budget={self._budget}, "
            f"reasoning_effort={llm_config.reasoning_effort}"
        )

    @property
    def llm_config(self) -> "LLMConfig":
        return self._config

    def _get_client(self) -> OpenAI:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = OpenAI(
                        api_key=self._config.api_key,
                        base_url=self._config.base_url,
                    )
        return self._client

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(15),
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError,
             InternalServerError, BadRequestError, EmptyResponseError)
        ),
    )
    def ask(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[Dict[str, str]] = None,
        timeout: int = 300,
        **kwargs,
    ) -> Tuple[List[_Message], Dict[str, int]]:
        if system_message:
            messages = [system_message] + messages

        target_n = kwargs.pop("n", 1)
        max_request_n = self._config.max_request_n or target_n
        budget = kwargs.pop("max_tokens", None) or self._budget

        all_choices: List[_Message] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while len(all_choices) < target_n:
            current_n = min(target_n - len(all_choices), max_request_n)
            params = {
                "model": self._config.model,
                "messages": messages,
                # reasoning models require max_completion_tokens, NOT max_tokens
                "max_completion_tokens": budget,
                "timeout": timeout,
                "n": current_n,
            }
            # temperature is not honored by reasoning models; OpenRouter tolerates
            # (ignores) it, so we pass it through to stay faithful to the oss path.
            if self._config.temperature is not None:
                params["temperature"] = self._config.temperature
            if self._config.reasoning_effort is not None:
                params["reasoning_effort"] = self._config.reasoning_effort
            params.update(kwargs)

            try:
                resp = self._get_client().chat.completions.create(**params)
            except BadRequestError as e:
                msg = str(e).lower()
                if ("context" in msg or "length" in msg) and budget > 8000:
                    budget = int(budget * 0.9)
                    logger.warning(f"Context error; reducing budget to {budget} and retrying.")
                    continue
                raise
            except AuthenticationError:
                logger.error("Authentication error — check the OpenRouter key.")
                raise

            if not resp.choices:
                raise EmptyResponseError(f"No choices: {resp}")

            for ch in resp.choices:
                content = (ch.message.content or "").strip()
                if not content:
                    # oss fallback: answer emitted only in the reasoning channel
                    content = (getattr(ch.message, "reasoning_content", None) or "").strip()
                if not content:
                    raise EmptyResponseError(
                        "Empty content AND empty reasoning_content — likely budget "
                        f"exhausted by reasoning (budget={budget})."
                    )
                all_choices.append(_Message(content))

            if resp.usage:
                usage["prompt_tokens"] += resp.usage.prompt_tokens
                usage["completion_tokens"] += resp.usage.completion_tokens
                usage["total_tokens"] += resp.usage.total_tokens

        return all_choices, usage
