"""
GPT5LLM — reasoning-model backend, VERSION B ("my own").

Same goal as gpt5_oss_backend.py (VERSION A) but a cleaner, more defensive
implementation purpose-built for gpt-5 via OpenRouter. Differences vs A:

  * OMITS `temperature` entirely for reasoning models. Reasoning models pin
    temperature; sending 0.7 errors on the *direct* OpenAI endpoint (OpenRouter
    only tolerates it by ignoring). Omitting keeps this backend portable.
  * Treats finish_reason == "length" as a real failure: the visible answer was
    truncated mid-<result>, so it bumps the budget and retries instead of
    handing a half-written SQL to the parser.
  * Logs reasoning-token usage separately so the (hidden, billed) reasoning cost
    is visible during the smoke test.
  * content -> reasoning_content fallback (same safety net as A / rule_aligner).

Keeps the <reasoning>/<result> XML contract; injects NO markers.

Wired via app/llm/_factory.py on config field `llm_backend_variant == "gpt5"`.

NOTE: untested draft — validate with the 20-Q smoke test on HPC first.
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

# Hard ceiling so a runaway "length" retry loop can't bankrupt a batch.
_MAX_BUDGET = 60_000


class _Message:
    __slots__ = ("content", "role")

    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"


class EmptyResponseError(Exception):
    pass


class GPT5LLM:
    """Clean OpenRouter reasoning backend, ask()-compatible with dextersql.core.llm.LLM."""

    def __init__(self, llm_config: "LLMConfig") -> None:
        self._config = llm_config
        self._client: Optional[OpenAI] = None
        self._lock = threading.Lock()
        # >=25k reasoning+output headroom per OpenAI guidance.
        self._budget = min(max(int(llm_config.max_tokens or 0), 25_000), _MAX_BUDGET)
        self._effort = llm_config.reasoning_effort or "medium"
        logger.debug(
            f"GPT5LLM ready: model={llm_config.model}, budget={self._budget}, "
            f"reasoning_effort={self._effort} (temperature omitted for reasoning)"
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

    @staticmethod
    def _extract(msg) -> str:
        """content, else the reasoning channel (finished-but-content-empty case)."""
        content = (getattr(msg, "content", None) or "").strip()
        if content:
            return content
        return (getattr(msg, "reasoning_content", None) or "").strip()

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
        budget = min(kwargs.pop("max_tokens", None) or self._budget, _MAX_BUDGET)
        kwargs.pop("temperature", None)  # never forward temperature to a reasoning model

        all_choices: List[_Message] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while len(all_choices) < target_n:
            current_n = min(target_n - len(all_choices), max_request_n)
            params = {
                "model": self._config.model,
                "messages": messages,
                "max_completion_tokens": budget,
                "reasoning_effort": self._effort,
                "timeout": timeout,
                "n": current_n,
            }
            params.update(kwargs)

            try:
                resp = self._get_client().chat.completions.create(**params)
            except BadRequestError as e:
                m = str(e).lower()
                if ("context" in m or "length" in m) and budget > 8000:
                    budget = int(budget * 0.9)
                    logger.warning(f"Context error; budget -> {budget}; retrying.")
                    continue
                raise
            except AuthenticationError:
                logger.error("Auth error — check the OpenRouter key.")
                raise

            if not resp.choices:
                raise EmptyResponseError(f"No choices: {resp}")

            # If the visible answer was cut off, the <result> tag is likely
            # incomplete -> grow the budget and retry rather than parse garbage.
            truncated = any(getattr(c, "finish_reason", None) == "length" for c in resp.choices)
            if truncated and budget < _MAX_BUDGET:
                budget = min(int(budget * 1.5), _MAX_BUDGET)
                logger.warning(f"finish_reason=length; budget -> {budget}; retrying.")
                raise EmptyResponseError("truncated (finish_reason=length)")

            for ch in resp.choices:
                content = self._extract(ch.message)
                if not content:
                    raise EmptyResponseError(
                        f"Empty content and reasoning_content (budget={budget})."
                    )
                all_choices.append(_Message(content))

            if resp.usage:
                usage["prompt_tokens"] += resp.usage.prompt_tokens
                usage["completion_tokens"] += resp.usage.completion_tokens
                usage["total_tokens"] += resp.usage.total_tokens
                # surface hidden reasoning tokens for cost visibility during smoke test
                details = getattr(resp.usage, "completion_tokens_details", None)
                rt = getattr(details, "reasoning_tokens", None) if details else None
                if rt:
                    logger.info(f"[gpt5] reasoning_tokens={rt} completion={resp.usage.completion_tokens}")

        return all_choices, usage
