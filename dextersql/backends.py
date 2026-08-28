"""
LLM backends for DexterSQL.

`DepTreeGenerator` only needs a callable:

    llm_call(messages, n) -> (List[str] raw_responses, usage_dict)

so it stays decoupled from any particular client. This module provides a default
implementation against any OpenAI-compatible endpoint (vLLM, TGI, OpenAI,
OpenRouter, ...), plus a stub for tests.

The `n` samples must come from the SAME prompt — that is what gives the method
several candidates for one LLM call.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_ZERO = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class OpenAICompatibleBackend:
    """Chat-completions client for any OpenAI-compatible server.

    Notes
    -----
    * `n` is requested server-side when supported; if the server rejects it (some
      deployments do), we fall back to issuing n sequential requests so the caller
      still gets n samples.
    * Reasoning models may leave `message.content` empty and put the chain of
      thought in a non-standard `reasoning`/`reasoning_content` field. We return
      the visible content and fall back to the reasoning field only if content is
      empty, so a response is never silently dropped.
    """

    def __init__(self, model: str,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 timeout: float = 600.0,
                 max_retries: int = 2):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install openai  # required for OpenAICompatibleBackend") from e
        self.model = model
        self.base_url = base_url or os.environ.get("DEXTERSQL_BASE_URL") or None
        self.api_key = api_key or os.environ.get("DEXTERSQL_API_KEY") or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = self._make_client()
        self._supports_n = True

    def _make_client(self):
        import httpx
        import openai
        # Disable keep-alive pooling: long-lived self-hosted servers commonly close
        # idle sockets, and a reused dead socket can hang instead of raising.
        http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=32),
            timeout=self.timeout,
        )
        return openai.OpenAI(base_url=self.base_url, api_key=self.api_key,
                             http_client=http_client, max_retries=0)

    # ── the callable the generator wants ─────────────────────────────────────
    def __call__(self, messages: List[Dict[str, str]], n: int = 1
                 ) -> Tuple[List[str], Dict[str, int]]:
        n = max(1, int(n))
        if self._supports_n:
            try:
                return self._request(messages, n)
            except Exception as e:
                if "n" in str(e).lower() and "support" in str(e).lower():
                    self._supports_n = False   # fall through to sequential
                else:
                    raise
        texts: List[str] = []
        usage = dict(_ZERO)
        for _ in range(n):
            t, u = self._request(messages, 1)
            texts.extend(t)
            for k in usage:
                usage[k] += u.get(k, 0)
        return texts, usage

    def _request(self, messages, n) -> Tuple[List[str], Dict[str, int]]:
        import openai
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages, n=n,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                )
                texts = []
                for c in resp.choices:
                    msg = c.message
                    content = (getattr(msg, "content", None) or "").strip()
                    if not content:
                        content = (getattr(msg, "reasoning_content", None)
                                   or getattr(msg, "reasoning", None) or "").strip()
                    texts.append(content)
                u = getattr(resp, "usage", None)
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(u, "total_tokens", 0) or 0,
                } if u else dict(_ZERO)
                return texts, usage
            except (openai.APIConnectionError, openai.APITimeoutError,
                    openai.RateLimitError, openai.InternalServerError) as e:
                last = e
                self._client = self._make_client()      # drop possibly-dead socket
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM request failed after {self.max_retries + 1} attempts: {last}")


class StubBackend:
    """Returns canned responses in order. For tests — no network."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[List[Dict[str, str]]] = []

    def __call__(self, messages, n: int = 1) -> Tuple[List[str], Dict[str, int]]:
        self.calls.append(messages)
        out, self.responses = self.responses[:n], self.responses[n:]
        return out, dict(_ZERO)


def make_llm_call(model: str, base_url: Optional[str] = None,
                  api_key: Optional[str] = None, temperature: float = 0.7,
                  max_tokens: int = 2048) -> Callable:
    """Convenience factory for the default backend."""
    return OpenAICompatibleBackend(model=model, base_url=base_url, api_key=api_key,
                                   temperature=temperature, max_tokens=max_tokens)
