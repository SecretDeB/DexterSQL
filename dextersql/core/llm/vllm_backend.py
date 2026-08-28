"""
vLLM backend for GPT-OSS 120B (and other large OSS models).

Implements the same ask() interface as LLM so it's a drop-in replacement
for any pipeline stage. Uses:
  - vLLM PagedAttention engine for high-throughput inference
  - FP4 quantization by default (fits 120B on 4×A100 80GiB)
  - Tensor parallelism auto-detected from CUDA_VISIBLE_DEVICES

NOTE on marker injection:
  The Ambiguity_ project uses ###FINAL_BEGIN### / ###FINAL_END### markers
  to separate reasoning from answers. We do NOT inject those here because
  Pipeline prompts already use <reasoning> + <result> XML tags for the
  same purpose. Injecting markers would create competing output format
  instructions and could cause the model to drop the <result> tags that
  all downstream parsers depend on.

Engine is cached at class level — reloading 120B weights kills the kernel.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from dextersql.core.logger import logger

if TYPE_CHECKING:
    from dextersql.core.config.config import LLMConfig


# ---------------------------------------------------------------------------
# Minimal message wrapper — duck-typed replacement for
# openai.types.chat.ChatCompletionMessage so downstream code works unchanged.
# ---------------------------------------------------------------------------
class _Message:
    __slots__ = ("content", "role")

    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"


# ---------------------------------------------------------------------------
# VLLMLLM — drop-in replacement for dextersql.core.llm.LLM
# ---------------------------------------------------------------------------

class VLLMLLM:
    """
    vLLM-backed LLM that matches the ask() interface of dextersql.core.llm.LLM.

    The class-level engine cache means the 120B model is loaded only once
    per process even if multiple pipeline stages create new VLLMLLM instances
    from the same config.
    """

    _engine_cache: Dict[Tuple, Any] = {}
    _cache_lock = threading.Lock()
    # vLLM offline engine is NOT thread-safe — concurrent chat() calls from
    # the pipeline's ThreadPoolExecutor would crash.  This lock serializes
    # all inference calls.  Performance is unaffected: vLLM batches internally.
    _inference_lock = threading.Lock()

    def __init__(self, llm_config: "LLMConfig") -> None:
        self._config = llm_config
        self._engine = self._get_or_create_engine()
        logger.debug(
            f"VLLMLLM ready: model={llm_config.model}, "
            f"tp={llm_config.vllm_tensor_parallel_size}, "
            f"quant={llm_config.vllm_quantization}"
        )

    # ------------------------------------------------------------------
    # Public property (mirrors LLM.llm_config)
    # ------------------------------------------------------------------

    @property
    def llm_config(self) -> "LLMConfig":
        return self._config

    # ------------------------------------------------------------------
    # Engine management
    # ------------------------------------------------------------------

    def _engine_cache_key(self) -> Tuple:
        cfg = self._config
        tp = self._resolve_tensor_parallel_size(cfg.vllm_tensor_parallel_size)
        return (cfg.model, tp, cfg.vllm_quantization, cfg.vllm_gpu_memory_utilization,
                cfg.vllm_max_model_len)

    @staticmethod
    def _resolve_tensor_parallel_size(tp: int) -> int:
        if tp >= 1:
            return tp
        try:
            import torch
            return max(1, torch.cuda.device_count())
        except Exception:
            return 1

    def _get_or_create_engine(self) -> Any:
        key = self._engine_cache_key()
        if key in VLLMLLM._engine_cache:
            return VLLMLLM._engine_cache[key]
        with VLLMLLM._cache_lock:
            if key in VLLMLLM._engine_cache:
                return VLLMLLM._engine_cache[key]
            engine = self._load_engine()
            VLLMLLM._engine_cache[key] = engine
            return engine

    def _load_engine(self) -> Any:
        from vllm import LLM as VLLMEngine

        cfg = self._config
        tp = self._resolve_tensor_parallel_size(cfg.vllm_tensor_parallel_size)

        logger.info(
            f"[vllm] loading {cfg.model} | tp={tp} | "
            f"quant={cfg.vllm_quantization} | "
            f"gpu_mem_util={cfg.vllm_gpu_memory_utilization}"
        )

        kwargs: Dict[str, Any] = dict(
            model=cfg.model,
            tensor_parallel_size=tp,
            dtype="auto",
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,  # skip CUDA graph capture — hangs with mxfp4+tp=2
        )
        if cfg.vllm_quantization:
            kwargs["quantization"] = cfg.vllm_quantization
        if cfg.vllm_max_model_len is not None:
            kwargs["max_model_len"] = cfg.vllm_max_model_len

        return VLLMEngine(**kwargs)

    # ------------------------------------------------------------------
    # Core ask() — matches LLM.ask() signature exactly
    # ------------------------------------------------------------------

    def ask(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[Dict[str, str]] = None,
        timeout: int = 300,
        **kwargs,
    ) -> Tuple[List[_Message], Dict[str, int]]:
        """
        Generate n completions for the given messages.

        Returns (List[_Message], token_usage_dict) exactly like LLM.ask().
        Token counts are estimated from vLLM output metadata.
        """
        from vllm import SamplingParams

        if system_message:
            messages = [system_message] + messages

        n = kwargs.pop("n", 1)
        temperature = float(kwargs.pop("temperature", self._config.temperature) or 0.0)
        max_tokens  = int(kwargs.pop("max_tokens", self._config.max_tokens))

        sampling = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=float(kwargs.pop("top_p", 1.0)),
            max_tokens=max_tokens,
        )

        # No marker injection — pipeline prompts already use <reasoning> +
        # <result> XML tags; downstream parsers rely on <result> being present.
        # llm.chat() applies the model's chat template automatically (vLLM >= 0.5).
        #
        # Lock required: pipeline stages use ThreadPoolExecutor; vLLM offline
        # engine is single-threaded and will crash under concurrent access.
        with VLLMLLM._inference_lock:
            outputs = self._engine.chat(messages, sampling_params=sampling)

        result_messages: List[_Message] = []
        prompt_tokens = 0
        completion_tokens = 0

        for req_output in outputs:
            # Collect prompt token count once per request
            prompt_tokens += len(req_output.prompt_token_ids) if req_output.prompt_token_ids else 0
            for completion in req_output.outputs:
                content = (completion.text or "").strip()
                result_messages.append(_Message(content))
                completion_tokens += len(completion.token_ids) if completion.token_ids else 0

        token_usage = {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        }

        if not result_messages:
            logger.warning("[vllm] ask() returned no completions")

        return result_messages, token_usage
