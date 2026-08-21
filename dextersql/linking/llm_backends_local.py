"""
llm_backends_local.py
---------------------
Reusable local LLM backends (Qwen via HF generate; GPT-OSS via HF pipeline).

Update (Jan 2026):
- GPT-OSS may emit "analysis"/reasoning in the same stream.
- We extract and return ONLY the final answer by default using explicit
  begin/end markers (multi-line safe).
- Marker instructions are injected into BOTH the system message and the
  LAST user message (more reliable).
- IMPORTANT: backends are now CACHED so a notebook rerun will NOT reload
  model weights and kill your kernel.

Update (May 2026):
- Added VLLMOSSBackend for GPT-OSS 120B on HPC via vLLM.
  Uses FP4 quantization + tensor parallelism across all visible GPUs.
  Significantly faster than HF pipeline for large-batch inference.
  Entry point: make_backend("vllm") or make_backend("gptoss120b_vllm")
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any

import os
import re

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# -----------------------------
# Marker-based final extraction
# -----------------------------
FINAL_BEGIN = "###FINAL_BEGIN###"
FINAL_END = "###FINAL_END###"


class LLMBackend:
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> str:
        raise NotImplementedError

    def generate_with_meta(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs
    ) -> Dict[str, Any]:
        txt = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": txt, "raw": txt, "thoughts": ""}


class HFTransformersBackend(LLMBackend):
    """
    Chat-style HF backend using AutoTokenizer/AutoModelForCausalLM and the model's chat_template.
    Good for Qwen2.5-* Instruct.
    """
    def __init__(self,
                 model_id: str = "Qwen/Qwen2.5-7B-Instruct",
                 device_map: str = "auto",
                 dtype: Optional[str] = None,
                 trust_remote_code: bool = True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

        torch_dtype = None
        if dtype:
            dmap = {
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            torch_dtype = dmap.get(dtype.lower())

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> str:
        import torch

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt")

        # device placement
        if hasattr(self.model, "device"):
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

        # Safe defaults
        use_args = {
            "do_sample": gen_kwargs.pop("do_sample", False),
            "temperature": gen_kwargs.pop("temperature", None),
            "top_p": gen_kwargs.pop("top_p", None),
            "num_beams": gen_kwargs.pop("num_beams", None),
        }
        use_args.update(gen_kwargs)
        use_args = {k: v for k, v in use_args.items() if v is not None}

        with torch.inference_mode():
            out_ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, **use_args)

        gen_ids = out_ids[0][enc["input_ids"].shape[1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        return (text or "").strip()


def _pull_raw_from_hf_chat_output(out_obj) -> str:
    if not out_obj:
        return ""
    if isinstance(out_obj, list) and out_obj:
        out_obj = out_obj[0]
    if isinstance(out_obj, dict):
        for k in ("generated_text", "text"):
            if k in out_obj and isinstance(out_obj[k], str):
                return out_obj[k]
        if "generated_text" in out_obj and isinstance(out_obj["generated_text"], list):
            msgs = out_obj["generated_text"]
            if msgs and isinstance(msgs[-1], dict) and "content" in msgs[-1]:
                return str(msgs[-1]["content"])
    return str(out_obj)


def _extract_final_block(raw: str) -> str:
    raw = raw or ""
    i = raw.rfind(FINAL_BEGIN)
    if i == -1:
        return ""
    j = raw.rfind(FINAL_END)
    if j == -1 or j < i:
        return ""
    return (raw[i + len(FINAL_BEGIN): j] or "").strip()


def _fallback_extract_answer(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    quotes = re.findall(r'"([^"]{1,2000})"', raw)
    if quotes:
        return quotes[-1].strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    if paras:
        tail = re.sub(r"^\s*analysis\s*", "", paras[-1], flags=re.IGNORECASE)
        return tail.strip()
    return raw


def _extract_final_and_thoughts(raw: str) -> Tuple[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return "", ""

    final = _extract_final_block(raw)
    if final:
        i = raw.rfind(FINAL_BEGIN)
        j = raw.rfind(FINAL_END)
        before = raw[:i].strip()
        after = raw[j + len(FINAL_END):].strip()
        thoughts = (before + ("\n" if before and after else "") + after).strip()
        return final, thoughts

    # fallback
    final_guess = _fallback_extract_answer(raw)
    return final_guess, raw


def _inject_marker_instruction(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    extra = (
        "\n\n"
        "IMPORTANT OUTPUT FORMAT (MANDATORY):\n"
        f"1) Put your FINAL answer between these exact markers:\n"
        f"{FINAL_BEGIN}\n"
        f"<final answer>\n"
        f"{FINAL_END}\n"
        "2) Do NOT write anything inside the markers except the final answer.\n"
        "3) You may write reasoning OUTSIDE the markers.\n"
    )

    out: List[Dict[str, str]] = []
    sys_injected = False

    for msg in messages:
        if msg.get("role") == "system" and not sys_injected:
            out.append({"role": "system", "content": (msg.get("content") or "") + extra})
            sys_injected = True
        else:
            out.append(msg)

    if not sys_injected:
        out.insert(0, {"role": "system", "content": extra.strip()})

    last_user_idx = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is not None:
        out[last_user_idx] = {"role": "user", "content": (out[last_user_idx].get("content") or "") + extra}
    else:
        out.append({"role": "user", "content": extra.strip()})

    return out


class OSSHFPBackend(LLMBackend):
    """
    OSS backend using transformers.pipeline text-generation. Works for openai/gpt-oss-20b.
    Returns final answer by default (marker extraction).
    """
    def __init__(self,
                 model_id: str = "openai/gpt-oss-20b",
                 device_map: str = "auto",
                 dtype: Optional[str] = None,
                 trust_remote_code: bool = True,
                 inject_final_markers: bool = True,
                 max_memory: Optional[Dict] = None):
        import torch
        from transformers import pipeline
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.inject_final_markers = inject_final_markers

        torch_dtype = None
        if dtype:
            dmap = {
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            torch_dtype = dmap.get(dtype.lower())

        if max_memory is not None:
            # Load model + tokenizer manually so max_memory is only used at
            # load time and never forwarded to generate() calls by pipeline().
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
            )
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        else:
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )

        self._default_pad_token_id = None
        try:
            self._default_pad_token_id = getattr(self.pipe, "tokenizer", None).eos_token_id
        except Exception:
            self._default_pad_token_id = None

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> str:
        meta = self.generate_with_meta(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return (meta.get("text") or "").strip()

    def generate_with_meta(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs
    ) -> Dict[str, Any]:
        do_sample = gen_kwargs.pop("do_sample", False)
        temperature = gen_kwargs.pop("temperature", None)

        pad_token_id = gen_kwargs.pop("pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self._default_pad_token_id

        use_messages = _inject_marker_instruction(messages) if self.inject_final_markers else messages

        pipe_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=pad_token_id,
        )
        pipe_kwargs.update(gen_kwargs)

        out = self.pipe(use_messages, **pipe_kwargs)
        raw = _pull_raw_from_hf_chat_output(out).strip()

        # Free reserved-but-unallocated GPU memory between calls.
        # Critical for large models (120B) with variable-length prompts.
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

        final, thoughts = _extract_final_and_thoughts(raw)
        return {"text": final, "raw": raw, "thoughts": thoughts}


# -----------------------------
# vLLM Backend (HPC, 120B, FP4)
# -----------------------------

class VLLMOSSBackend(LLMBackend):
    """
    vLLM backend for GPT-OSS 120B on HPC servers.

    Key differences vs OSSHFPBackend:
    - Uses vLLM's optimised inference engine (PagedAttention, continuous batching)
    - FP4 quantization by default — fits 120B comfortably on 4×A100 80GiB
    - Tensor parallelism auto-detected from visible GPUs (CUDA_VISIBLE_DEVICES)
    - Roughly 3-5× faster throughput than the HF pipeline at batch=1

    Usage::
        backend = make_backend("vllm")                      # fp4, all GPUs
        backend = make_backend("vllm", tensor_parallel_size=4)
        backend = make_backend("vllm", quantization="fp8")  # override quant
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-oss-120b",
        tensor_parallel_size: int = -1,       # -1 = auto: all visible GPUs
        quantization: str = "mxfp4",           # mxfp4 | fp8 | awq | gptq | None
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = 32768,  # cap at 32k; full 131k needs more KV cache RAM
        trust_remote_code: bool = True,
        inject_final_markers: bool = True,
    ):
        from vllm import LLM

        self.model_id = model_id
        self.inject_final_markers = inject_final_markers

        # Auto-detect GPU count from CUDA_VISIBLE_DEVICES / torch
        if tensor_parallel_size < 1:
            try:
                import torch as _torch
                tensor_parallel_size = max(1, _torch.cuda.device_count())
            except Exception:
                tensor_parallel_size = 1

        print(
            f"  [vllm-oss120b] loading {model_id}\n"
            f"    tensor_parallel={tensor_parallel_size} | quant={quantization} "
            f"| gpu_mem_util={gpu_memory_utilization}"
        )

        llm_kwargs: Dict[str, Any] = dict(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)

    def generate(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs
    ) -> str:
        meta = self.generate_with_meta(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return (meta.get("text") or "").strip()

    def generate_with_meta(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs
    ) -> Dict[str, Any]:
        from vllm import SamplingParams

        use_messages = (
            _inject_marker_instruction(messages) if self.inject_final_markers else messages
        )

        temperature = float(gen_kwargs.pop("temperature", 0.0) or 0.0)
        top_p       = float(gen_kwargs.pop("top_p", 1.0) or 1.0)

        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        # llm.chat() applies the model's chat template automatically (vLLM ≥ 0.5)
        outputs = self.llm.chat(use_messages, sampling_params=sampling)
        raw = (outputs[0].outputs[0].text or "").strip()

        final, thoughts = _extract_final_and_thoughts(raw)
        return {"text": final, "raw": raw, "thoughts": thoughts}


# -----------------------------
# OpenAI API Backend
# -----------------------------

class OpenAIBackend(LLMBackend):
    """
    OpenAI API backend using GPT-5.2 (or any OpenAI model).
    Drop-in replacement for Qwen/GPT-OSS backends.
    """
    def __init__(self, model_id: str = "gpt-5.2", api_key: Optional[str] = None):
        import openai, os
        self.model_id = model_id
        self.kind = "openai"
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._openai = openai

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> str:
        token_param = "max_completion_tokens" if self.model_id in ["gpt-5-mini", "gpt-5.2"] else "max_tokens"
        response = self._openai.ChatCompletion.create(
            model=self.model_id,
            messages=messages,
            temperature=gen_kwargs.get("temperature", 0),
            **{token_param: max_new_tokens}
        )
        return (response["choices"][0]["message"]["content"] or "").strip()

    def generate_with_meta(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        text = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": text, "raw": text, "thoughts": ""}


# ---------------------------------------------------------------------------
# vLLM HTTP server backend (OpenAI-compatible, no in-process model load)
# ---------------------------------------------------------------------------

class VLLMServerBackend(LLMBackend):
    """
    Talk to an ALREADY-RUNNING vLLM server (OpenAI-compatible API) over HTTP.

    Unlike VLLMOSSBackend (which loads a fresh 120B copy in-process via
    `from vllm import LLM`), this backend issues chat-completion requests to a
    live server, so it runs on a plain CPU job and needs no GPUs of its own.

    The server URL is taken from (in priority order):
      1. the `base_url` argument
      2. the VLLM_SERVER_URL env var (may be "host:port", "http://host:port",
         or "http://host:port/v1" — all normalised to ".../v1")
      3. the handshake file at VLLM_SERVER_FILE (default
         /tmp/vllm_server.txt), whose first non-empty line is
         the "host:port" the server published.

    Reasoning-model handling mirrors the marker scheme used by the OSS/vLLM
    in-process backends: a FINAL_BEGIN/FINAL_END marker instruction is injected
    and the final answer is extracted from the visible content. When the visible
    content is empty (the model spent its budget on chain-of-thought), we fall
    back to `reasoning_content` if the server exposes it.
    """

    DEFAULT_SERVER_FILE = "/tmp/vllm_server.txt"

    def __init__(
        self,
        model_id: str = "openai/gpt-oss-120b",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        inject_final_markers: bool = True,
        timeout: float = 120.0,
    ):
        import openai

        self.model_id = model_id
        self.kind = "vllm_server"
        self.inject_final_markers = inject_final_markers

        resolved = self._resolve_base_url(base_url)
        self.base_url = resolved
        self._api_key = api_key or os.environ.get("VLLM_API_KEY", "dummy")
        self._timeout = timeout
        self.client = self._make_client()

    # -- client factory ------------------------------------------------------
    def _make_client(self):
        import httpx
        import openai as _openai
        # Disable HTTP keep-alive / connection pooling. The vLLM server closes
        # idle keep-alive sockets (server-side timeout); a pooled socket left in
        # CLOSE-WAIT, when reused, makes httpx spin at ~100% CPU in the low-level
        # socket layer WITHOUT ever raising APIConnectionError — so retry logic
        # never fires and the job hangs. With max_keepalive_connections=0 every
        # request opens a fresh connection and closes it, so a stale socket can
        # never be handed back to us.
        http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=20),
            timeout=self._timeout,
        )
        return _openai.OpenAI(
            base_url=self.base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            # max_retries=0: don't spin on dead connections — if the server
            # drops the TCP connection we want an immediate exception, not an
            # infinite retry loop burning 100% CPU.
            max_retries=0,
            http_client=http_client,
        )

    # -- URL resolution ------------------------------------------------------
    @classmethod
    def _normalise(cls, raw: str) -> str:
        raw = (raw or "").strip().rstrip("/")
        if not raw:
            return raw
        if not raw.startswith("http://") and not raw.startswith("https://"):
            raw = "http://" + raw
        if not raw.endswith("/v1"):
            raw = raw + "/v1"
        return raw

    @classmethod
    def _resolve_base_url(cls, base_url: Optional[str]) -> str:
        if base_url:
            return cls._normalise(base_url)
        env = os.environ.get("VLLM_SERVER_URL")
        if env:
            return cls._normalise(env)
        server_file = os.environ.get("VLLM_SERVER_FILE", cls.DEFAULT_SERVER_FILE)
        try:
            with open(server_file) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        return cls._normalise(line)
        except FileNotFoundError:
            pass
        raise EnvironmentError(
            "Could not determine vLLM server URL. Provide base_url, set "
            "VLLM_SERVER_URL=host:port, or ensure the handshake file exists "
            f"at {server_file}."
        )

    # -- generation ----------------------------------------------------------
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> str:
        meta = self.generate_with_meta(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return (meta.get("text") or "").strip()

    def generate_with_meta(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs
    ) -> Dict[str, Any]:
        import openai as _openai

        temperature = float(gen_kwargs.pop("temperature", 0.0) or 0.0)
        top_p = float(gen_kwargs.pop("top_p", 1.0) or 1.0)
        gen_kwargs.pop("do_sample", None)  # not an OpenAI-API param

        use_messages = (
            _inject_marker_instruction(messages) if self.inject_final_markers else messages
        )

        # Retry once on connection errors, re-creating the client each time so
        # a stale CLOSE-WAIT socket is never reused (avoids infinite CPU spin).
        last_exc = None
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=use_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                )
                break
            except (_openai.APIConnectionError, _openai.APITimeoutError) as exc:
                last_exc = exc
                print(f"  [VLLMServer] connection error (attempt {attempt+1}/2): {exc} — refreshing client")
                self.client = self._make_client()
        else:
            raise RuntimeError(f"VLLMServerBackend: failed after 2 attempts: {last_exc}")

        msg = resp.choices[0].message
        raw = (msg.content or "").strip()
        # Reasoning models may leave content empty and stash CoT in
        # reasoning_content — fall back to it so we never silently return "".
        if not raw:
            raw = (getattr(msg, "reasoning_content", None) or "").strip()

        final, thoughts = _extract_final_and_thoughts(raw)
        return {"text": final, "raw": raw, "thoughts": thoughts}


# -----------------------------
# Backend cache (CRITICAL)
# -----------------------------
_BACKEND_CACHE: Dict[Tuple, LLMBackend] = {}


def clear_backend_cache() -> None:
    """
    Clear cached backends. Use when switching models or if GPU memory is stuck.
    After calling, restart kernel is still the cleanest option.
    """
    _BACKEND_CACHE.clear()


def make_backend(kind: str,
                 model_id: Optional[str] = None,
                 device_map: str = "auto",
                 dtype: Optional[str] = None,
                 trust_remote_code: bool = True,
                 inject_final_markers: bool = True,
                 cache: bool = True,
                 # vLLM-specific kwargs (ignored by non-vLLM backends)
                 tensor_parallel_size: int = -1,
                 quantization: str = "mxfp4",
                 gpu_memory_utilization: float = 0.90,
                 max_model_len: Optional[int] = 32768,
                 ) -> LLMBackend:
    """
    kind:
      - "qwen"              -> HFTransformersBackend (HF generate)
      - "gptoss"            -> OSSHFPBackend (HF pipeline, 20B)
      - "gptoss120b"        -> OSSHFPBackend (HF pipeline, 120B, multi-GPU)
      - "vllm"              -> VLLMOSSBackend (vLLM, 120B, FP4, HPC, in-process)
      - "gptoss120b_vllm"   -> same as "vllm"
      - "vllm_server"       -> VLLMServerBackend (HTTP to a live vLLM server; CPU-only)
      - "openai"            -> OpenAIBackend (API)
      - "openrouter"        -> OpenRouterRulesBackend (API)

    inject_final_markers (OSS/vLLM backends only):
      True  (default) - inject ###FINAL_BEGIN### / ###FINAL_END### markers and
                        strip reasoning from output. Use this for SQL generation
                        where the model may emit a chain-of-thought preamble.
      False           - return raw model output directly, no marker injection.
                        Use this for short-profile generation where the output is
                        a single sentence and the fallback extractor would grab
                        reasoning noise instead.

    vLLM-specific kwargs (only used when kind="vllm" / "gptoss120b_vllm"):
      tensor_parallel_size  : number of GPUs for tensor parallelism; -1 = auto
      quantization          : "fp4" (default) | "fp8" | "awq" | "gptq" | None
      gpu_memory_utilization: fraction of GPU VRAM vLLM may use (default 0.90)
      max_model_len         : max sequence length override (None = model default)

    NOTE: cache=True means: repeated calls in the same kernel reuse the model instance.
    NOTE: inject_final_markers=False creates a SEPARATE cache entry from the default
          True instance, so both can coexist in the same process without conflict.
    """
    k = (kind or "").lower().strip()

    if k in ("qwen", "hf"):
        mid = model_id or "Qwen/Qwen2.5-7B-Instruct"
        # strongly recommended default on A100 if user didn't set dtype
        if dtype is None and device_map != "cpu":
            dtype = "bfloat16"
        key = (k, mid, str(device_map), str(dtype), bool(trust_remote_code))
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = HFTransformersBackend(mid, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("gptoss", "oss"):
        mid = model_id or "openai/gpt-oss-20b"
        if dtype is None and device_map != "cpu":
            dtype = "bfloat16"
        # include inject_final_markers in cache key so profile backend (False)
        # and SQL backend (True) are stored as separate entries
        key = (k, mid, str(device_map), str(dtype), bool(trust_remote_code), bool(inject_final_markers))
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = OSSHFPBackend(mid, device_map=device_map, dtype=dtype,
                                trust_remote_code=trust_remote_code,
                                inject_final_markers=inject_final_markers)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("gptoss120b", "oss120b"):
        mid = model_id or "openai/gpt-oss-120b"
        if dtype is None and device_map != "cpu":
            dtype = "bfloat16"
        key = (k, mid, str(device_map), str(dtype), bool(trust_remote_code), bool(inject_final_markers))
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        # Distribute 120B model evenly across all visible GPUs.
        # Cap each GPU at 65GiB — leaves ~16GiB headroom per GPU for KV cache
        # and activations from long prompts (full_long combo can be very long).
        # Falls back to CPU RAM for any overflow.
        import torch as _torch
        n_gpus = _torch.cuda.device_count()
        _max_memory: Dict = {i: "65GiB" for i in range(n_gpus)}
        _max_memory["cpu"] = "50GiB"
        print(f"  [gptoss120b] distributing across {n_gpus} GPU(s), "
              f"65GiB cap each + 50GiB CPU overflow")
        backend = OSSHFPBackend(mid, device_map=device_map, dtype=dtype,
                                trust_remote_code=trust_remote_code,
                                inject_final_markers=inject_final_markers,
                                max_memory=_max_memory)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("vllm", "gptoss120b_vllm", "vllm_oss120b"):
        mid = model_id or "openai/gpt-oss-120b"
        # Auto tensor-parallel if caller didn't override
        tp = tensor_parallel_size
        if tp < 1:
            try:
                import torch as _torch
                tp = max(1, _torch.cuda.device_count())
            except Exception:
                tp = 1
        key = (k, mid, tp, quantization, bool(inject_final_markers))
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = VLLMOSSBackend(
            model_id=mid,
            tensor_parallel_size=tp,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            inject_final_markers=inject_final_markers,
        )
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("vllm_server", "vllm-server", "vllm_http", "server"):
        mid = model_id or "openai/gpt-oss-120b"
        # base_url resolved inside the backend (arg/env/handshake file).
        base_url = os.environ.get("VLLM_SERVER_URL") or ""
        key = (k, mid, base_url, bool(inject_final_markers), False)
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = VLLMServerBackend(
            model_id=mid,
            inject_final_markers=inject_final_markers,
        )
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("openai", "gpt"):
        mid = model_id or "gpt-5.2"
        key = (k, mid, "", "", False)
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = OpenAIBackend(mid)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("openrouter",):
        mid = model_id or "openai/gpt-oss-120b"
        key = (k, mid, "", "", False)
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        # Reuse the OpenRouter backend from the rules subpackage — it accepts
        # (messages, max_new_tokens, temperature, **kwargs) which matches the
        # generate_candidates.py / rule_corrector_l2.py calling convention.
        from train_pipeline.llm_backends_rules import OpenRouterRulesBackend
        backend = OpenRouterRulesBackend(model_id=mid)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    raise ValueError(
        f"Unknown backend kind: {kind!r}. "
        f"Use 'qwen', 'gptoss', 'gptoss120b', 'vllm', 'vllm_server', "
        f"'openai', or 'openrouter'."
    )
