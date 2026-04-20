"""Inference engine — wraps the model + tokenizer + current cache.

Single-user, single-GPU. The FastAPI server holds one of these and
serializes requests behind an asyncio lock.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from kvcache_explored.chat_template import ChatMessage, render as render_chat
from kvcache_explored.kvcache import KVCache
from kvcache_explored.memory import MemoryProbe
from kvcache_explored.model import Qwen3ForCausalLM
from kvcache_explored.sampling import SamplingParams, greedy, sample
from kvcache_explored.weights import load_qwen3


@dataclass
class GenParams:
    use_cache: bool
    enable_thinking: bool
    max_new_tokens: int
    max_seq_len: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    seed: int | None


@dataclass
class Step:
    token_id: int
    text: str
    step_ms: float
    step_index: int
    seq_len: int
    kv_length: int


class InferenceEngine:
    """Thin orchestration layer — no model definition lives here."""

    def __init__(self, device: str = "mps", dtype: torch.dtype = torch.bfloat16) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.model: Qwen3ForCausalLM
        self.tokenizer = None
        self.probe: MemoryProbe
        self.cache: KVCache | None = None
        self._lock = asyncio.Lock()
        self._cancelled = False
        # Set by _run_loop when it returns early; "length" means we hit max_new_tokens.
        self.last_finish_reason: str = "length"

    # ---- lifecycle -------------------------------------------------------- #

    def load(self) -> None:
        self.model, repo_dir = load_qwen3(device=self.device, dtype=self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(str(repo_dir))
        self.probe = MemoryProbe(self.model)

    # ---- one request ------------------------------------------------------ #

    async def generate(
        self, messages: list[ChatMessage], params: GenParams
    ) -> AsyncIterator[Step]:
        """Stream Step events. Holds the global lock for the whole run."""
        async with self._lock:
            self._cancelled = False
            assert self.tokenizer is not None

            prompt_text = render_chat(
                messages,
                enable_thinking=params.enable_thinking,
                add_generation_prompt=True,
            )
            input_ids = self.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            prompt_len = input_ids.shape[1]

            if params.seed is not None:
                torch.manual_seed(params.seed)

            # Prepare cache if requested.
            if params.use_cache:
                # Grow cache to the requested max_seq_len if it doesn't exist or is too small.
                need = min(params.max_seq_len, self.model.cfg.max_position_embeddings)
                if self.cache is None or self.cache.max_seq_len != need:
                    self.cache = KVCache(
                        num_layers=self.model.cfg.num_layers,
                        num_kv_heads=self.model.cfg.num_kv_heads,
                        head_dim=self.model.cfg.head_dim,
                        max_seq_len=need,
                        dtype=self.dtype,
                        device=self.device,
                    )
                else:
                    self.cache.reset()
                self.probe.set_cache(self.cache)
            else:
                self.cache = None
                self.probe.set_cache(None)

            sampling = SamplingParams(
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                min_p=params.min_p,
            )

            self.last_finish_reason = "length"
            try:
                async for step in self._run_loop(input_ids, params, sampling, prompt_len):
                    yield step
                    if self._cancelled:
                        self.last_finish_reason = "cancelled"
                        return
                # The inner loop may also return *without* yielding first
                # (it checks ``_cancelled`` at the top of each iteration
                # so mid-token cancellation exits cleanly). Catch that
                # case here — otherwise finish_reason stays at "length".
                if self._cancelled:
                    self.last_finish_reason = "cancelled"
            finally:
                # Release MPS pool memory now so the telemetry line drops
                # back to the baseline (params + kv_cache) instead of
                # holding the peak cache-off activation footprint until the
                # next request reuses it.
                if self.device.type == "mps":
                    torch.mps.empty_cache()

    def cancel(self) -> None:
        self._cancelled = True

    async def restart(self) -> None:
        """Drop KV cache and release MPS pool without reloading weights.

        Reloading the full model on MPS takes 10+ seconds; in practice the
        "restart" users want after a run has gone sideways is:
          - cancel any in-flight generation,
          - free the KV cache (~3.5 GB at 32k),
          - flush the MPS allocator pool.
        Weights stay resident; the next request reuses them.

        We acquire the same lock as ``generate`` so we can't collide with
        an in-flight run.
        """
        self._cancelled = True
        async with self._lock:
            self.cache = None
            self.probe.set_cache(None)
            self.last_finish_reason = "length"
            self._cancelled = False
            if self.device.type == "mps":
                torch.mps.empty_cache()

    # ---------------------------------------------------------------------- #
    # Internal: a sampling-aware mirror of the greedy generators. Kept here
    # rather than in generate.py because it needs the tokenizer (to decode
    # each token to text) and cancellation support, both of which are
    # server-level concerns.
    # ---------------------------------------------------------------------- #

    async def _run_loop(
        self,
        prompt_ids: torch.Tensor,
        params: GenParams,
        sampling: SamplingParams,
        prompt_len: int,
    ) -> AsyncIterator[Step]:
        """Inference loop.

        No ``@torch.no_grad()`` decorator: the decorator form exits the
        no-grad context at every ``yield`` (async generators unwind their
        frame on yield), so autograd silently turns back on between tokens
        and the cache-off path leaks the entire decode graph via
        ``torch.cat(seq, new)`` — the old ``seq`` becomes a non-leaf grad
        parent of the new one. Using ``torch.inference_mode()`` as an
        explicit context manager *inside* the function body is the
        suspend-safe form.
        """
        assert self.tokenizer is not None
        device = self.device
        eos_ids = set(self.tokenizer.all_special_ids or [])
        # Qwen uses <|im_end|> to close assistant turns; include it.
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int):
            eos_ids.add(im_end)

        def _pick(logits: torch.Tensor) -> int:
            if params.temperature <= 0.0:
                return greedy(logits)
            return sample(logits, sampling, enable_thinking=params.enable_thinking)

        def _sync_time() -> float:
            if device.type == "mps":
                torch.mps.synchronize()
            return time.perf_counter()

        decoder = _StreamingDecoder(self.tokenizer)
        step_index = 0
        # Every N tokens, flush the MPS allocator pool back to the OS.
        # Trades a small latency blip (~tens of ms) for keeping the
        # driver line flat — important in the cache-off path where peak
        # activation blocks scale with sequence length and the pool
        # otherwise ratchets up to the worst-case footprint.
        EMPTY_CACHE_EVERY = 10

        def _maybe_flush_pool() -> None:
            if device.type == "mps" and step_index > 0 and step_index % EMPTY_CACHE_EVERY == 0:
                torch.mps.empty_cache()

        # Inner helpers do all the GPU work; they return only Python
        # primitives so no MPS-tensor reference survives across the
        # ``yield`` (the generator frame would otherwise pin them and the
        # underlying ``[0, -1]`` view holds the full ``(1, T, V)`` lm_head
        # output, scaling with sequence length).
        def _step_cached(input_ids: torch.Tensor, start_pos: int) -> tuple[int, float]:
            with torch.inference_mode():
                t0 = _sync_time()
                logits = self.model(input_ids, start_pos=start_pos, cache=self.cache)[0, -1]
                tok_id = _pick(logits)
                t1 = _sync_time()
            return tok_id, (t1 - t0) * 1000.0

        def _step_uncached(ids: list[int]) -> tuple[int, float]:
            with torch.inference_mode():
                t0 = _sync_time()
                seq = torch.tensor([ids], device=device, dtype=torch.long)
                logits = self.model(seq, start_pos=0, cache=None)[0, -1]
                tok_id = _pick(logits)
                t1 = _sync_time()
            return tok_id, (t1 - t0) * 1000.0

        if params.use_cache:
            assert self.cache is not None
            next_id, step_ms = _step_cached(prompt_ids, 0)
            yield Step(
                token_id=next_id,
                text=decoder.feed(next_id),
                step_ms=step_ms,
                step_index=step_index,
                seq_len=prompt_len,
                kv_length=self.cache.length,
            )
            step_index += 1
            if next_id in eos_ids:
                self.last_finish_reason = "eos"
                return

            for _ in range(params.max_new_tokens - 1):
                if self._cancelled:
                    return
                if self.cache.length + 1 > self.cache.max_seq_len:
                    self.last_finish_reason = "overflow"
                    return
                ids = torch.tensor([[next_id]], device=device, dtype=torch.long)
                next_id, step_ms = _step_cached(ids, self.cache.length)
                del ids
                yield Step(
                    token_id=next_id,
                    text=decoder.feed(next_id),
                    step_ms=step_ms,
                    step_index=step_index,
                    seq_len=self.cache.length,
                    kv_length=self.cache.length,
                )
                step_index += 1
                if next_id in eos_ids:
                    self.last_finish_reason = "eos"
                    return
                _maybe_flush_pool()
                # Cooperative yield — lets the telemetry task run.
                await asyncio.sleep(0)
        else:
            # Store ids as a plain Python list so we never carry a tensor
            # reference across iterations (also avoids any autograd graph
            # chaining if grad-mode were ever to slip back in).
            ids_list: list[int] = prompt_ids[0].tolist()
            for _ in range(params.max_new_tokens):
                if self._cancelled:
                    return
                next_id, step_ms = _step_uncached(ids_list)
                ids_list.append(next_id)
                yield Step(
                    token_id=next_id,
                    text=decoder.feed(next_id),
                    step_ms=step_ms,
                    step_index=step_index,
                    seq_len=len(ids_list) - 1,
                    kv_length=0,
                )
                step_index += 1
                if next_id in eos_ids:
                    self.last_finish_reason = "eos"
                    return
                _maybe_flush_pool()
                await asyncio.sleep(0)


class _StreamingDecoder:
    """Buffer tokens until the decoder yields complete Unicode characters.

    Qwen's BPE splits multi-byte glyphs (emoji, some CJK) across multiple
    token IDs. Decoding each ID in isolation produces invalid UTF-8
    fragments that render as U+FFFD (the replacement character). Instead,
    we accumulate pending IDs and only emit text once ``tokenizer.decode``
    on the buffered IDs produces a string that doesn't *end* on a partial
    byte sequence (the standard heuristic: decoded text doesn't end with
    U+FFFD).
    """

    def __init__(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        self.tokenizer = tokenizer
        self._pending: list[int] = []

    def feed(self, token_id: int) -> str:
        self._pending.append(token_id)
        text = self.tokenizer.decode(self._pending, skip_special_tokens=False)
        if text.endswith("\ufffd"):
            # Incomplete multi-byte character — wait for the continuation.
            return ""
        self._pending.clear()
        return text
