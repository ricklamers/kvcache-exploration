"""Generation loops — greedy for now (M2). Sampling arrives in M3.

Two modes, both of which stream one token at a time so the UI can see the
per-step cost in real time.

- ``cache=on``:  prefill the prompt once, then decode one token per step with
                 a KVCache. Per-step attention cost is O(prefix_len).
- ``cache=off``: no cache. Each step runs a full forward over the entire
                 sequence so far (prompt + all tokens decoded so far).
                 Per-step cost is O(prefix_len**2). This is the deliberately
                 expensive baseline that makes the quadratic scaling
                 visible in the UI.

Both modes produce identical token IDs for the same prompt (verified in
scripts/verify_against_hf.py).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from kvcache_explored.kvcache import KVCache
from kvcache_explored.model import Qwen3ForCausalLM


@dataclass
class StepEvent:
    token_id: int
    step_ms: float
    seq_len: int  # tokens seen by the model on this step (prefix + new)
    kv_length: int  # tokens stored in the cache (0 when cache is off)


def _timed_step_ms(device: torch.device) -> float:
    # MPS has no events API as nice as CUDA's, but synchronize + wall clock is fine
    # for per-token timing at ~10–100 ms granularity.
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def generate_with_cache(
    model: Qwen3ForCausalLM,
    prompt_ids: Tensor,  # (1, P) on model device
    max_new_tokens: int,
    cache: KVCache,
) -> Iterator[StepEvent]:
    device = prompt_ids.device

    # Prefill: one big forward, caches all prompt tokens.
    t0 = _timed_step_ms(device)
    logits = model(prompt_ids, start_pos=0, cache=cache)
    next_id = int(logits[0, -1].argmax().item())
    t1 = _timed_step_ms(device)
    yield StepEvent(
        token_id=next_id,
        step_ms=(t1 - t0) * 1000.0,
        seq_len=prompt_ids.shape[1],
        kv_length=cache.length,
    )

    # Decode loop: feed one token at a time.
    for _ in range(max_new_tokens - 1):
        t0 = _timed_step_ms(device)
        ids = torch.tensor([[next_id]], device=device, dtype=torch.long)
        logits = model(ids, start_pos=cache.length, cache=cache)
        next_id = int(logits[0, -1].argmax().item())
        t1 = _timed_step_ms(device)
        yield StepEvent(
            token_id=next_id,
            step_ms=(t1 - t0) * 1000.0,
            seq_len=cache.length,  # already updated by append
            kv_length=cache.length,
        )


@torch.no_grad()
def generate_without_cache(
    model: Qwen3ForCausalLM,
    prompt_ids: Tensor,
    max_new_tokens: int,
) -> Iterator[StepEvent]:
    """The quadratic-per-step baseline. Re-runs the full forward each step."""
    device = prompt_ids.device
    seq = prompt_ids

    for _ in range(max_new_tokens):
        t0 = _timed_step_ms(device)
        logits = model(seq, start_pos=0, cache=None)
        next_id = int(logits[0, -1].argmax().item())
        t1 = _timed_step_ms(device)
        seq = torch.cat([seq, torch.tensor([[next_id]], device=device, dtype=torch.long)], dim=1)
        yield StepEvent(
            token_id=next_id,
            step_ms=(t1 - t0) * 1000.0,
            seq_len=seq.shape[1] - 1,  # before appending
            kv_length=0,
        )


def greedy_rollout(
    model: Qwen3ForCausalLM,
    prompt_ids: Tensor,
    steps: int,
    *,
    cache: KVCache | None = None,
) -> list[int]:
    """Convenience wrapper used by the verifier."""
    if cache is not None:
        return [e.token_id for e in generate_with_cache(model, prompt_ids, steps, cache)]
    return [e.token_id for e in generate_without_cache(model, prompt_ids, steps)]
