"""CLI demo of the quadratic vs linear per-token decode cost.

Runs the same prompt twice — once with KV cache on, once with it off — and
prints per-token latency so the shape of the curve is visible even without
opening the UI.

Run: ``uv run python scripts/demo_quadratic.py``
"""

from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer

from kvcache_explored.generate import generate_with_cache, generate_without_cache
from kvcache_explored.kvcache import KVCache
from kvcache_explored.weights import HF_REPO, load_qwen3

PROMPT = (
    "You are a careful, thorough writer. Explain, step by step, how a modern "
    "transformer large language model uses its KV cache to speed up "
    "autoregressive decoding. Cover attention, memory layout, and why "
    "skipping the cache re-does quadratic work per step. "
)
MAX_NEW = 64


def main() -> None:
    device = torch.device("mps")
    dtype = torch.bfloat16
    model, repo = load_qwen3(device=device, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(repo)

    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    print(f"prompt length: {ids.shape[1]} tokens; generating {MAX_NEW} tokens")

    # --- cache ON --- #
    cache = KVCache(
        num_layers=model.cfg.num_layers,
        num_kv_heads=model.cfg.num_kv_heads,
        head_dim=model.cfg.head_dim,
        max_seq_len=ids.shape[1] + MAX_NEW + 4,
        dtype=dtype,
        device=device,
    )
    t0 = time.perf_counter()
    on_steps = list(generate_with_cache(model, ids, MAX_NEW, cache))
    on_total = (time.perf_counter() - t0) * 1000.0

    # --- cache OFF --- #
    t0 = time.perf_counter()
    off_steps = list(generate_without_cache(model, ids, MAX_NEW))
    off_total = (time.perf_counter() - t0) * 1000.0

    print()
    print(f"{'step':>4} | {'cache ON (ms)':>14} | {'cache OFF (ms)':>16} | ratio")
    print("-" * 60)
    for i, (on, off) in enumerate(zip(on_steps, off_steps)):
        if i % 4 == 0 or i == len(on_steps) - 1:
            ratio = off.step_ms / max(1e-6, on.step_ms)
            print(f"{i:>4} | {on.step_ms:>14.1f} | {off.step_ms:>16.1f} | {ratio:5.2f}x")
    print()
    print(f"total: cache ON = {on_total:7.0f} ms  ({MAX_NEW / (on_total/1000):.1f} tok/s)")
    print(f"       cache OFF = {off_total:7.0f} ms  ({MAX_NEW / (off_total/1000):.1f} tok/s)")
    print(f"       speedup with cache: {off_total / on_total:.2f}x")


if __name__ == "__main__":
    main()
