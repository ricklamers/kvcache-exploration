"""CLI memory report — parity check for what the UI will display.

Run: ``uv run python scripts/memory_report.py``
"""

from __future__ import annotations

import torch

from kvcache_explored.generate import generate_with_cache
from kvcache_explored.kvcache import KVCache
from kvcache_explored.memory import MemoryProbe, estimate_kv_cache_bytes
from kvcache_explored.weights import load_qwen3


def _fmt_mb(n: int) -> str:
    return f"{n / (1024**2):7.1f} MB"


def main() -> None:
    device = torch.device("mps")
    dtype = torch.bfloat16

    print(f"device={device}  dtype={dtype}")
    model, _ = load_qwen3(device=device, dtype=dtype)
    cfg = model.cfg

    probe = MemoryProbe(model)
    print()
    print("--- static parameter breakdown ---")
    for k, v in probe.sample().params_breakdown.items():
        print(f"  {k:12s} {_fmt_mb(v)}")

    print()
    print("--- KV cache budgets (hypothetical) ---")
    for L in (1_024, 4_096, 16_384, 32_768):
        b = estimate_kv_cache_bytes(cfg, L, dtype)
        print(f"  max_seq_len={L:>6d}  kv_cache={_fmt_mb(b)}")

    print()
    print("--- live: growing a 2k-token cache with a short generation ---")
    cache = KVCache(
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_seq_len=2048,
        dtype=dtype,
        device=device,
    )
    probe.set_cache(cache)
    prompt = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]], device=device, dtype=torch.long)
    s = probe.sample()
    print(
        f"  [before]  current={_fmt_mb(s.current_allocated_bytes)}  "
        f"driver={_fmt_mb(s.driver_bytes)}  "
        f"kv_used={_fmt_mb(s.kv_cache_bytes_used)} / {_fmt_mb(s.kv_cache_bytes_allocated)}"
    )
    for i, ev in enumerate(generate_with_cache(model, prompt, 16, cache)):
        if i % 4 == 0:
            s = probe.sample()
            print(
                f"  [step {i:3d}] current={_fmt_mb(s.current_allocated_bytes)}  "
                f"driver={_fmt_mb(s.driver_bytes)}  "
                f"kv_used={_fmt_mb(s.kv_cache_bytes_used)}  "
                f"activations={_fmt_mb(s.activations_bytes)}  "
                f"(step_ms={ev.step_ms:.1f})"
            )


if __name__ == "__main__":
    main()
