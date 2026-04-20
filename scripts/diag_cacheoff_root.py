"""Pinpoint the +0.3 MB/step live-memory growth in cache-off decode.

We run two variants of the loop and compare:

A. baseline (current engine code)
B. explicit ``del seq; del logits; gc.collect()`` between steps

If B is flat and A grows, the leak is held by Python references to old
tensors. If B also grows, the growth is somewhere internal to torch/MPS.
"""

from __future__ import annotations

import gc

import torch
from transformers import AutoTokenizer

from kvcache_explored.weights import HF_REPO, load_qwen3

PROMPT = "Write a long detailed essay about prime numbers."
NEW_TOKENS = 40
MB = 1024 * 1024


def run(label: str, model, ids_list_init: list[int], explicit_free: bool, device, dtype) -> None:
    ids_list = list(ids_list_init)
    torch.mps.empty_cache()
    base = torch.mps.current_allocated_memory()
    print(f"\n=== {label} ===  baseline live={base/MB:.1f} MB")
    samples: list[int] = []
    for step in range(NEW_TOKENS):
        with torch.inference_mode():
            seq = torch.tensor([ids_list], device=device, dtype=torch.long)
            logits = model(seq, start_pos=0, cache=None)[0, -1]
            next_id = int(logits.argmax().item())
        ids_list.append(next_id)

        if explicit_free:
            del seq
            del logits
            gc.collect()

        torch.mps.synchronize()
        live = torch.mps.current_allocated_memory()
        samples.append(live)
        if step % 4 == 0 or step == NEW_TOKENS - 1:
            print(f"  step {step:3d}  seqlen={len(ids_list):4d}  live={live/MB:7.1f} MB  Δ={(live-base)/MB:+.1f}")

    final = samples[-1]
    print(f"  total Δlive over {NEW_TOKENS} steps: {(final-base)/MB:+.1f} MB  ({(final-base)/NEW_TOKENS/1024:.0f} KB/step)")


def run_inner_fn(label: str, model, ids_list_init: list[int], device, dtype) -> None:
    """Variant C — mirror the engine fix: do the GPU work inside an inner
    function that returns only Python primitives. The view-into-(1,T,V)
    stays scoped to the function and is freed when it returns."""
    ids_list = list(ids_list_init)
    torch.mps.empty_cache()
    base = torch.mps.current_allocated_memory()
    print(f"\n=== {label} ===  baseline live={base/MB:.1f} MB")

    def _step(ids: list[int]) -> int:
        with torch.inference_mode():
            seq = torch.tensor([ids], device=device, dtype=torch.long)
            logits = model(seq, start_pos=0, cache=None)[0, -1]
            return int(logits.argmax().item())

    for step in range(NEW_TOKENS):
        next_id = _step(ids_list)
        ids_list.append(next_id)
        torch.mps.synchronize()
        live = torch.mps.current_allocated_memory()
        if step % 4 == 0 or step == NEW_TOKENS - 1:
            print(f"  step {step:3d}  seqlen={len(ids_list):4d}  live={live/MB:7.1f} MB  Δ={(live-base)/MB:+.1f}")
    final = torch.mps.current_allocated_memory()
    print(f"  total Δlive over {NEW_TOKENS} steps: {(final-base)/MB:+.1f} MB  ({(final-base)/NEW_TOKENS/1024:.0f} KB/step)")


def main() -> None:
    device = torch.device("mps")
    dtype = torch.bfloat16
    print("loading model...")
    model, _ = load_qwen3(device=device, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(HF_REPO)
    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ids_list = ids[0].tolist()

    run("A. baseline (no explicit free)", model, ids_list, explicit_free=False, device=device, dtype=dtype)
    run("B. explicit del + gc.collect()", model, ids_list, explicit_free=True,  device=device, dtype=dtype)
    run_inner_fn("C. inner-fn (the engine fix)", model, ids_list, device=device, dtype=dtype)


if __name__ == "__main__":
    main()
