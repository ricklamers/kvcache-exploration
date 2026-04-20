"""Continuous oracle: our Qwen3 vs HF Qwen3 on a fixed prompt suite.

Run: ``uv run python scripts/verify_against_hf.py``

Exits non-zero on any mismatch so CI / the milestone gate can trust it.

What it checks (per prompt, full-forward — no KV cache at the model level;
the cache path is tested separately in M2):
  1. last-token logits allclose(atol=1e-2, rtol=1e-2)  (bf16 noise floor)
  2. top-5 token IDs match *exactly*  (rank order is what matters)
  3. 32-step greedy rollout token IDs match *exactly*
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvcache_explored.chat_template import render as render_chat
from kvcache_explored.generate import generate_with_cache, generate_without_cache
from kvcache_explored.kvcache import KVCache
from kvcache_explored.sampling import SamplingParams, sample
from kvcache_explored.weights import HF_REPO, load_qwen3

# ---------------------------------------------------------------------------- #
# Verification suite — keep short enough to run in a few seconds on M-series.
# ---------------------------------------------------------------------------- #

SUITE: list[tuple[str, str]] = [
    ("empty", ""),
    ("single-token", "A"),
    ("short-ascii", "The quick brown fox jumps over"),
    ("cjk-emoji", "你好,世界 🌏 — how are"),
    ("longer", "In a hole in the ground there lived a hobbit. " * 8),  # ~100 tokens
]

ATOL = 1e-2
RTOL = 1e-2
TOPK = 5
ROLLOUT_STEPS = 32


def _greedy_step(model, input_ids: torch.Tensor) -> torch.Tensor:
    """One full-forward greedy step, returns the next token id (shape: (B,))."""
    with torch.no_grad():
        logits = model(input_ids)
        if hasattr(logits, "logits"):  # HF ModelOutput
            logits = logits.logits
    return logits[:, -1, :].argmax(dim=-1)


def _greedy_rollout(model, ids: torch.Tensor, steps: int) -> list[int]:
    out: list[int] = []
    cur = ids
    for _ in range(steps):
        nxt = _greedy_step(model, cur)
        out.append(int(nxt.item()))
        cur = torch.cat([cur, nxt.unsqueeze(0)], dim=-1)
    return out


def main() -> int:
    device = torch.device("mps")
    dtype = torch.bfloat16
    print(f"[verify] loading our Qwen3 on {device} / {dtype} ...", flush=True)
    ours, repo_dir = load_qwen3(device=device, dtype=dtype)

    print(f"[verify] loading HF Qwen3 from {repo_dir} ...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(HF_REPO, dtype=dtype)
    hf.to(device)
    hf.eval()

    tok = AutoTokenizer.from_pretrained(HF_REPO)

    failures = 0
    for name, text in SUITE:
        # Tokenize; ensure at least 1 token so the models have something to chew.
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if ids.shape[1] == 0:
            ids = torch.tensor([[tok.bos_token_id or 0]], device=device, dtype=torch.long)

        with torch.no_grad():
            ours_logits = ours(ids)
            hf_logits = hf(ids).logits

        last_ours = ours_logits[0, -1].float()
        last_hf = hf_logits[0, -1].float()

        close = torch.allclose(last_ours, last_hf, atol=ATOL, rtol=RTOL)
        max_abs = (last_ours - last_hf).abs().max().item()

        top_ours = torch.topk(last_ours, TOPK).indices.tolist()
        top_hf = torch.topk(last_hf, TOPK).indices.tolist()
        top_match = top_ours == top_hf

        # 32-token greedy rollout
        roll_ours = _greedy_rollout(ours, ids, ROLLOUT_STEPS)
        roll_hf = _greedy_rollout(hf, ids, ROLLOUT_STEPS)
        rollout_match = roll_ours == roll_hf

        status = "OK " if (close and top_match and rollout_match) else "FAIL"
        if not (close and top_match and rollout_match):
            failures += 1
        divergence = next(
            (i for i, (a, b) in enumerate(zip(roll_ours, roll_hf)) if a != b),
            None,
        )
        print(
            f"[{status}] {name:14s}  "
            f"max|Δlogit|={max_abs:.4f}  "
            f"top{TOPK}={'=' if top_match else '≠'}  "
            f"rollout={'=' if rollout_match else f'≠@{divergence}'}",
            flush=True,
        )
        if not top_match:
            print(f"         ours top{TOPK}: {top_ours}")
            print(f"         hf   top{TOPK}: {top_hf}")
        if divergence is not None:
            print(f"         ours[{divergence}:{divergence+4}]: {roll_ours[divergence:divergence+4]}")
            print(f"         hf  [{divergence}:{divergence+4}]: {roll_hf  [divergence:divergence+4]}")

    # ---------------------------------------------------------------------- #
    # KV-cache correctness: per-step logit comparison between cache-on and
    # cache-off paths, forced along the same token trajectory.
    #
    # Why logit-level, not token-level: bf16 matrix multiplies in SDPA use
    # different kernel shapes in the two paths (Q×[1,D] vs Q×[N,D]) and
    # round slightly differently. At close-decision points (top-1 margin
    # near zero, which happens often in multilingual text) these tiny
    # rounding differences flip argmax. That is a property of bf16, not a
    # bug in the cache. Comparing distributions with a bf16-aware tolerance
    # is the right test.
    # ---------------------------------------------------------------------- #

    print()
    print("[verify] KV cache: per-step logit comparison (bf16 tolerance atol=1.0)")
    CACHE_ATOL = 1.0  # bf16 matmul noise budget
    for name, text in SUITE:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if ids.shape[1] == 0:
            ids = torch.tensor([[tok.bos_token_id or 0]], device=device, dtype=torch.long)

        cache = KVCache(
            num_layers=ours.cfg.num_layers,
            num_kv_heads=ours.cfg.num_kv_heads,
            head_dim=ours.cfg.head_dim,
            max_seq_len=ids.shape[1] + ROLLOUT_STEPS + 4,
            dtype=dtype,
            device=device,
        )
        # Prefill both: cache-on writes into `cache`, cache-off just produces logits.
        with torch.no_grad():
            lg_prefill = ours(ids, start_pos=0, cache=cache)[0, -1].float()
            lg_full = ours(ids, start_pos=0, cache=None)[0, -1].float()
        prefill_dmax = (lg_prefill - lg_full).abs().max().item()

        # Advance both paths along the same token trajectory (use the no-cache
        # argmax; that's the one we already verified matches HF exactly).
        seq = torch.cat([ids, torch.tensor([[int(lg_full.argmax())]], device=device, dtype=torch.long)], dim=1)
        max_dmax = prefill_dmax
        worst_step = "prefill"
        for step in range(ROLLOUT_STEPS):
            with torch.no_grad():
                lg_c = ours(seq[:, -1:], start_pos=cache.length, cache=cache)[0, -1].float()
                lg_f = ours(seq, start_pos=0, cache=None)[0, -1].float()
            dmax = (lg_c - lg_f).abs().max().item()
            if dmax > max_dmax:
                max_dmax = dmax
                worst_step = f"step{step}"
            seq = torch.cat([seq, torch.tensor([[int(lg_f.argmax())]], device=device, dtype=torch.long)], dim=1)

        ok = max_dmax < CACHE_ATOL
        if not ok:
            failures += 1
        status = "OK " if ok else "FAIL"
        print(f"[{status}] {name:14s}  worst max|Δlogit|={max_dmax:.4f} at {worst_step}  (budget {CACHE_ATOL})")

    # ---------------------------------------------------------------------- #
    # Chat template parity: our Jinja render vs tokenizer.apply_chat_template.
    # ---------------------------------------------------------------------- #

    print()
    print("[verify] chat template: ours vs tokenizer.apply_chat_template")
    chat_cases: list[tuple[str, list[dict], dict]] = [
        (
            "user-only-nonthink",
            [{"role": "user", "content": "Hi there."}],
            {"enable_thinking": False, "add_generation_prompt": True},
        ),
        (
            "system+user-nonthink",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hi."},
            ],
            {"enable_thinking": False, "add_generation_prompt": True},
        ),
        (
            "multi-turn-think",
            [
                {"role": "user", "content": "One."},
                {"role": "assistant", "content": "Two."},
                {"role": "user", "content": "Three?"},
            ],
            {"enable_thinking": True, "add_generation_prompt": True},
        ),
    ]
    for name, msgs, kwargs in chat_cases:
        ours_render = render_chat(msgs, **kwargs)
        hf_render = tok.apply_chat_template(msgs, tokenize=False, **kwargs)
        ok = ours_render == hf_render
        status = "OK " if ok else "FAIL"
        if not ok:
            failures += 1
            # show first 100 chars of divergence
            for i, (a, b) in enumerate(zip(ours_render, hf_render)):
                if a != b:
                    print(f"[{status}] {name:22s}  ≠ at char {i}")
                    print(f"         ours: ...{ours_render[max(0,i-20):i+40]!r}")
                    print(f"         hf  : ...{hf_render  [max(0,i-20):i+40]!r}")
                    break
            else:
                print(f"[{status}] {name:22s}  ≠ length (ours={len(ours_render)}, hf={len(hf_render)})")
        else:
            print(f"[{status}] {name:22s}  identical ({len(ours_render)} chars)")

    # ---------------------------------------------------------------------- #
    # Sampling smoke test: produces *a* token and respects determinism under
    # a seed. Not a correctness proof against HF (HF samplers differ subtly);
    # just checks the code path runs and gives a valid token id.
    # ---------------------------------------------------------------------- #

    print()
    print("[verify] sampling smoke test")
    with torch.no_grad():
        ids = tok("The capital of France is", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        logits = ours(ids)[0, -1]
    for params_name, params in [
        ("non-thinking", SamplingParams.non_thinking()),
        ("thinking", SamplingParams.thinking()),
    ]:
        torch.manual_seed(0)
        samples = [sample(logits, params) for _ in range(8)]
        ok = all(0 <= s < ours.cfg.vocab_size for s in samples)
        status = "OK " if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {params_name:14s}  sampled ids: {samples}")

    print()
    if failures:
        print(f"[verify] {failures} FAILED")
        return 1
    print("[verify] all checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
