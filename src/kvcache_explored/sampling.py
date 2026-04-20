"""Token sampler: temperature → top-k → top-p → min-p → multinomial.

Qwen3-0.6B's recommended defaults (from the model card's "Best Practices"):

    non-thinking:  temperature=0.7, top_p=0.8,  top_k=20, min_p=0
    thinking:      temperature=0.6, top_p=0.95, top_k=20, min_p=0

The Qwen team advises *against* greedy decoding in thinking mode (it induces
degenerate repetition), so ``sample()`` refuses a temperature of exactly 0
when ``enable_thinking`` is true and the caller is expected to route
explicit ``temperature=0`` through ``greedy()`` instead.

Order of operations matches Hugging Face's ``TemperatureLogitsWarper →
TopKLogitsWarper → TopPLogitsWarper → MinPLogitsWarper`` chain, for
consistency with any downstream comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.8
    min_p: float = 0.0

    @classmethod
    def non_thinking(cls) -> "SamplingParams":
        return cls(temperature=0.7, top_k=20, top_p=0.8, min_p=0.0)

    @classmethod
    def thinking(cls) -> "SamplingParams":
        return cls(temperature=0.6, top_k=20, top_p=0.95, min_p=0.0)


def greedy(logits: Tensor) -> int:
    """Argmax sampler, kept separate from the stochastic path for clarity."""
    return int(logits.argmax().item())


def sample(logits: Tensor, params: SamplingParams, *, enable_thinking: bool = False) -> int:
    """Sample one token id from a 1-D logits tensor (shape (V,)).

    Fallback: if every filter culls the distribution down to zero support
    (numerically possible with aggressive min-p), we softmax over the
    un-filtered temperature-scaled logits and sample from that.
    """
    if logits.ndim != 1:
        raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")

    if params.temperature <= 0.0:
        if enable_thinking:
            raise ValueError(
                "Qwen3 recommends against greedy decoding in thinking mode; "
                "use temperature=0.6 or call greedy() explicitly."
            )
        return greedy(logits)

    x = logits.float() / params.temperature

    # top-k: keep the k largest logits.
    if params.top_k > 0 and params.top_k < x.shape[-1]:
        kth = torch.topk(x, params.top_k).values[-1]
        x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)

    # top-p (nucleus): keep the smallest set whose cumulative prob >= p.
    if 0.0 < params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # Drop everything strictly after the first index whose cum >= p.
        keep = cum - probs <= params.top_p  # equivalently: keep[i] iff cum_before_i <= p
        keep[0] = True  # always keep the most likely token
        sorted_logits = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
        x = torch.empty_like(x).scatter_(-1, sorted_idx, sorted_logits)

    # min-p: drop tokens whose prob is below min_p * max_prob.
    if params.min_p > 0.0:
        probs = torch.softmax(x, dim=-1)
        threshold = params.min_p * probs.max()
        x = torch.where(probs < threshold, torch.full_like(x, float("-inf")), x)

    # Fallback: if everything got culled (shouldn't happen with Qwen defaults).
    if torch.isinf(x).all():
        x = logits.float() / params.temperature

    probs = torch.softmax(x, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
