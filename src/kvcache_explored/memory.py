"""Memory accounting for the UI telemetry channel.

Two kinds of numbers we report:

1. **Static parameter breakdown** — computed once, by walking
   ``model.named_parameters()`` and attributing each weight to a component
   (embeddings, per-layer attention, per-layer MLP, norms). This is the
   "here's how big the model is, broken down by part" view. It never changes.

2. **Live MPS counters** — ``torch.mps.current_allocated_memory()`` and
   ``torch.mps.driver_allocated_memory()``. We subtract the known
   ``params_bytes`` and ``kv_cache_bytes`` from ``current_allocated`` to
   estimate activation memory. Rough, but it's the honest signal — we're
   showing what the GPU actually holds, not a theoretical model.

Everything is bytes (int) so the UI can format to MB/GB however it likes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kvcache_explored.kvcache import KVCache
from kvcache_explored.model import Qwen3Config, Qwen3ForCausalLM


@dataclass
class ParamBreakdown:
    embeddings: int = 0  # embed_tokens (shared with lm_head when tied)
    attention: int = 0  # sum across layers of q/k/v/o + q/k norms
    mlp: int = 0  # sum across layers of gate/up/down
    norms: int = 0  # input_layernorm + post_attention_layernorm per layer, plus final norm
    lm_head: int = 0  # only nonzero if not tied
    total: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "embeddings": self.embeddings,
            "attention": self.attention,
            "mlp": self.mlp,
            "norms": self.norms,
            "lm_head": self.lm_head,
            "total": self.total,
        }


def breakdown_params(model: Qwen3ForCausalLM) -> ParamBreakdown:
    """Walk named_parameters once and bucket them.

    Accounts for tied embeddings: when ``lm_head.weight is embed_tokens.weight``
    (same storage) we attribute the bytes to ``embeddings`` only.
    """
    b = ParamBreakdown()
    seen_ids: set[int] = set()
    for name, p in model.named_parameters():
        # Skip duplicate storage (tied weights).
        pid = p.data_ptr()
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        nbytes = p.numel() * p.element_size()
        b.total += nbytes

        if name == "embed_tokens.weight":
            b.embeddings += nbytes
        elif name == "lm_head.weight":
            b.lm_head += nbytes
        elif name == "norm.weight":
            b.norms += nbytes
        elif ".self_attn." in name:
            b.attention += nbytes
        elif ".mlp." in name:
            b.mlp += nbytes
        elif "layernorm" in name:
            b.norms += nbytes
        else:
            raise RuntimeError(f"unclassified parameter: {name} ({nbytes} bytes)")
    return b


@dataclass
class MemorySample:
    params_bytes: int
    params_breakdown: dict[str, int]
    kv_cache_bytes_allocated: int  # full capacity of the cache tensors
    kv_cache_bytes_used: int  # prefix actually filled
    kv_cache_length: int
    kv_cache_capacity: int
    kv_per_layer_bytes_used: list[int]
    activations_bytes: int  # current_allocated - params - kv_cache_allocated
    current_allocated_bytes: int  # torch.mps.current_allocated_memory()
    driver_bytes: int  # torch.mps.driver_allocated_memory()


class MemoryProbe:
    """Snapshot source used by the WebSocket telemetry loop."""

    def __init__(self, model: Qwen3ForCausalLM, cache: KVCache | None = None) -> None:
        self.model = model
        self.cache = cache
        self._params = breakdown_params(model)

    def set_cache(self, cache: KVCache | None) -> None:
        self.cache = cache

    @property
    def params_bytes(self) -> int:
        return self._params.total

    def sample(self) -> MemorySample:
        if torch.backends.mps.is_available():
            cur = int(torch.mps.current_allocated_memory())
            drv = int(torch.mps.driver_allocated_memory())
        else:
            cur = 0
            drv = 0

        kv_alloc = self.cache.bytes_allocated if self.cache else 0
        kv_used = self.cache.bytes_used if self.cache else 0
        kv_len = self.cache.length if self.cache else 0
        kv_cap = self.cache.max_seq_len if self.cache else 0
        per_layer = self.cache.per_layer_bytes_used() if self.cache else []

        activations = max(0, cur - self._params.total - kv_alloc)

        return MemorySample(
            params_bytes=self._params.total,
            params_breakdown=self._params.as_dict(),
            kv_cache_bytes_allocated=kv_alloc,
            kv_cache_bytes_used=kv_used,
            kv_cache_length=kv_len,
            kv_cache_capacity=kv_cap,
            kv_per_layer_bytes_used=per_layer,
            activations_bytes=activations,
            current_allocated_bytes=cur,
            driver_bytes=drv,
        )


def estimate_kv_cache_bytes(cfg: Qwen3Config, max_seq_len: int, dtype: torch.dtype) -> int:
    """How big *would* a KV cache be for a given context budget and dtype?

    Used by the UI to show the memory cost of the slider *before* anyone
    clicks "generate".
    """
    element = torch.empty(0, dtype=dtype).element_size()
    return 2 * cfg.num_layers * cfg.num_kv_heads * max_seq_len * cfg.head_dim * element
