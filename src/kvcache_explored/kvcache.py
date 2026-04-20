"""A pre-allocated KV cache.

Layout: one tensor pair per transformer layer, each of shape
``(1, num_kv_heads, max_seq_len, head_dim)``. Batch is fixed at 1 for this
project (single-user chat demo). The cache is pre-allocated at construction
time so its memory cost is *up-front* and visible in the UI — this mirrors
how production inference servers (vLLM, TRT-LLM, SGLang) size their caches.

Contract with the attention module (see model.py):

    k_full, v_full = cache.append(layer_idx, k_new, v_new)

where ``k_new``/``v_new`` have shape ``(1, num_kv_heads, new_tokens, head_dim)``
and the returned tensors cover the full prefix so far. ``append`` mutates
internal state; call once per layer per forward.

A single ``cache.length`` counter tracks how many tokens are stored; it is
incremented by the **last** layer's append call, so the *i*-th layer sees the
pre-increment length and writes into ``[length : length + new_tokens]``.
"""

from __future__ import annotations

import torch
from torch import Tensor


class KVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (num_layers, 1, num_kv_heads, max_seq_len, head_dim)
        # Two big tensors (K, V) rather than a list of small ones: one allocation
        # per role, much kinder to MPS and makes `bytes_allocated` a trivial
        # numel * element_size call.
        self.k = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=dtype, device=self.device)
        self.length = 0

    # ---- memory accounting ------------------------------------------------ #

    @property
    def bytes_allocated(self) -> int:
        return (self.k.numel() + self.v.numel()) * self.k.element_size()

    @property
    def bytes_used(self) -> int:
        per_layer_bytes = 2 * self.num_kv_heads * self.length * self.head_dim * self.k.element_size()
        return self.num_layers * per_layer_bytes

    def per_layer_bytes_used(self) -> list[int]:
        b = 2 * self.num_kv_heads * self.length * self.head_dim * self.k.element_size()
        return [b] * self.num_layers

    # ---- append & reset --------------------------------------------------- #

    def append(self, layer_idx: int, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Write k_new/v_new at slot [length : length+T] for this layer.

        Returns the full filled prefix (k, v) for this layer, shape
        ``(B, Hkv, length_after, D)``.
        """
        T = k_new.shape[2]
        end = self.length + T
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: layer {layer_idx} would write [{self.length}:{end}] "
                f"into capacity {self.max_seq_len}"
            )
        self.k[layer_idx, :, :, self.length : end, :] = k_new
        self.v[layer_idx, :, :, self.length : end, :] = v_new

        k_full = self.k[layer_idx, :, :, :end, :]
        v_full = self.v[layer_idx, :, :, :end, :]

        # Only the last layer advances the length counter — that way every
        # layer in this forward sees the same pre-state.
        if layer_idx == self.num_layers - 1:
            self.length = end
        return k_full, v_full

    def reset(self) -> None:
        self.length = 0
        # Don't bother zeroing the storage — the length cursor makes unseen
        # slots unreachable, and zero-fill on MPS is surprisingly slow.
