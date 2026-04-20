"""Qwen3-0.6B from scratch in PyTorch.

Written to read top-to-bottom like a tutorial. Every layer is the same small
idea repeated, so we unroll it explicitly rather than hiding behind a base
class.

Architecture facts for Qwen3-0.6B, verified against
``Qwen/Qwen3-0.6B/config.json`` and the Qwen3 tech report (arXiv:2505.09388):

    layers=28, hidden=1024, intermediate=3072,
    num_attention_heads=16, num_kv_heads=8, head_dim=128,
    vocab=151936, rope_theta=1_000_000.0,
    tie_word_embeddings=True, qk_norm=True, qkv_bias=False,
    rms_norm_eps=1e-6, native max context=32768.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 151_936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40_960  # config.json value; the *supported* length is 32_768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Matches HF: cast to fp32 for the variance, cast back to input dtype.
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x.to(in_dtype)) * self.weight


# --------------------------------------------------------------------------- #
# Rotary position embedding
# --------------------------------------------------------------------------- #


def build_rope_cache(
    max_seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Precompute cos and sin tables of shape (max_seq_len, head_dim)."""
    # inv_freq shape: (head_dim // 2,)
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (T, half)
    # HF Qwen3 duplicates frequencies along the last axis: [f0, f1, ..., f0, f1, ...]
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q,k: (B, H, T, D); cos,sin: (T, D) already sliced to the positions we want."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


# --------------------------------------------------------------------------- #
# Attention (grouped-query, with QK-norm — Qwen3 specific)
# --------------------------------------------------------------------------- #


class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_groups = cfg.num_heads // cfg.num_kv_heads  # how many Q heads share a KV head

        q_dim = cfg.num_heads * cfg.head_dim  # 16 * 128 = 2048 (not == hidden, this is normal)
        kv_dim = cfg.num_kv_heads * cfg.head_dim  # 8 * 128 = 1024

        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)

        # Qwen3 QK-norm: per-head-dim RMSNorm applied to q and k *before* RoPE.
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        cache: "KVCache | None" = None,
        layer_idx: int = 0,
    ) -> Tensor:
        B, T, _ = x.shape

        # 1. Project to q, k, v and split into heads.
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # shapes: q (B, Hq, T, D), k/v (B, Hkv, T, D)

        # 2. QK-norm (Qwen3-specific — omit this and logits diverge).
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. RoPE.
        q, k = apply_rope(q, k, cos, sin)

        # 4. Cache update: if we have a cache, append new k/v and read back the full prefix.
        if cache is not None:
            k, v = cache.append(layer_idx, k, v)

        # 5. Repeat KV heads to match Q heads (explicit, not SDPA's enable_gqa, so it works in any torch version).
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # 6. SDPA. is_causal is only valid when Q and K have the same length
        #    (i.e. prefill or full-forward cache-off path). When Q is shorter
        #    than K (incremental decode with cache) we pass attn_mask=None
        #    because the new token legally attends to the entire prefix.
        is_causal = attn_mask is None and q.shape[-2] == k.shape[-2] and q.shape[-2] > 1
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )

        # 7. Merge heads and project back.
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# SwiGLU MLP
# --------------------------------------------------------------------------- #


class MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# Transformer block (pre-norm, residual)
# --------------------------------------------------------------------------- #


class Block(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        cache: "KVCache | None",
        layer_idx: int,
    ) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_mask, cache, layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# --------------------------------------------------------------------------- #
# Whole model
# --------------------------------------------------------------------------- #


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # lm_head is tied to embed_tokens (see Qwen3-0.6B config). We store a
        # distinct Linear for clarity; weight tying happens in weights.py.
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # RoPE tables live on the model so they move with .to(device).
        # Allocated lazily via `build_rope` because device/dtype aren't known yet.
        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)

    def build_rope(self, max_seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        cos, sin = build_rope_cache(
            max_seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype
        )
        self._rope_cos = cos
        self._rope_sin = sin

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Tensor,  # (B, T)
        *,
        start_pos: int = 0,  # absolute position of input_ids[:, 0]
        cache: "KVCache | None" = None,
    ) -> Tensor:
        # This project is inference-only. Wrapping the forward method itself
        # with ``torch.inference_mode()`` makes that property a property of
        # the *model*, not of every caller: no generator decorator, no
        # per-script context manager, and no async-suspend edge case can
        # accidentally reintroduce autograd tracking. Inference tensors that
        # escape here cannot be used in a grad-enabled context later, which
        # is exactly the invariant we want.
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)

        # Slice precomputed RoPE for the positions we're about to use.
        if self._rope_cos.numel() == 0:
            raise RuntimeError("call model.build_rope(max_seq_len, device, dtype) once before forward")
        cos = self._rope_cos[start_pos : start_pos + T]
        sin = self._rope_sin[start_pos : start_pos + T]

        # Build a causal mask only when we're doing a multi-token forward
        # (prefill, or cache-off full forward). For single-token decode with
        # cache, SDPA needs no mask at all — the new token attends to every
        # cached token legally.
        attn_mask: Tensor | None
        if T > 1:
            # SDPA will also accept is_causal=True when Q and K are the same
            # length. If Q < K (shouldn't happen with T>1 here, but for safety
            # when `start_pos > 0` AND T > 1 — prompt prefill with priors in
            # cache), we build an explicit mask.
            kv_len = (cache.length + T) if cache is not None else T
            if cache is not None and cache.length > 0:
                attn_mask = _causal_mask(T, kv_len, start_pos, x.device, x.dtype)
            else:
                attn_mask = None  # Attention uses is_causal=True in this shape
        else:
            attn_mask = None

        for i, block in enumerate(self.layers):
            x = block(x, cos, sin, attn_mask, cache, i)

        x = self.norm(x)
        return self.lm_head(x)


def _causal_mask(
    q_len: int, kv_len: int, q_start: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Additive mask for SDPA: 0 where attend, -inf where forbidden.

    Query token at local index i has absolute position q_start + i. It may
    attend to any KV position j <= q_start + i.
    """
    q_pos = torch.arange(q_len, device=device) + q_start
    k_pos = torch.arange(kv_len, device=device)
    # (q_len, kv_len) bool: True means allowed
    allowed = k_pos[None, :] <= q_pos[:, None]
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask  # broadcast to (B, H, q_len, kv_len) by SDPA
