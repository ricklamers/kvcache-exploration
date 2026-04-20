"""Load Qwen3-0.6B safetensors from the HF hub into our Qwen3ForCausalLM.

The param name mapping is deliberately explicit — no globs, no wildcards —
so a reader can audit every rename.

HF name                                                    → our name
-----------------------------------------------------------   ---------------------------------
model.embed_tokens.weight                                  → embed_tokens.weight
model.norm.weight                                          → norm.weight
model.layers.{i}.input_layernorm.weight                    → layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight           → layers.{i}.post_attention_layernorm.weight
model.layers.{i}.self_attn.{q,k,v,o}_proj.weight           → layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.layers.{i}.self_attn.{q,k}_norm.weight               → layers.{i}.self_attn.{q,k}_norm.weight
model.layers.{i}.mlp.{gate,up,down}_proj.weight            → layers.{i}.mlp.{gate,up,down}_proj.weight
lm_head.weight   (absent when tie_word_embeddings)         → lm_head.weight (tied to embed_tokens)
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from kvcache_explored.model import Qwen3Config, Qwen3ForCausalLM

HF_REPO = "Qwen/Qwen3-0.6B"


def download_weights(repo: str = HF_REPO) -> Path:
    """Fetch model weights + tokenizer files from HF, return local dir."""
    path = snapshot_download(
        repo_id=repo,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "tokenizer*",
            "*.txt",
        ],
    )
    return Path(path)


def _rename_hf_to_ours(name: str) -> str:
    if name.startswith("model."):
        return name[len("model.") :]
    # lm_head.weight when present passes through unchanged
    return name


def load_qwen3(
    *,
    device: torch.device | str = "mps",
    dtype: torch.dtype = torch.bfloat16,
    repo: str = HF_REPO,
    max_seq_len: int = 32_768,
) -> tuple[Qwen3ForCausalLM, Path]:
    """Materialize the model on `device` in `dtype` with HF weights loaded.

    Returns the model plus the local dir where tokenizer files also live
    (callers use this to load the HF tokenizer without a second download).
    """
    repo_dir = download_weights(repo)

    # 1. Read HF config.json just to sanity-check our hardcoded Qwen3Config.
    import json

    cfg_json = json.loads((repo_dir / "config.json").read_text())
    cfg = Qwen3Config(
        vocab_size=cfg_json["vocab_size"],
        hidden_size=cfg_json["hidden_size"],
        intermediate_size=cfg_json["intermediate_size"],
        num_layers=cfg_json["num_hidden_layers"],
        num_heads=cfg_json["num_attention_heads"],
        num_kv_heads=cfg_json["num_key_value_heads"],
        head_dim=cfg_json.get("head_dim", cfg_json["hidden_size"] // cfg_json["num_attention_heads"]),
        max_position_embeddings=cfg_json["max_position_embeddings"],
        rope_theta=cfg_json["rope_theta"],
        rms_norm_eps=cfg_json["rms_norm_eps"],
        tie_word_embeddings=cfg_json.get("tie_word_embeddings", True),
    )

    # 2. Build our model.
    model = Qwen3ForCausalLM(cfg)

    # 3. Collect and concatenate all safetensors shards.
    shards = sorted(repo_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors found in {repo_dir}")
    hf_state: dict[str, torch.Tensor] = {}
    for shard in shards:
        hf_state.update(load_file(str(shard)))

    # 4. Rename and load.
    ours = {_rename_hf_to_ours(name): tensor for name, tensor in hf_state.items()}

    # Handle tied lm_head: HF config says tie → lm_head.weight is not in the
    # safetensors, so we point our lm_head at the embedding weight.
    if "lm_head.weight" not in ours and cfg.tie_word_embeddings:
        ours["lm_head.weight"] = ours["embed_tokens.weight"]

    missing, unexpected = model.load_state_dict(ours, strict=False)
    if missing:
        raise RuntimeError(f"missing params when loading Qwen3: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected params when loading Qwen3: {unexpected}")

    # 5. Cast + move, and tie (share storage, not just equal values).
    model = model.to(device=device, dtype=dtype)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight

    # 6. Precompute RoPE tables.
    model.build_rope(max_seq_len, torch.device(device), dtype)

    model.eval()
    return model, repo_dir
