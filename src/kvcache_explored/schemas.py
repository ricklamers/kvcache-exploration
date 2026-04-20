"""Typed WebSocket message schemas.

Kept in one file so the React frontend can mirror these types exactly. All
messages are tagged with a ``type`` field so a single WS can carry multiple
event shapes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---- /ws/chat: client → server --------------------------------------------- #


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class GenerateRequest(BaseModel):
    type: Literal["generate"] = "generate"
    messages: list[ChatMessage]
    use_cache: bool = True
    enable_thinking: bool = False
    max_new_tokens: int = 512
    max_seq_len: int = 32_768  # KV cache capacity; ignored when use_cache=False
    # Sampling. `temperature=0` → greedy (only allowed when enable_thinking=False).
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.8
    min_p: float = 0.0
    seed: int | None = None


class CancelRequest(BaseModel):
    type: Literal["cancel"] = "cancel"


# ---- /ws/chat: server → client --------------------------------------------- #


class GenStarted(BaseModel):
    type: Literal["started"] = "started"
    prompt_tokens: int
    max_new_tokens: int
    use_cache: bool
    max_seq_len: int  # KV cache capacity for this run (0 if cache off)


class GenToken(BaseModel):
    type: Literal["token"] = "token"
    token_id: int
    text: str
    step_ms: float
    step_index: int  # 0 = first generated token (post-prefill)
    seq_len: int  # full context length the model saw on this step
    kv_length: int  # tokens now in cache (0 if cache off)


class GenDone(BaseModel):
    type: Literal["done"] = "done"
    total_ms: float
    total_tokens: int
    finish_reason: Literal["length", "eos", "cancelled", "overflow"] = "length"


class GenError(BaseModel):
    type: Literal["error"] = "error"
    message: str


# ---- /ws/telemetry: server → client ---------------------------------------- #


class TelemetrySample(BaseModel):
    type: Literal["mem"] = "mem"
    t_ms: float = Field(..., description="milliseconds since telemetry socket opened")
    params_bytes: int
    params_breakdown: dict[str, int]
    kv_cache_bytes_allocated: int
    kv_cache_bytes_used: int
    kv_cache_length: int
    kv_cache_capacity: int
    kv_per_layer_bytes_used: list[int]
    activations_bytes: int
    current_allocated_bytes: int
    driver_bytes: int


class ModelInfo(BaseModel):
    """Sent once on connect so the UI can render static facts (name, dims, etc)."""

    type: Literal["info"] = "info"
    model_name: str
    device: str
    dtype: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    bytes_per_kv_token: int  # 2 * num_layers * num_kv_heads * head_dim * element_size
