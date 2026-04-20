/**
 * WebSocket message types mirroring src/kvcache_explored/schemas.py.
 * Kept hand-maintained; only ~7 shapes, and the backend is the source of truth.
 */

export type Role = "user" | "assistant" | "system";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface GenerateRequest {
  type: "generate";
  messages: ChatMessage[];
  use_cache: boolean;
  enable_thinking: boolean;
  max_new_tokens: number;
  max_seq_len: number;
  temperature: number;
  top_k: number;
  top_p: number;
  min_p: number;
  seed: number | null;
}

export interface GenStarted {
  type: "started";
  prompt_tokens: number;
  max_new_tokens: number;
  use_cache: boolean;
  max_seq_len: number;
}

export interface GenToken {
  type: "token";
  token_id: number;
  text: string;
  step_ms: number;
  step_index: number;
  seq_len: number;
  kv_length: number;
}

export interface GenDone {
  type: "done";
  total_ms: number;
  total_tokens: number;
  finish_reason: "length" | "eos" | "cancelled" | "overflow" | "error";
}

export interface GenError {
  type: "error";
  message: string;
}

export interface TelemetrySample {
  type: "mem";
  t_ms: number;
  params_bytes: number;
  params_breakdown: Record<string, number>;
  kv_cache_bytes_allocated: number;
  kv_cache_bytes_used: number;
  kv_cache_length: number;
  kv_cache_capacity: number;
  kv_per_layer_bytes_used: number[];
  activations_bytes: number;
  current_allocated_bytes: number;
  driver_bytes: number;
}

export interface ModelInfo {
  type: "info";
  model_name: string;
  device: string;
  dtype: string;
  num_layers: number;
  num_heads: number;
  num_kv_heads: number;
  head_dim: number;
  hidden_size: number;
  intermediate_size: number;
  vocab_size: number;
  max_position_embeddings: number;
  bytes_per_kv_token: number;
}

export type ChatMsg = GenStarted | GenToken | GenDone | GenError;
export type TelemetryMsg = ModelInfo | TelemetrySample;
