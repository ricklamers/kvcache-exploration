# KVCache Explored on Apple Silicon

A minimal, didactic implementation of Qwen3-0.6B in pure PyTorch, running on
Apple Silicon (MPS) in bf16, with a live chat UI that shows **why** KV cache
exists — by putting memory growth and per-token latency side by side in real
time.

- Single-file model (`src/kvcache_explored/model.py`), readable top to bottom.
- Hugging Face `transformers` stays installed as a **continuous oracle** — a
  verifier script asserts our logits match HF bitwise (top-5 + 32-token
  greedy rollout) and our chat-template output matches
  `tokenizer.apply_chat_template` byte-for-byte.
- React + Vite frontend with a VS Code / Cursor dark theme. Two WebSockets:
  `/ws/chat` streams tokens, `/ws/telemetry` streams memory at ~20 Hz.
- Cache on/off toggle so the pedagogical point — attention compute is
  **O(n²) per step** without a cache — is a single click away.

## Requirements

- Apple Silicon Mac (M1 or newer). 16 GB unified memory recommended if you
  want to run at the full 32k-token KV cache budget.
- [`uv`](https://docs.astral.sh/uv/) for Python tooling. No `pip`, no
  `conda`.
- Node 18+ and npm for the frontend.

## Install & run

```sh
# Python deps (creates .venv, installs torch, transformers, fastapi, etc)
uv sync

# First run downloads Qwen3-0.6B (~1.2 GB) from Hugging Face to ~/.cache.
uv run python scripts/verify_against_hf.py

# Frontend
(cd web && npm install && npm run build)

# Single-command launch — FastAPI serves the compiled web/dist at :8000
uv run uvicorn kvcache_explored.server:app --host 127.0.0.1 --port 8000
# → http://127.0.0.1:8000
```

During frontend development, run Vite separately with HMR:

```sh
(cd web && npm run dev)    # Vite on :5173, proxies /ws/* and /api/* to :8000
```

## What the demo teaches

Open the app, type a prompt, hit send. The right pane shows memory over
time (params, KV cache, activations, driver headroom) and per-token
latency.

Now toggle **KV cache** off and send the same prompt again:

- The latency chart develops a visible upward slope — each decoded token
  re-runs the whole forward, so attention compute scales **O((P+t)²)** per
  step (where P = prompt length, t = decoded so far).
- The memory chart loses the green "kv cache" band.
- The status bar's tokens/sec drops sharply.

That side-by-side is the whole point: **the cost moves. It has to land
somewhere.** KV cache trades linear memory growth for flat per-token time.
Without it, memory stays lean but each step pays the full quadratic cost.

For a CLI-only version of the same observation:

```sh
uv run python scripts/demo_quadratic.py
```

## What to look at in the code

- `src/kvcache_explored/model.py` — Qwen3-0.6B from scratch. RMSNorm, GQA
  attention with **QK-norm** (the Qwen3-specific bit), RoPE (θ=1e6),
  SwiGLU MLP. Pre-norm residual blocks, tied embeddings.
- `src/kvcache_explored/kvcache.py` — pre-allocated K/V tensors, one slot
  per layer. Length cursor advances on the last layer's append so every
  layer in a forward sees the same pre-state.
- `src/kvcache_explored/generate.py` — `generate_with_cache` (prefill once,
  decode one token per step) and `generate_without_cache` (re-prefill the
  whole sequence every step; the deliberately expensive baseline).
- `src/kvcache_explored/memory.py` — static parameter breakdown +
  `torch.mps.current_allocated_memory()` / `driver_allocated_memory()`.
- `src/kvcache_explored/chat_template.py` + `templates/qwen3_chat.jinja` —
  Qwen3's official Jinja chat template, in its own file.
- `src/kvcache_explored/sampling.py` — top-k → top-p → min-p → temperature,
  with Qwen's documented defaults baked in (`SamplingParams.non_thinking()`
  / `SamplingParams.thinking()`).
- `scripts/verify_against_hf.py` — the continuous HF oracle. Run it on
  every change.

## Verifier

```sh
uv run python scripts/verify_against_hf.py
```

Checks:
1. Our Qwen3's last-token logits match HF within bf16 tolerance (atol=1e-2).
2. Top-5 token IDs match exactly on a 5-prompt suite (empty, ASCII, CJK+emoji, long).
3. 32-token greedy rollout matches HF exactly.
4. Our KV cache path vs our no-cache path: per-step logit comparison with
   an atol=1.0 bf16 budget (cache correctness independent of HF, since
   ours == HF is already proven).
5. Our Jinja-rendered chat prompt equals `tokenizer.apply_chat_template(...)`
   byte-for-byte across user-only, system+user, and multi-turn-thinking cases.
6. Sampler smoke test (thinking / non-thinking presets both produce valid
   token ids).

Expected output: `[verify] all checks OK`.

## Sampling defaults

Copied verbatim from the [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)'s "Best Practices":

| Mode                         | temperature | top_p | top_k | min_p |
|------------------------------|-------------|-------|-------|-------|
| Non-thinking (default)       | 0.7         | 0.8   | 20    | 0.0   |
| Thinking (`<think>` blocks)  | 0.6         | 0.95  | 20    | 0.0   |

The Qwen team advises *against* greedy decoding in thinking mode; the UI's
thinking-mode toggle swaps the sampling preset automatically, and
`sample(..., enable_thinking=True)` refuses `temperature=0`.

## Memory cost at the default context

At the full 32,768-token KV cache budget in bf16, the cache occupies
**~3.58 GB** — about 3× the model weights (~1.14 GB). That relative cost
is the single most important thing this project is designed to make
visible. Smaller budgets scale linearly: 4k → 448 MB, 16k → 1.79 GB.

```
param   |   bytes
--------+----------
embed   |  296.8 MB
attn    |  336.0 MB
mlp     |  504.0 MB
norms   |    0.1 MB
lm_head |    0.0 MB  (tied to embeddings)
total   | 1136.9 MB

kv cache at various max_seq_len (bf16):
   1024 →  112 MB
   4096 →  448 MB
  16384 → 1792 MB
  32768 → 3584 MB
```

## Project layout

```
pyproject.toml                uv-managed, single source of truth
uv.lock                       committed
src/kvcache_explored/
  model.py                    Qwen3-0.6B in ~200 lines of PyTorch
  weights.py                  HF safetensors → our state_dict loader
  kvcache.py                  pre-allocated KV cache
  generate.py                 cache-on and cache-off decode loops
  sampling.py                 top-k / top-p / min-p / temperature sampler
  chat_template.py            Jinja renderer wrapping Qwen3's template
  templates/
    qwen3_chat.jinja          verbatim copy of Qwen3's chat template
  memory.py                   static + live MPS memory accounting
  schemas.py                  typed WS messages (pydantic, mirrored in TS)
  engine.py                   InferenceEngine (model + tokenizer + locks)
  server.py                   FastAPI + /ws/chat + /ws/telemetry
scripts/
  verify_against_hf.py        continuous HF oracle
  demo_quadratic.py           CLI demo of cache-on vs cache-off per-token cost
  diag_cache.py               per-step cache vs no-cache logit diagnostic
  memory_report.py            CLI memory breakdown
  smoke_ws.py                 hits /ws/chat with a tiny request
web/
  package.json                vite + react + ts
  vite.config.ts              dev server proxies /ws/* and /api/* to :8000
  src/
    App.tsx                   two-pane layout + status bar
    store.ts                  zustand store (chat + telemetry + runs)
    theme.ts                  VS Code / Cursor palette
    styles.css                flat dark-IDE CSS, no framework
    components/
      Chat.tsx                transcript + composer + controls
      Telemetry.tsx           cards + charts panel
      MemoryChart.tsx         uPlot stacked area (params, kv, acts, driver)
      LatencyChart.tsx        per-token latency with previous-run overlay
      KVCacheBar.tsx          fill bar, warn/error-colored
      KVLayerHeatmap.tsx      28 cells, per-layer KV fill
      StatusBar.tsx           VS Code–styled bottom bar
    hooks/
      useChatSocket.ts        /ws/chat client
      useTelemetrySocket.ts   /ws/telemetry client
    types/
      ws.ts                   TS mirror of schemas.py
PLAN.md                       design doc; updated as the project evolves
```

## Why HF stays a dependency

`transformers` is in the main (not dev) dependency group on purpose. It's
not ceremony — the verifier re-runs against it on every change, and until
we've seen several consecutive green runs post–major modification, we
don't trust our copy without the oracle alongside it.

## License

MIT.
