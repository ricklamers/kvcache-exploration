# KVCache Explored on Apple Silicon — Plan

A minimal, didactic implementation that shows **how KV cache shapes the cost of
autoregressive inference** on an Apple Silicon GPU, with a live chat UI that
streams tokens *and* live memory telemetry side by side.

---

## 1. Goals & non-goals

**Goals**
- Single-file, readable Qwen3-0.6B in pure PyTorch (~300 lines).
- Run on Apple Silicon via the `mps` backend with bf16 weights (the model's
  native training precision).
- Make the cost of attention **visible**: memory grows linearly with KV cache
  length; per-token latency grows linearly (because each new token attends over
  all prior K/V); total prefill/decode time grows quadratically.
- Provide a toggle to run *with* vs *without* KV cache, so the quadratic vs
  cubic total-cost difference is directly observable.
- Web chat UI with a side panel that streams real memory state over WebSockets.
- **HF `transformers` is our continuous verifier**, not just a one-shot check.
  See §7 below — we keep it in the dependency graph and keep running our
  implementation *against* it on every change until logits match bit-for-bit
  (within numerical tolerance) for a fixed suite of prompts.

**Non-goals**
- No training, no LoRA, no quantization (bf16 only — 0.6B fits easily).
- No speculative decoding, no batching beyond 1.
- No production hardening (auth, rate limiting, etc.).
- Not aiming to *replace* `transformers` — we aim to *reproduce* it. The HF
  model stays installed and loaded alongside ours as the ground truth oracle
  throughout development.

---

## 2. Architecture at a glance

```
┌─────────────────────────────┐        WebSocket          ┌──────────────────────┐
│  Browser (chat + charts)    │ ◄───── tokens ──────────► │  FastAPI server      │
│  - chat transcript pane     │ ◄───── mem samples ────── │  - /ws/chat          │
│  - memory stacked area      │ ◄───── timing ──────────► │  - /ws/telemetry     │
│  - per-layer KV heatmap     │                           │                      │
└─────────────────────────────┘                           │  InferenceEngine     │
                                                          │   ├─ Qwen3 (model.py)│
                                                          │   ├─ KVCache         │
                                                          │   └─ MemoryProbe     │
                                                          └──────────────────────┘
                                                                   │
                                                                   ▼
                                                             torch.mps (GPU)
```

---

## 3. Tooling & environment — **`uv` only, strictly**

- The Python side is managed **exclusively** by [`uv`](https://docs.astral.sh/uv/).
  No `pip`, no `conda`, no `poetry`, no `requirements.txt` hand-edits.
- `pyproject.toml` is the single source of truth. A `uv.lock` is committed.
- Every command in the README starts with `uv run …` (or `uv sync` once on
  first clone). CI runs the same commands.
- The frontend is managed exclusively by `npm` inside `web/` (Vite's
  convention). Two package managers, one per language, no overlap.

## 4. File layout

```
PLAN.md
README.md
pyproject.toml             # uv-managed; single source of Python deps
uv.lock                    # committed
src/
  kvcache_explored/
    __init__.py
    model.py               # Qwen3 definition, no deps beyond torch
    weights.py             # HF safetensors → our state_dict loader
    kvcache.py             # KVCache object + memory accounting
    generate.py            # prefill + decode loop, with/without cache
    memory.py              # MPS memory probing + per-component accounting
    sampling.py            # top-k / top-p / min-p / temperature sampler
    chat_template.py       # Jinja renderer (thin wrapper, see §10)
    templates/
      qwen3_chat.jinja     # the chat template itself, isolated
    server.py              # FastAPI app, WebSocket handlers
scripts/
  download_weights.py      # pulls Qwen3-0.6B from HF to ~/.cache
  verify_against_hf.py     # see §5 — the continuous HF oracle
web/                       # Vite + React frontend (own package.json)
  package.json
  vite.config.ts
  index.html
  src/
    main.tsx
    App.tsx
    components/
      Chat.tsx
      MemoryChart.tsx
      LatencyChart.tsx
      KVCacheBar.tsx
      KVLayerHeatmap.tsx
    hooks/
      useChatSocket.ts
      useTelemetrySocket.ts
    theme.ts               # VS Code / Cursor dark palette
    styles.css
```

---

## 5. Model (`model.py`)

Qwen3-0.6B architecture (verified against `Qwen/Qwen3-0.6B/config.json` and
the Qwen3 tech report, arXiv:2505.09388):
- 28 transformer blocks
- hidden size 1024, intermediate 3072
- 16 query heads, 8 KV heads (GQA ratio 2:1)
- head dim 128, with **QK-norm** (RMSNorm applied to q and k before RoPE) —
  this is a Qwen3-specific change from Qwen2 and must be implemented or
  logits will diverge
- RoPE with θ = 1,000,000, no ALiBi, no QKV bias (Qwen3 removed it)
- SwiGLU MLP, RMSNorm everywhere, tied input/output embeddings
- vocab 151,936
- native max context: **32,768** (config ships `max_position_embeddings:
  40960` but the model card specifies 32k as the native supported length; no
  YaRN extension is offered for the 0.6B variant)

Components I will write from scratch (no `transformers` imports in the model):
1. `RMSNorm`
2. `RotaryEmbedding` with precomputed cos/sin tables
3. `Attention` with GQA — takes an optional `KVCache` slot
4. `MLP` (SwiGLU: `down(silu(gate(x)) * up(x))`)
5. `Block` (pre-norm, residual)
6. `Qwen3ForCausalLM` (embeddings, blocks, final norm, lm_head tied to embed)

The file should read top-to-bottom like a tutorial. No abstract base classes,
no registries, no config dataclasses with 40 fields — just the shapes that
Qwen3-0.6B actually uses, pulled from a tiny `Qwen3Config` namedtuple.

---

## 6. Weight loading (`weights.py`)

- `huggingface_hub.snapshot_download("Qwen/Qwen3-0.6B")` on first run (cached
  in `~/.cache/huggingface`).
- Map HF parameter names → our module names via a small explicit rename table
  (~10 rules). No magic, no `**kwargs`; the renames are listed literally so a
  reader can audit them.
- Load into `torch.bfloat16` on `torch.device("mps")`. Only bf16 — no fp16
  fallback — since that's the model's native training precision.

## 7. HF `transformers` as continuous verifier (`scripts/verify_against_hf.py`)

This is a load-bearing part of the project, not an afterthought.

**Contract.** As long as our implementation is under development, there exists
a script that:
1. Loads the same `Qwen/Qwen3-0.6B` weights twice — once into HF
   `AutoModelForCausalLM`, once into our `Qwen3ForCausalLM`.
2. Runs both models on the same device (`mps`) with the same bf16 dtype, over
   a fixed **verification suite** of prompts covering:
   - empty / single-token
   - short ASCII
   - multilingual (CJK, emoji)
   - long context (> 2k tokens) to exercise RoPE beyond the short regime
   - a multi-turn chat-templated prompt
3. Compares, for each prompt:
   - final-token logits: `torch.allclose(ours, hf, atol=1e-2, rtol=1e-2)` —
     bf16 is noisy so we can't demand tighter than this
   - **top-5 token IDs must match exactly** — the stricter assertion, because
     rank order is what actually matters for decoding
   - greedy rollout of 32 tokens: token IDs must match **exactly** (any
     divergence here means a bug)
4. Also runs an incremental-decoding comparison: feed one token at a time to
   HF with its `past_key_values` and to our model with our `KVCache`,
   asserting logits match at every step. This is the single best test for
   KV cache correctness.

**The rule:** if `verify_against_hf.py` fails, nothing else is allowed to
proceed. It runs as the first step of every milestone below. HF
`transformers` stays a first-class dependency for the duration of the
project; it is only considered for removal *after* the full suite has been
green across several consecutive commits and we're confident the code is
stable.

## 8. KV cache (`kvcache.py`)

A tiny class that owns per-layer K and V tensors of shape
`(n_layers, 2, n_kv_heads, max_seq_len, head_dim)` pre-allocated to a chosen
`max_seq_len`. Why pre-allocated:
- makes memory cost *visible up-front* rather than growing in jumps;
- matches how real inference servers do it;
- lets the UI show the cache as a fixed-size buffer filling up.

Interface:
```python
cache = KVCache(n_layers, n_kv_heads, head_dim, max_seq_len, dtype, device)
cache.append(layer_idx, k, v)  # writes at cache.length for this layer
cache.length                    # current filled prefix
cache.bytes_allocated           # total, in bytes
cache.bytes_used                # length / max_seq_len fraction
```

The attention module takes `cache` + `layer_idx` + `start_pos`. When cache is
`None`, we run the no-cache path: each decode step re-projects K and V for the
*entire* prefix. This is the expensive baseline that makes the quadratic
effect obvious.

---

## 9. Generation (`generate.py`) + sampling (`sampling.py`)

Two cache modes, selectable per request:

| Mode          | Prefill cost | Decode step cost | Total for N tokens |
|---------------|--------------|------------------|--------------------|
| `cache=on`    | O(P²) once   | O(P + t)         | O(P² + PN + N²)    |
| `cache=off`   | O(P²) once   | O((P+t)²)        | O((P+N)³) roughly  |

Where P = prompt length, t = tokens decoded so far. The UI graphs per-token
latency so the difference is visible as a flat line vs a rising line.

The loop yields a stream of events:
```python
{"type": "token", "id": int, "text": str, "step_ms": float, "seq_len": int}
{"type": "mem",   "allocated": int, "driver": int, "kv_bytes": int, "per_layer": [...]}
```

**Sampling — defaults lifted verbatim from the Qwen3 model card's "Best
Practices" section** (https://huggingface.co/Qwen/Qwen3-0.6B):

| Mode                         | temperature | top_p | top_k | min_p | presence_penalty |
|------------------------------|-------------|-------|-------|-------|------------------|
| Non-thinking (default)       | 0.7         | 0.8   | 20    | 0.0   | 0.0              |
| Thinking (`<think>` blocks)  | 0.6         | 0.95  | 20    | 0.0   | 0.0 (up to 1.5)  |

The Qwen team explicitly advises **against greedy decoding in thinking mode**
(produces degenerate repetition), so we will *not* expose a "temperature=0"
shortcut in the UI and will warn if a user attempts `temperature=0` while
thinking is on. Our sampler therefore implements top-k + top-p + min-p +
temperature (in that order), which is the full combination the Qwen defaults
need. Greedy is still available as an internal path, used only by the HF
verifier.

## 10. Chat template — Jinja, in its own file (`chat_template.py`, `templates/qwen3_chat.jinja`)

- The Qwen3 tokenizer ships a Jinja chat template in `tokenizer_config.json`.
  We will **use Jinja** (the `jinja2` package) rather than reimplementing the
  format by hand — this is the only way to guarantee byte-identical output to
  HF across all conversation shapes (system prompts, tool calls, the
  `enable_thinking` toggle, etc.).
- The template itself lives in `templates/qwen3_chat.jinja`, copied once from
  the tokenizer config. Keeping it as a standalone file makes it diff-able
  against future Qwen releases and auditable by a human reader.
- `chat_template.py` is a ~30-line wrapper that loads the Jinja template,
  exposes `render(messages, *, enable_thinking: bool) -> str`, and is the
  **only** entry point to template rendering in the codebase. No ad-hoc
  string formatting of `<|im_start|>` tokens anywhere else.
- The verifier (§7) asserts that our Jinja-rendered prompt matches
  `tokenizer.apply_chat_template(...)` exactly for a set of conversations.

## 11. Memory probing (`memory.py`)

Apple Silicon specifics:
- `torch.mps.current_allocated_memory()` — live tensor bytes.
- `torch.mps.driver_allocated_memory()` — driver reservation (upper bound).
- We also compute **static** breakdowns by walking `model.named_parameters()`
  once at load time to attribute bytes to: embeddings, per-layer attention
  weights, per-layer MLP weights, norms, lm_head (tied → 0).
- KV cache bytes are known from the cache object itself.
- "Activations" = `current_allocated_memory() - params - kv_cache`. Rough but
  illustrative; that's the point.

Telemetry sample (sent every ~50 ms during generation):
```json
{
  "t_ms": 1234.5,
  "params_bytes": 1213410304,
  "kv_cache_bytes": 58720256,
  "kv_cache_length": 128,
  "kv_cache_capacity": 4096,
  "activations_bytes": 41234432,
  "driver_bytes": 1400000000,
  "per_layer_kv_bytes": [2097152, 2097152, ...]
}
```

---

## 12. Server (`server.py`)

FastAPI with two WebSocket endpoints:
- `/ws/chat` — receives `{prompt, use_cache, max_new_tokens, sampling, ...}`,
  streams token + timing events.
- `/ws/telemetry` — pushes memory samples at ~20 Hz from a background task
  that reads MPS counters. Separate socket so chart updates don't block on
  token generation.

One shared `InferenceEngine` singleton holds the model and serializes requests
(asyncio lock — single GPU, single model).

**In development:** `uv run uvicorn kvcache_explored.server:app --reload` on
port 8000. The Vite dev server (port 5173) proxies `/ws/*` to 8000.
**In production / one-shot demo:** we `npm run build` the frontend and the
FastAPI app serves the compiled `web/dist/` as static files from `/`, so a
single `uv run` command brings up the whole app.

## 13. Web UI (`web/` — Vite + React + TypeScript)

Stack:
- **Vite** for dev server / bundler (HMR, zero-config TS).
- **React 18** + **TypeScript** for a proper component model and typed
  message contracts between WS payloads and UI state.
- **Zustand** for the small amount of shared state (telemetry stream, chat
  transcript). Redux is overkill here; prop-drilling is unpleasant.
- **uPlot** for the live charts. Chart.js repaints the whole canvas on every
  update; at 20 Hz with a few thousand points that gets janky. uPlot is the
  fastest option in the JS ecosystem and stays smooth even with long traces.
- **`react-markdown`** for rendering assistant messages (code fences matter
  for an IDE-styled UI).
- **No UI framework** (no MUI / Chakra / Ant). The aesthetic is deliberate
  (see below) and we want full control over the palette.

### Theme — VS Code / Cursor dark IDE aesthetic

- Palette derived from VS Code "Dark Modern" + Cursor's accent treatment.
  Exact tokens in `web/src/theme.ts`:
  - background `#1e1e1e`, panel `#252526`, border `#3c3c3c`
  - foreground `#d4d4d4`, muted `#858585`
  - accent (cache-on, primary CTA) `#4ec9b0` (the VS Code teal)
  - warn (cache near capacity) `#dcdcaa`
  - error / quadratic bend highlight `#f48771`
  - chart series colors drawn from the same palette, not bright neon
- Typography: UI in `Inter`; code, numbers, and chart labels in `JetBrains
  Mono` (both via `@fontsource`, so we're not fetching Google Fonts at
  runtime).
- Layout: two columns (chat on the left, 40%; telemetry on the right, 60%).
  Thin, flat borders. No rounded corners on panels — square panels feel more
  "IDE" and less "consumer chat app". Status bar along the bottom (model
  name, device, peak memory, tokens/sec) styled like the VS Code status bar.
- Focus states and hover treatments follow VS Code's conventions (1px
  inset outline in accent color, no heavy drop shadows).

### Components

**Left column — chat**
- `Chat.tsx`: transcript, message input, send button.
- Controls row: KV cache on/off toggle, thinking-mode toggle (drives the
  Jinja template), sampling preset selector (thinking / non-thinking / raw),
  max-new-tokens input (defaults to a full-budget generation, see below).
- Each assistant token gets a small inline `ms` badge on hover.

**Right column — telemetry**
- `MemoryChart.tsx`: stacked area — params, KV cache, activations, driver
  headroom — over wall-clock time.
- `KVCacheBar.tsx`: horizontal fill bar, `length / capacity`. Tick marks
  colored with the warn/error palette entries as fill approaches 100%.
- `KVLayerHeatmap.tsx`: 28-cell row, one per transformer block. Uniform in
  practice, but the visual makes "the cache is 28 independent buffers"
  concrete.
- `LatencyChart.tsx`: per-token decode time (ms). Overlay of the current
  run's trace vs the previous run's trace so cache-on vs cache-off is
  visible simultaneously.
- Status bar: model name, device (`mps`), dtype, params bytes, peak driver
  bytes, tokens/sec (EMA).

### Transport

- Typed WebSocket message schemas defined once in
  `src/kvcache_explored/schemas.py` as Pydantic models and mirrored in
  `web/src/types/ws.ts` (a small codegen step via `datamodel-code-generator`
  or hand-maintained — start hand-maintained; it's ~6 types).

---

## 14. Default context window

The `max_seq_len` slider in the UI **defaults to 32,768** — the full
supported native context of Qwen3-0.6B. A 32k bf16 KV cache is ~450 MB,
which is larger than the 1.2 GB of model weights themselves — that is
exactly the point this project exists to make visible. The slider exposes
smaller values (1k, 2k, 4k, 8k, 16k, 32k) so a user can experiment with
smaller budgets, but the default is full-context so the "KV cache can
dominate memory" lesson is the first thing anyone sees.

The "max new tokens" control defaults to `min(4096, 32768 - prompt_tokens)`
to produce a meaningful length while leaving headroom — the plan is not to
crash-on-first-use, just to default to interesting.

## 15. What the demo teaches (the "quadratic" moment)

When the user runs a long prompt with cache OFF, the per-token-latency line
chart bends upward visibly — step N takes ~N× longer than step 1. Toggling
cache ON makes the same chart flat. Meanwhile the memory chart shows the
opposite tradeoff: cache ON grows a new green band that cache OFF doesn't
have. That single side-by-side comparison is the whole pedagogical payload.

---

## 16. Dependencies

### Python (managed by `uv`, declared in `pyproject.toml`)

```
torch>=2.4              # bf16 on MPS is solid from 2.3+
safetensors
huggingface_hub
tokenizers              # load tokenizer.json directly
transformers            # MANDATORY — our continuous verifier, see §7
jinja2                  # chat template rendering, see §10
fastapi
uvicorn[standard]
websockets
pydantic>=2             # typed WS message schemas
```

Dev-only (declared in the `dev` dependency group):
```
pytest
ruff
```

`transformers` is listed in the main dependencies, not dev dependencies,
because the live server reuses it for the verifier endpoint that compares
our logits against HF in real time. We will only consider moving it to dev
(or removing it) after several clean runs post-M2.

### Frontend (managed by `npm` inside `web/`)

```
react, react-dom
typescript, vite, @vitejs/plugin-react
zustand
uplot
react-markdown
@fontsource/inter, @fontsource/jetbrains-mono
```

No CSS framework, no component library.

---

## 17. Milestones & order of work

Every milestone begins by running `uv run python scripts/verify_against_hf.py`.
If it doesn't pass, the milestone does not proceed — that is the hard rule
from §7.

1. **M1 — Model forward pass correct.** `model.py` + `weights.py` + the
   verifier script. Last-token top-5 matches HF; greedy 32-token rollout
   matches HF exactly on the verification suite.
2. **M2 — KV cache + generation.** `kvcache.py` + `generate.py` + the
   incremental-decoding check in the verifier. Cache-on and cache-off paths
   both produce the same tokens as HF's `past_key_values` path.
3. **M3 — Sampling + chat template.** `sampling.py` + `chat_template.py` +
   `templates/qwen3_chat.jinja`. Verify the rendered prompt equals
   `tokenizer.apply_chat_template(...)` for a set of conversations;
   sampling uses Qwen's documented defaults.
4. **M4 — Memory probe.** `memory.py`. Prints a per-step report to stdout.
5. **M5 — Server + streaming chat.** FastAPI, `/ws/chat`, tokens stream.
6. **M6 — Frontend skeleton.** Vite + React app, dark IDE theme, chat
   pane connected, assistant tokens appear live.
7. **M7 — Telemetry WebSocket + charts.** `/ws/telemetry`, memory area
   chart, latency chart, KV fill bar, KV layer heatmap. This is where the
   "aha" lives.
8. **M8 — Polish.** README with screenshots, preset prompts that make the
   quadratic bend unmissable at cache-off with a 32k budget.

Each milestone is independently runnable — I won't build M5–M8 on top of a
broken M1–M4.
