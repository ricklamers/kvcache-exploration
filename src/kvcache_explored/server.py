"""FastAPI server: /ws/chat streams tokens, /ws/telemetry streams memory.

Serves the compiled Vite bundle from ``web/dist/`` (if it exists) at ``/``,
so a single ``uv run uvicorn kvcache_explored.server:app`` brings up the
whole app. During frontend development, run Vite separately on :5173 and
point it at this server (default: ``:8000``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kvcache_explored.chat_template import render as render_chat
from kvcache_explored.engine import GenParams, InferenceEngine
from kvcache_explored.schemas import (
    ChatMessage,
    GenDone,
    GenError,
    GenStarted,
    GenToken,
    GenerateRequest,
    ModelInfo,
    TelemetrySample,
)

log = logging.getLogger("kvcache_explored.server")

app = FastAPI(title="KVCache Explored")
_engine = InferenceEngine()


@app.on_event("startup")
def _startup() -> None:
    log.warning("loading Qwen3-0.6B on MPS ...")
    _engine.load()
    log.warning("model ready")


class TokenizeRequest(BaseModel):
    messages: list[ChatMessage]
    enable_thinking: bool = True
    add_generation_prompt: bool = True


class TokenizeResponse(BaseModel):
    prompt_tokens: int
    prompt_chars: int
    max_new_tokens_available: int  # max_position_embeddings - prompt_tokens


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest) -> JSONResponse:
    """Render the chat template and count tokens.

    Frontend uses this to show a live "max_new_tokens_available" next to the
    slider so the user can spend the whole context budget without manually
    counting prompt tokens. Called on every transcript/thinking-toggle
    change, so it must stay fast — a render + a BPE encode, both cheap.
    """
    assert _engine.tokenizer is not None
    max_ctx = _engine.model.cfg.max_position_embeddings
    # Empty conversation: the Qwen3 template dereferences ``messages[0]``
    # unconditionally, so short-circuit before touching Jinja.
    if not req.messages:
        return JSONResponse(
            TokenizeResponse(
                prompt_tokens=0,
                prompt_chars=0,
                max_new_tokens_available=max_ctx,
            ).model_dump()
        )
    text = render_chat(
        [m.model_dump() for m in req.messages],
        enable_thinking=req.enable_thinking,
        add_generation_prompt=req.add_generation_prompt,
    )
    ids = _engine.tokenizer(text, add_special_tokens=False).input_ids
    n = len(ids)
    return JSONResponse(
        TokenizeResponse(
            prompt_tokens=n,
            prompt_chars=len(text),
            max_new_tokens_available=max(1, max_ctx - n),
        ).model_dump()
    )


@app.post("/api/restart")
async def restart() -> JSONResponse:
    """Cancel any in-flight run, drop the KV cache, flush MPS pool."""
    await _engine.restart()
    return JSONResponse({"ok": True})


@app.get("/api/health")
def health() -> JSONResponse:
    m = _engine.model
    dtype_str = str(_engine.dtype).replace("torch.", "")
    element_size = torch.empty(0, dtype=_engine.dtype).element_size()
    info = ModelInfo(
        model_name="Qwen/Qwen3-0.6B",
        device=str(_engine.device),
        dtype=dtype_str,
        num_layers=m.cfg.num_layers,
        num_heads=m.cfg.num_heads,
        num_kv_heads=m.cfg.num_kv_heads,
        head_dim=m.cfg.head_dim,
        hidden_size=m.cfg.hidden_size,
        intermediate_size=m.cfg.intermediate_size,
        vocab_size=m.cfg.vocab_size,
        max_position_embeddings=m.cfg.max_position_embeddings,
        bytes_per_kv_token=2 * m.cfg.num_layers * m.cfg.num_kv_heads * m.cfg.head_dim * element_size,
    )
    return JSONResponse(info.model_dump())


# --------------------------------------------------------------------------- #
# /ws/chat
# --------------------------------------------------------------------------- #


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket) -> None:
    """One WebSocket carries two conversations: generate-requests from the
    client (streamed back as token events), and cancel messages that must
    be honored *mid-generation*.

    Since ``async for step in _engine.generate(...)`` monopolizes this
    coroutine, we spin up a sibling task that drains the socket in the
    background and flips ``_engine.cancel()`` on any cancel message. The
    sibling is torn down between generations so there's no orphan task
    state between requests.
    """
    await ws.accept()

    async def _drain_for_cancel(stop: asyncio.Event) -> None:
        # Reads until either a cancel arrives or the outer caller sets ``stop``
        # (by cancelling this task). Any other message is ignored here —
        # the client shouldn't interleave a second generate request while
        # one is already in flight.
        try:
            while not stop.is_set():
                msg = await ws.receive_json()
                if msg.get("type") == "cancel":
                    _engine.cancel()
                    # Keep reading: the client may follow up with more messages.
        except WebSocketDisconnect:
            _engine.cancel()
        except asyncio.CancelledError:
            pass

    try:
        while True:
            raw = await ws.receive_json()
            if raw.get("type") == "cancel":
                # No run in flight → nothing to do.
                continue
            try:
                req = GenerateRequest.model_validate(raw)
            except Exception as e:
                await ws.send_json(GenError(message=f"bad request: {e}").model_dump())
                continue

            t0 = time.perf_counter()
            await ws.send_json(
                GenStarted(
                    prompt_tokens=0,  # filled in by first token event
                    max_new_tokens=req.max_new_tokens,
                    use_cache=req.use_cache,
                    max_seq_len=req.max_seq_len if req.use_cache else 0,
                ).model_dump()
            )

            params = GenParams(
                use_cache=req.use_cache,
                enable_thinking=req.enable_thinking,
                max_new_tokens=req.max_new_tokens,
                max_seq_len=req.max_seq_len,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                min_p=req.min_p,
                seed=req.seed,
            )

            stop = asyncio.Event()
            drain = asyncio.create_task(_drain_for_cancel(stop))
            total_tokens = 0
            finish = "length"
            try:
                async for step in _engine.generate(req.messages, params):
                    total_tokens += 1
                    await ws.send_json(
                        GenToken(
                            token_id=step.token_id,
                            text=step.text,
                            step_ms=step.step_ms,
                            step_index=step.step_index,
                            seq_len=step.seq_len,
                            kv_length=step.kv_length,
                        ).model_dump()
                    )
            except Exception as e:
                log.exception("generation failed")
                await ws.send_json(GenError(message=str(e)).model_dump())
                finish = "error"
            finally:
                stop.set()
                drain.cancel()
                try:
                    await drain
                except (asyncio.CancelledError, Exception):
                    pass

            if finish != "error":
                finish = _engine.last_finish_reason
            total_ms = (time.perf_counter() - t0) * 1000.0
            await ws.send_json(
                GenDone(
                    total_ms=total_ms,
                    total_tokens=total_tokens,
                    finish_reason=finish,  # type: ignore[arg-type]
                ).model_dump()
            )
    except WebSocketDisconnect:
        _engine.cancel()


# --------------------------------------------------------------------------- #
# /ws/telemetry — 20 Hz memory samples
# --------------------------------------------------------------------------- #


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket) -> None:
    await ws.accept()
    t0 = time.perf_counter()

    # Send the ModelInfo on connect, then stream memory samples.
    m = _engine.model
    dtype_str = str(_engine.dtype).replace("torch.", "")
    element_size = torch.empty(0, dtype=_engine.dtype).element_size()
    await ws.send_json(
        ModelInfo(
            model_name="Qwen/Qwen3-0.6B",
            device=str(_engine.device),
            dtype=dtype_str,
            num_layers=m.cfg.num_layers,
            num_heads=m.cfg.num_heads,
            num_kv_heads=m.cfg.num_kv_heads,
            head_dim=m.cfg.head_dim,
            hidden_size=m.cfg.hidden_size,
            intermediate_size=m.cfg.intermediate_size,
            vocab_size=m.cfg.vocab_size,
            max_position_embeddings=m.cfg.max_position_embeddings,
            bytes_per_kv_token=2 * m.cfg.num_layers * m.cfg.num_kv_heads * m.cfg.head_dim * element_size,
        ).model_dump()
    )

    try:
        while True:
            s = _engine.probe.sample()
            await ws.send_json(
                TelemetrySample(
                    t_ms=(time.perf_counter() - t0) * 1000.0,
                    params_bytes=s.params_bytes,
                    params_breakdown=s.params_breakdown,
                    kv_cache_bytes_allocated=s.kv_cache_bytes_allocated,
                    kv_cache_bytes_used=s.kv_cache_bytes_used,
                    kv_cache_length=s.kv_cache_length,
                    kv_cache_capacity=s.kv_cache_capacity,
                    kv_per_layer_bytes_used=s.kv_per_layer_bytes_used,
                    activations_bytes=s.activations_bytes,
                    current_allocated_bytes=s.current_allocated_bytes,
                    driver_bytes=s.driver_bytes,
                ).model_dump()
            )
            await asyncio.sleep(0.05)  # ~20 Hz
    except WebSocketDisconnect:
        return


# --------------------------------------------------------------------------- #
# Static files (only if the Vite bundle exists).
# --------------------------------------------------------------------------- #

_web_dist = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
if _web_dist.is_dir():
    app.mount("/", StaticFiles(directory=_web_dist, html=True), name="web")
