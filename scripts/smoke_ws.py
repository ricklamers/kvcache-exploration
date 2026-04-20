"""End-to-end WS smoke test: hit /ws/chat with a tiny request and print events.

Run: ``uv run python scripts/smoke_ws.py``

Assumes the server is running on :8000.
"""

from __future__ import annotations

import asyncio
import json

import websockets


async def run() -> None:
    uri = "ws://127.0.0.1:8000/ws/chat"
    async with websockets.connect(uri) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "generate",
                    "messages": [{"role": "user", "content": "Say hi in 5 words."}],
                    "use_cache": True,
                    "max_new_tokens": 16,
                    "max_seq_len": 256,
                    "temperature": 0.7,
                    "top_k": 20,
                    "top_p": 0.8,
                    "min_p": 0.0,
                    "seed": 42,
                }
            )
        )
        text_out = ""
        while True:
            msg = json.loads(await ws.recv())
            t = msg.get("type")
            if t == "token":
                text_out += msg["text"]
                print(f"[{msg['step_index']:>3d}] +{msg['step_ms']:>6.1f}ms  kv_len={msg['kv_length']:<4d}  {msg['text']!r}")
            elif t == "done":
                print(f"[done] {msg['total_tokens']} tokens in {msg['total_ms']:.0f}ms  reason={msg['finish_reason']}")
                print(f"[text] {text_out!r}")
                break
            elif t == "started":
                print(f"[started] use_cache={msg['use_cache']}  max_new={msg['max_new_tokens']}")
            elif t == "error":
                print(f"[error] {msg['message']}")
                break
            else:
                print(msg)


asyncio.run(run())
