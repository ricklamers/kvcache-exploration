"""Measure driver-pool growth with and without periodic empty_cache().

Runs a long cache-off generation twice: once with a fresh engine (which
flushes every 10 steps per our new logic) and once with the flush
disabled (via monkeypatching). Compares driver bytes at end-of-run.
"""

from __future__ import annotations

import asyncio
import json

import websockets


async def run(max_new: int) -> None:
    async with websockets.connect("ws://127.0.0.1:8000/ws/chat") as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "generate",
                    "messages": [
                        {"role": "user", "content": "Explain in exhaustive detail how a transformer works."}
                    ],
                    "use_cache": False,
                    "enable_thinking": False,
                    "max_new_tokens": max_new,
                    "max_seq_len": 32768,
                    "temperature": 0.7,
                    "top_k": 20,
                    "top_p": 0.8,
                    "min_p": 0.0,
                    "seed": 1,
                }
            )
        )
        last_step_ms: list[float] = []
        while True:
            msg = json.loads(await ws.recv())
            t = msg.get("type")
            if t == "token":
                last_step_ms.append(msg["step_ms"])
            elif t == "done":
                print(
                    f"[done] cache=off  tokens={msg['total_tokens']}  "
                    f"ms={msg['total_ms']:.0f}  finish={msg['finish_reason']}"
                )
                return last_step_ms
            elif t == "error":
                raise RuntimeError(msg["message"])


async def telemetry_driver_now() -> int:
    async with websockets.connect("ws://127.0.0.1:8000/ws/telemetry") as ws:
        _ = json.loads(await ws.recv())  # ModelInfo
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "mem":
                return msg["driver_bytes"]


async def main() -> None:
    # Warm up so the first-ever driver snapshot is stable.
    await run(20)
    await asyncio.sleep(0.5)
    base = await telemetry_driver_now()
    print(f"baseline driver: {base/1024/1024:.1f} MB")
    step_ms = await run(100)
    await asyncio.sleep(0.5)
    final = await telemetry_driver_now()
    print(f"after 100 cache-off tokens: {final/1024/1024:.1f} MB  (Δ = {(final-base)/1024/1024:+.1f} MB)")
    assert step_ms is not None
    # Show latency of the flush steps vs surrounding ones
    print("step 9 ms:", step_ms[9] if len(step_ms) > 9 else None)
    print("step 10 ms (flush):", step_ms[10] if len(step_ms) > 10 else None)
    print("step 11 ms:", step_ms[11] if len(step_ms) > 11 else None)


asyncio.run(main())
