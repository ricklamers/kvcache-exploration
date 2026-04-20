"""Issue several generations in a row and check that MPS memory does not grow.

Before the inference-mode fix, each cache-off request was leaking the full
decode graph into activations memory, growing ~10+ GB per call. After the
fix, live-tensor bytes should return to the baseline (params + kv_cache)
after each run.
"""

import asyncio
import json

import websockets


async def run() -> None:
    prompts = [
        ("Count to 10.", True),
        ("Name 5 fruits.", False),
        ("List 5 colors.", True),
        ("Give me a limerick.", False),
    ]

    async def one(prompt: str, use_cache: bool) -> None:
        async with websockets.connect("ws://127.0.0.1:8000/ws/chat") as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "generate",
                        "messages": [{"role": "user", "content": prompt}],
                        "use_cache": use_cache,
                        "enable_thinking": False,
                        "max_new_tokens": 64,
                        "max_seq_len": 32768,
                        "temperature": 0.7,
                        "top_k": 20,
                        "top_p": 0.8,
                        "min_p": 0.0,
                        "seed": 1,
                    }
                )
            )
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") == "done":
                    print(
                        f"[done] use_cache={use_cache}  "
                        f"{msg['total_tokens']} tok in {msg['total_ms']:.0f}ms"
                    )
                    return

    async def read_telemetry_once() -> tuple[int, int, int]:
        async with websockets.connect("ws://127.0.0.1:8000/ws/telemetry") as ws:
            # First message is ModelInfo, second is first TelemetrySample.
            _ = json.loads(await ws.recv())
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") == "mem":
                    return (
                        msg["current_allocated_bytes"],
                        msg["kv_cache_bytes_allocated"],
                        msg["activations_bytes"],
                    )

    for prompt, use_cache in prompts:
        await one(prompt, use_cache)
        await asyncio.sleep(0.3)  # let empty_cache settle
        cur, kv, act = await read_telemetry_once()
        mb = 1024 * 1024
        print(
            f"  live={cur/mb:.0f} MB  kv_cache_alloc={kv/mb:.0f} MB  "
            f"activations={act/mb:.0f} MB"
        )


asyncio.run(run())
