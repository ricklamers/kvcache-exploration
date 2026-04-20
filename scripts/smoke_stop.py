"""Test mid-generation cancellation.

Sends a generate request, lets a few tokens stream, then sends a cancel.
The server should stop promptly and return finish_reason=cancelled.
"""

from __future__ import annotations

import asyncio
import json
import sys

import websockets


async def run() -> int:
    async with websockets.connect("ws://127.0.0.1:8000/ws/chat") as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "generate",
                    "messages": [
                        {"role": "user", "content": "Write a 1000-word essay about cheese."}
                    ],
                    "use_cache": True,
                    "enable_thinking": False,
                    "max_new_tokens": 2000,
                    "max_seq_len": 32768,
                    "temperature": 0.7,
                    "top_k": 20,
                    "top_p": 0.8,
                    "min_p": 0.0,
                    "seed": 1,
                }
            )
        )
        token_count = 0
        stop_sent = False
        while True:
            msg = json.loads(await ws.recv())
            t = msg.get("type")
            if t == "token":
                token_count += 1
                if token_count >= 10 and not stop_sent:
                    print(f"[client] got {token_count} tokens; sending cancel")
                    await ws.send(json.dumps({"type": "cancel"}))
                    stop_sent = True
            elif t == "done":
                print(
                    f"[done] tokens={msg['total_tokens']}  "
                    f"ms={msg['total_ms']:.0f}  finish={msg['finish_reason']}"
                )
                if msg["finish_reason"] != "cancelled":
                    print("  EXPECTED finish_reason=cancelled — BUG")
                    return 1
                if msg["total_tokens"] >= 2000:
                    print("  cancel was not honored — BUG")
                    return 1
                return 0
            elif t == "error":
                print(f"[error] {msg['message']}")
                return 1


async def restart() -> int:
    import aiohttp  # type: ignore[import-not-found]

    async with aiohttp.ClientSession() as s:
        async with s.post("http://127.0.0.1:8000/api/restart") as r:
            body = await r.json()
            print(f"[restart] status={r.status} body={body}")
            return 0 if body.get("ok") else 1


def main() -> int:
    rc = asyncio.run(run())
    return rc


if __name__ == "__main__":
    sys.exit(main())
