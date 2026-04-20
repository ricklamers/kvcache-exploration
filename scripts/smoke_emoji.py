"""Verify streaming decoder emits emoji/CJK whole, not as U+FFFD fragments."""

import asyncio
import json

import websockets


async def run() -> None:
    async with websockets.connect("ws://127.0.0.1:8000/ws/chat") as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "generate",
                    "messages": [
                        {"role": "user", "content": "Reply with 4 emojis and nothing else."}
                    ],
                    "use_cache": True,
                    "enable_thinking": False,
                    "max_new_tokens": 40,
                    "max_seq_len": 512,
                    "temperature": 0.7,
                    "top_k": 20,
                    "top_p": 0.8,
                    "min_p": 0.0,
                    "seed": 42,
                }
            )
        )
        text = ""
        steps = []
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "token":
                steps.append((msg["token_id"], msg["text"]))
                text += msg["text"]
            elif msg.get("type") == "done":
                break
            elif msg.get("type") == "error":
                print("ERROR:", msg["message"])
                return
        print("final text:", repr(text))
        print("contains U+FFFD:", "\ufffd" in text)
        print("steps:")
        for tid, t in steps:
            print(f"  {tid:6d}  {t!r}")


asyncio.run(run())
