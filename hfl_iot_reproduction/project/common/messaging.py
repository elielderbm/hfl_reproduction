import json, base64, time, asyncio, websockets, os
from dataclasses import dataclass

def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

async def ws_server(host, port, handler):
    async with websockets.serve(handler, host, port, ping_interval=20, ping_timeout=20):
        await asyncio.Future()  # run forever

async def ws_connect(uri):
    return await websockets.connect(uri, ping_interval=20, ping_timeout=20)

def now_ms(): return int(time.time()*1000)
