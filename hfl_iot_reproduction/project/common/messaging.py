import json, base64, time, asyncio, websockets, os
from dataclasses import dataclass

def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

async def ws_server(host, port, handler):
    ping_interval = _parse_ping(os.getenv("WS_PING_INTERVAL"), 20)
    ping_timeout = _parse_ping(os.getenv("WS_PING_TIMEOUT"), 20)
    async with websockets.serve(
        handler,
        host,
        port,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
        max_size=_parse_max_size(os.getenv("WS_MAX_SIZE", "16mb")),
    ):
        await asyncio.Future()  # run forever

async def ws_connect(uri):
    ping_interval = _parse_ping(os.getenv("WS_PING_INTERVAL"), 20)
    ping_timeout = _parse_ping(os.getenv("WS_PING_TIMEOUT"), 20)
    return await websockets.connect(
        uri,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
        max_size=_parse_max_size(os.getenv("WS_MAX_SIZE", "16mb")),
    )

def _parse_max_size(raw):
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("", "0", "none", "null"):
        return None
    try:
        if s.endswith("mb"):
            return int(float(s[:-2]) * 1024 * 1024)
        if s.endswith("kb"):
            return int(float(s[:-2]) * 1024)
        return int(float(s))
    except ValueError:
        return None

def _parse_ping(raw, default):
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("", "0", "none", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return default

def now_ms(): return int(time.time()*1000)
