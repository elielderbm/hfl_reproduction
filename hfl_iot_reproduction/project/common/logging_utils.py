import os, json, time, math
from pathlib import Path

LOGS = Path(os.getenv("LOGS_DIR", "/workspace/logs"))

def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj

def log_line(stream, record: dict):
    LOGS.mkdir(parents=True, exist_ok=True)
    p = LOGS/f"{stream}.jsonl"
    record["_ts"] = int(time.time()*1000)
    with open(p, "a", encoding="utf-8") as f:
        safe = _sanitize(record)
        f.write(json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n")

def log_metric(stream, **kwargs):
    log_line(stream, {"type":"metric", **kwargs})

def log_event(stream, **kwargs):
    log_line(stream, {"type":"event", **kwargs})
