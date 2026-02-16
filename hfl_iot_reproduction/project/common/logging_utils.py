import os, json, time
from pathlib import Path

LOGS = Path("/workspace/logs")

def log_line(stream, record: dict):
    LOGS.mkdir(parents=True, exist_ok=True)
    p = LOGS/f"{stream}.jsonl"
    record["_ts"] = int(time.time()*1000)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log_metric(stream, **kwargs):
    log_line(stream, {"type":"metric", **kwargs})

def log_event(stream, **kwargs):
    log_line(stream, {"type":"event", **kwargs})
