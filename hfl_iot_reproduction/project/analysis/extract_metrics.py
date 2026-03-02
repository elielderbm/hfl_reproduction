import json
from project.analysis.paths import LOGS, OUT

OUT.mkdir(parents=True, exist_ok=True)

def collect_to_csv():
    rows = []
    log_files = list(LOGS.glob("*.jsonl"))
    for p in log_files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    rows.append({"file": p.name, **j})
                except Exception:
                    # fallback for legacy NaN/Inf emitted by older loggers
                    try:
                        fixed = (
                            line.replace("NaN", "null")
                            .replace("Infinity", "null")
                            .replace("-Infinity", "null")
                        )
                        j = json.loads(fixed)
                        rows.append({"file": p.name, **j})
                    except Exception:
                        pass

    # write unified CSV (always with header to avoid EmptyDataError)
    of = OUT / "metrics_all.csv"
    import pandas as pd
    tmp = of.with_suffix(of.suffix + ".tmp")

    if not rows:
        base_cols = ["file", "type", "_ts"]
        pd.DataFrame(columns=base_cols).to_csv(tmp, index=False)
        tmp.replace(of)
        if not log_files:
            print(f"[analysis] no log files found in {LOGS}. Generated empty {of}.")
        else:
            print(f"[analysis] log files found in {LOGS}, but no valid JSONL rows. Generated empty {of}.")
        return

    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(of)
    print(f"[analysis] wrote {of}")

if __name__ == "__main__":
    collect_to_csv()
