import json, csv, os
from pathlib import Path

LOGS = Path("/workspace/logs")
OUT  = Path("/workspace/outputs")
OUT.mkdir(parents=True, exist_ok=True)

def collect_to_csv():
    rows = []
    for p in LOGS.glob("*.jsonl"):
        with open(p,"r",encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    rows.append({"file":p.name, **j})
                except:
                    pass
    # write unified CSV
    of = OUT/"metrics_all.csv"
    import pandas as pd
    pd.DataFrame(rows).to_csv(of, index=False)
    print(f"[analysis] wrote {of}")

if __name__ == "__main__":
    collect_to_csv()
