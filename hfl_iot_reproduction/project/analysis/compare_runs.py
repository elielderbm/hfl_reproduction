from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from project.analysis.paths import OUT


def _load_cloud_summary(run_dir: Path) -> pd.DataFrame:
    csv = run_dir / "metrics_all.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing metrics_all.csv in {run_dir}")
    df = pd.read_csv(csv)
    metrics = df[df["type"] == "metric"].copy() if "type" in df.columns else df.copy()
    cloud = metrics[metrics["file"].str.startswith("cloud")].copy() if "file" in metrics.columns else pd.DataFrame()
    if cloud.empty:
        return pd.DataFrame()
    rows = []
    groups = cloud.groupby("target") if "target" in cloud.columns else [("global", cloud)]
    for target, sub in groups:
        sub = sub.sort_values("round") if "round" in sub.columns else sub.sort_values("_ts")
        row = {
            "target": target,
            "rmse_last": float(sub["global_rmse"].dropna().iloc[-1]) if "global_rmse" in sub.columns and sub["global_rmse"].notna().any() else None,
            "rmse_best": float(sub["global_rmse"].min()) if "global_rmse" in sub.columns and sub["global_rmse"].notna().any() else None,
            "score_last": float(sub["global_score"].dropna().iloc[-1]) if "global_score" in sub.columns and sub["global_score"].notna().any() else None,
            "score_best": float(sub["global_score"].max()) if "global_score" in sub.columns and sub["global_score"].notna().any() else None,
            "r2_last": float(sub["global_r2"].dropna().iloc[-1]) if "global_r2" in sub.columns and sub["global_r2"].notna().any() else None,
            "mape_last": float(sub["global_mape"].dropna().iloc[-1]) if "global_mape" in sub.columns and sub["global_mape"].notna().any() else None,
            "rounds": int(sub["round"].max()) if "round" in sub.columns and sub["round"].notna().any() else len(sub),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _format_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base run directory (contains metrics_all.csv)")
    ap.add_argument("--new", default=str(OUT), help="New run directory (contains metrics_all.csv)")
    ap.add_argument("--out", default=str(OUT / "compare_runs.md"))
    args = ap.parse_args()

    base_dir = Path(args.base)
    new_dir = Path(args.new)
    out_path = Path(args.out)

    base = _load_cloud_summary(base_dir)
    new = _load_cloud_summary(new_dir)
    if base.empty or new.empty:
        print("[compare_runs] missing cloud metrics in one of the runs.")
        return

    merged = base.merge(new, on="target", suffixes=("_base", "_new"))
    for col in ("rmse_last", "rmse_best", "score_last", "score_best", "r2_last", "mape_last"):
        b = f"{col}_base"
        n = f"{col}_new"
        if b in merged.columns and n in merged.columns:
            merged[f"{col}_delta"] = merged[n] - merged[b]

    lines = []
    lines.append("# Compare Runs (Cloud Global)")
    lines.append("")
    lines.append(f"Base: {base_dir}")
    lines.append(f"New: {new_dir}")
    lines.append("")
    lines.append(_format_md(merged))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare_runs] wrote {out_path}")


if __name__ == "__main__":
    main()
