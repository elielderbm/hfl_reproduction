from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from project.analysis.paths import OUT


def _pick_cols(df: pd.DataFrame):
    score_col = None
    error_col = None
    error_label = "error"
    r2_col = None
    mape_col = None

    if "train_score" in df.columns:
        score_col = "train_score"
    elif "val_score" in df.columns:
        score_col = "val_score"
    elif "train_acc" in df.columns:
        score_col = "train_acc"
    elif "val_acc" in df.columns:
        score_col = "val_acc"

    if "train_rmse" in df.columns:
        error_col = "train_rmse"
        error_label = "rmse"
    elif "val_rmse" in df.columns:
        error_col = "val_rmse"
        error_label = "rmse"
    elif "train_loss" in df.columns:
        error_col = "train_loss"
        error_label = "mse"
    elif "val_loss" in df.columns:
        error_col = "val_loss"
        error_label = "mse"

    if "train_r2" in df.columns:
        r2_col = "train_r2"
    elif "val_r2" in df.columns:
        r2_col = "val_r2"
    if "train_mape" in df.columns:
        mape_col = "train_mape"
    elif "val_mape" in df.columns:
        mape_col = "val_mape"

    return score_col, error_col, error_label, r2_col, mape_col


def _window_delta(series: pd.Series, window: int):
    if series is None or series.empty:
        return None
    if len(series) < 2 * window:
        return None
    tail = series.tail(window).mean()
    prev = series.iloc[-2 * window : -window].mean()
    return float(tail - prev)


def _window_std(series: pd.Series, window: int):
    if series is None or series.empty:
        return None
    if len(series) < window:
        return None
    return float(series.tail(window).std())


def _summarize_iot(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    score_col, error_col, error_label, r2_col, mape_col = _pick_cols(df)
    rows = []
    for iot_id, sub in df.groupby("iot"):
        sub = sub.sort_values("round")
        score = sub[score_col].dropna() if score_col and score_col in sub.columns else pd.Series(dtype=float)
        error = sub[error_col].dropna() if error_col and error_col in sub.columns else pd.Series(dtype=float)
        r2 = sub[r2_col].dropna() if r2_col and r2_col in sub.columns else pd.Series(dtype=float)
        mape = sub[mape_col].dropna() if mape_col and mape_col in sub.columns else pd.Series(dtype=float)

        row = {
            "iot": iot_id,
            "rounds": int(sub["round"].max()) if "round" in sub.columns and sub["round"].notna().any() else len(sub),
            "score_first": float(score.iloc[0]) if len(score) else None,
            "score_last": float(score.iloc[-1]) if len(score) else None,
            "score_best": float(score.max()) if len(score) else None,
            "score_delta_last_window": _window_delta(score, window),
            "score_std_last_window": _window_std(score, window),
            f"{error_label}_first": float(error.iloc[0]) if len(error) else None,
            f"{error_label}_last": float(error.iloc[-1]) if len(error) else None,
            f"{error_label}_best": float(error.min()) if len(error) else None,
            f"{error_label}_delta_last_window": _window_delta(error, window),
            f"{error_label}_std_last_window": _window_std(error, window),
            "r2_last": float(r2.iloc[-1]) if len(r2) else None,
            "r2_best": float(r2.max()) if len(r2) else None,
            "mape_last": float(mape.iloc[-1]) if len(mape) else None,
            "mape_best": float(mape.min()) if len(mape) else None,
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("iot").sort_index()


def _summarize_cloud(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    cloud = df.sort_values("round") if "round" in df.columns else df.sort_values("_ts")
    score_col = "global_score" if "global_score" in cloud.columns else None
    rmse_col = "global_rmse" if "global_rmse" in cloud.columns else None
    r2_col = "global_r2" if "global_r2" in cloud.columns else None
    mape_col = "global_mape" if "global_mape" in cloud.columns else None

    score = cloud[score_col].dropna() if score_col else pd.Series(dtype=float)
    rmse = cloud[rmse_col].dropna() if rmse_col else pd.Series(dtype=float)
    r2 = cloud[r2_col].dropna() if r2_col else pd.Series(dtype=float)
    mape = cloud[mape_col].dropna() if mape_col else pd.Series(dtype=float)

    row = {
        "rounds": int(cloud["round"].max()) if "round" in cloud.columns and cloud["round"].notna().any() else len(cloud),
        "score_first": float(score.iloc[0]) if len(score) else None,
        "score_last": float(score.iloc[-1]) if len(score) else None,
        "score_best": float(score.max()) if len(score) else None,
        "score_delta_last_window": _window_delta(score, window),
        "score_std_last_window": _window_std(score, window),
        "rmse_first": float(rmse.iloc[0]) if len(rmse) else None,
        "rmse_last": float(rmse.iloc[-1]) if len(rmse) else None,
        "rmse_best": float(rmse.min()) if len(rmse) else None,
        "rmse_delta_last_window": _window_delta(rmse, window),
        "rmse_std_last_window": _window_std(rmse, window),
        "r2_last": float(r2.iloc[-1]) if len(r2) else None,
        "r2_best": float(r2.max()) if len(r2) else None,
        "mape_last": float(mape.iloc[-1]) if len(mape) else None,
        "mape_best": float(mape.min()) if len(mape) else None,
    }

    return pd.DataFrame([row])


def _write_md(path: Path, iot_df: pd.DataFrame, cloud_df: pd.DataFrame, window: int):
    lines = []
    lines.append("# Convergence & Error Report")
    lines.append("")
    lines.append(f"Window for deltas/std: last {window} rounds")
    lines.append("")

    def _to_table(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown()
        except Exception:
            return df.to_string()

    if not iot_df.empty:
        lines.append("## IoT Summary")
        lines.append(_to_table(iot_df))
        lines.append("")
    else:
        lines.append("## IoT Summary")
        lines.append("_No IoT metrics found._")
        lines.append("")

    if not cloud_df.empty:
        lines.append("## Cloud Summary")
        lines.append(_to_table(cloud_df))
        lines.append("")
    else:
        lines.append("## Cloud Summary")
        lines.append("_No cloud metrics found._")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default=str(OUT))
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--show-rounds", type=int, default=0, help="print last N rounds per IoT and cloud")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    csv = metrics_dir / "metrics_all.csv"
    if not csv.exists():
        print(f"[convergence] missing {csv}. Run extract_metrics first.")
        return

    try:
        df = pd.read_csv(csv)
    except Exception as e:
        print(f"[convergence] failed to read {csv}: {e}")
        return

    if df.empty:
        print(f"[convergence] {csv} is empty. Run the simulation and extract_metrics first.")
        return

    metrics = df[df["type"] == "metric"].copy() if "type" in df.columns else df.copy()
    iot = metrics[metrics["file"].str.startswith("iot")].copy() if "file" in metrics.columns else pd.DataFrame()
    cloud = metrics[metrics["file"].str.startswith("cloud")].copy() if "file" in metrics.columns else pd.DataFrame()

    iot_summary = _summarize_iot(iot, args.window)
    cloud_summary = _summarize_cloud(cloud, args.window)

    out = metrics_dir / "convergence_report.md"
    _write_md(out, iot_summary, cloud_summary, args.window)

    print(f"[convergence] wrote {out}")
    if not iot_summary.empty:
        print("\n== IoT Convergence ==")
        print(iot_summary.to_string())
    if not cloud_summary.empty:
        print("\n== Cloud Convergence ==")
        print(cloud_summary.to_string(index=False))

    if args.show_rounds and args.show_rounds > 0:
        score_col, error_col, error_label, r2_col, mape_col = _pick_cols(iot)
        if not iot.empty and "round" in iot.columns:
            g = iot.groupby(["iot", "round"]).mean(numeric_only=True).reset_index()
            print("\n== IoT Last Rounds ==")
            cols = ["iot", "round"] + [c for c in [score_col, error_col, r2_col, mape_col] if c]
            print(g.sort_values("round").groupby("iot").tail(args.show_rounds)[cols].to_string(index=False))
        if not cloud.empty and "round" in cloud.columns:
            cols = ["round"] + [c for c in ["global_score", "global_rmse", "global_r2", "global_mape"] if c in cloud.columns]
            print("\n== Cloud Last Rounds ==")
            print(cloud.sort_values("round").tail(args.show_rounds)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
