from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _safe(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return series.dropna()


def _mean(series: pd.Series) -> float | None:
    s = _safe(series)
    return float(s.mean()) if not s.empty else None


def _last(series: pd.Series) -> float | None:
    s = _safe(series)
    return float(s.iloc[-1]) if not s.empty else None


def _best_high(series: pd.Series) -> float | None:
    s = _safe(series)
    return float(s.max()) if not s.empty else None


def _best_low(series: pd.Series) -> float | None:
    s = _safe(series)
    return float(s.min()) if not s.empty else None


def _load_metrics(metrics_dir: Path) -> pd.DataFrame:
    csv = metrics_dir / "metrics_all.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing {csv}. Run extract_metrics first.")
    return pd.read_csv(csv)


def _pick_cols(iot: pd.DataFrame):
    score_col = None
    error_col = None
    r2_col = None
    mape_col = None
    acc_col = None

    if "train_score" in iot.columns:
        score_col = "train_score"
    elif "val_score" in iot.columns:
        score_col = "val_score"

    if "train_rmse" in iot.columns:
        error_col = "train_rmse"
    elif "val_rmse" in iot.columns:
        error_col = "val_rmse"

    if "train_r2" in iot.columns:
        r2_col = "train_r2"
    elif "val_r2" in iot.columns:
        r2_col = "val_r2"

    if "train_mape" in iot.columns:
        mape_col = "train_mape"
    elif "val_mape" in iot.columns:
        mape_col = "val_mape"

    if "val_acc" in iot.columns:
        acc_col = "val_acc"
    elif "train_acc" in iot.columns:
        acc_col = "train_acc"

    return score_col, error_col, r2_col, mape_col, acc_col


def _summary(df: pd.DataFrame) -> dict:
    metrics = df[df["type"] == "metric"].copy() if "type" in df.columns else df.copy()
    iot = metrics[metrics["file"].str.startswith("iot")].copy() if "file" in metrics.columns else pd.DataFrame()
    cloud = metrics[metrics["file"].str.startswith("cloud")].copy() if "file" in metrics.columns else pd.DataFrame()

    score_col, error_col, r2_col, mape_col, acc_col = _pick_cols(iot)

    local_score_last = None
    local_rmse_last = None
    local_r2_last = None
    local_mape_last = None
    local_acc_last = None
    if not iot.empty and "round" in iot.columns:
        g = iot.sort_values("round").groupby("iot")
        if score_col:
            local_score_last = _mean(g[score_col].last())
        if error_col:
            local_rmse_last = _mean(g[error_col].last())
        if r2_col:
            local_r2_last = _mean(g[r2_col].last())
        if mape_col:
            local_mape_last = _mean(g[mape_col].last())
        if acc_col:
            local_acc_last = _mean(g[acc_col].last())

    round_time_s = pd.Series(dtype=float)
    if not cloud.empty and "_ts" in cloud.columns:
        c = cloud.sort_values("_ts").copy()
        round_time_s = c["_ts"].diff() / 1000.0
        round_time_s = round_time_s[round_time_s > 0]

    throughput = pd.Series(dtype=float)
    if not cloud.empty and "_ts" in cloud.columns and "payload_bytes" in cloud.columns:
        sub = cloud.sort_values("_ts").copy()
        sub["delta_s"] = sub["_ts"].diff() / 1000.0
        sub = sub[sub["delta_s"] > 0]
        sub["throughput_kbps"] = (sub["payload_bytes"] / 1024.0) / sub["delta_s"]
        throughput = sub["throughput_kbps"].dropna()

    if not cloud.empty and "target" in cloud.columns:
        g_target = cloud.sort_values("round").groupby("target")
        global_score_last = _mean(g_target["global_score"].last()) if "global_score" in cloud.columns else None
        global_rmse_last = _mean(g_target["global_rmse"].last()) if "global_rmse" in cloud.columns else None
        global_r2_last = _mean(g_target["global_r2"].last()) if "global_r2" in cloud.columns else None
        global_mape_last = _mean(g_target["global_mape"].last()) if "global_mape" in cloud.columns else None
        global_acc_last = _mean(g_target["global_acc"].last()) if "global_acc" in cloud.columns else None
    else:
        global_score_last = _last(cloud["global_score"]) if "global_score" in cloud.columns else None
        global_rmse_last = _last(cloud["global_rmse"]) if "global_rmse" in cloud.columns else None
        global_r2_last = _last(cloud["global_r2"]) if "global_r2" in cloud.columns else None
        global_mape_last = _last(cloud["global_mape"]) if "global_mape" in cloud.columns else None
        global_acc_last = _last(cloud["global_acc"]) if "global_acc" in cloud.columns else None

    return {
        "local_score_last": local_score_last,
        "local_rmse_last": local_rmse_last,
        "local_r2_last": local_r2_last,
        "local_mape_last": local_mape_last,
        "local_acc_last": local_acc_last,
        "global_score_last": global_score_last,
        "global_rmse_last": global_rmse_last,
        "global_r2_last": global_r2_last,
        "global_mape_last": global_mape_last,
        "global_acc_last": global_acc_last,
        "round_time_s_mean": _mean(round_time_s),
        "throughput_kbps_mean": _mean(throughput),
        "enc_ms_mean": _mean(iot["enc_ms"]) if "enc_ms" in iot.columns else None,
    }


def _load_target_summary(metrics_dir: Path) -> pd.DataFrame | None:
    path = metrics_dir / "target_round_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    rows = []
    for target, sub in df.groupby("target"):
        sub = sub.sort_values("round")
        rows.append({
            "target": target,
            "score_last": float(sub["score"].iloc[-1]),
            "rmse_last": float(sub["rmse"].iloc[-1]),
            "r2_last": float(sub["r2"].iloc[-1]) if "r2" in sub.columns else None,
            "mape_last": float(sub["mape"].iloc[-1]) if "mape" in sub.columns else None,
        })
    return pd.DataFrame(rows).set_index("target").sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--async-dir", required=True)
    ap.add_argument("--sync-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    async_dir = Path(args.async_dir)
    sync_dir = Path(args.sync_dir)

    df_async = _load_metrics(async_dir)
    df_sync = _load_metrics(sync_dir)

    s_async = _summary(df_async)
    s_sync = _summary(df_sync)

    out_path = Path(args.out) if args.out else async_dir / "compare_async_sync.md"

    lines = []
    lines.append("# Run Comparison (Deep)")
    lines.append("")
    lines.append("## Core Metrics")
    lines.append("| Metric | Run A | Run B | Delta |")
    lines.append("| --- | --- | --- | --- |")

    def _delta(a, b, fmt="{:+.4f}"):
        if a is None or b is None:
            return "n/a"
        try:
            return fmt.format(a - b)
        except Exception:
            return "n/a"

    for key, label in [
        ("local_score_last", "Local Score (last, mean)"),
        ("local_rmse_last", "Local RMSE (last, mean)"),
        ("local_r2_last", "Local R2 (last, mean)"),
        ("local_mape_last", "Local MAPE (last, mean)"),
        ("local_acc_last", "Local Acc (last, mean)"),
        ("global_score_last", "Global Score (last)"),
        ("global_rmse_last", "Global RMSE (last)"),
        ("global_r2_last", "Global R2 (last)"),
        ("global_mape_last", "Global MAPE (last)"),
        ("global_acc_last", "Global Acc (last)"),
        ("round_time_s_mean", "Round Time (mean, s)"),
        ("throughput_kbps_mean", "Throughput (KB/s, mean)"),
        ("enc_ms_mean", "Encrypt Overhead (ms, mean)"),
    ]:
        lines.append(
            f"| {label} | {s_async.get(key, 'n/a')} | {s_sync.get(key, 'n/a')} | {_delta(s_async.get(key), s_sync.get(key))} |"
        )

    # Per-target comparison if available
    t_async = _load_target_summary(async_dir)
    t_sync = _load_target_summary(sync_dir)
    if t_async is not None and t_sync is not None:
        lines.append("")
        lines.append("## Per-Target Comparison (Global Model)")
        lines.append("| Target | Run A Score (last) | Run B Score (last) | Delta | Run A RMSE (last) | Run B RMSE (last) | Delta |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for target in sorted(set(t_async.index) | set(t_sync.index)):
            a = t_async.loc[target] if target in t_async.index else None
            b = t_sync.loc[target] if target in t_sync.index else None
            a_score = a["score_last"] if a is not None else None
            b_score = b["score_last"] if b is not None else None
            a_rmse = a["rmse_last"] if a is not None else None
            b_rmse = b["rmse_last"] if b is not None else None
            lines.append(
                f"| {target} | {a_score} | {b_score} | {_delta(a_score, b_score)} | {a_rmse} | {b_rmse} | {_delta(a_rmse, b_rmse)} |"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare_async_sync] wrote {out_path}")


if __name__ == "__main__":
    main()
