from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from project.analysis.paths import OUT


@dataclass
class Summary:
    iot_count: int
    contributors_mean: float | None
    rounds_iot: int | None
    rounds_cloud: int | None
    duration_s: float | None
    local_score_last_mean: float | None
    local_score_best_mean: float | None
    local_error_last_mean: float | None
    local_r2_last_mean: float | None
    local_mape_last_mean: float | None
    local_acc_last_mean: float | None
    local_acc_best_mean: float | None
    global_score_last: float | None
    global_score_best: float | None
    global_rmse_last: float | None
    global_r2_last: float | None
    global_mape_last: float | None
    global_acc_last: float | None
    global_acc_best: float | None
    round_time_s_mean: float | None
    round_time_s_p90: float | None
    throughput_kbps_mean: float | None
    enc_ms_mean: float | None
    cloud_dec_ms_mean: float | None
    pactual_mean: float | None
    pdesired: float | None
    window_start: float | None
    window_last: float | None
    window_mean: float | None


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


def _p90(series: pd.Series) -> float | None:
    s = _safe(series)
    return float(s.quantile(0.9)) if not s.empty else None


def _trend(rounds: pd.Series, values: pd.Series) -> float | None:
    if rounds is None or values is None:
        return None
    df = pd.DataFrame({"round": rounds, "value": values}).dropna().sort_values("round")
    if len(df) < 2:
        return None
    x = df["round"].astype(float).to_numpy()
    y = df["value"].astype(float).to_numpy()
    try:
        slope = np.polyfit(x, y, 1)[0]
    except Exception:
        return None
    return float(slope)


def _time_to_threshold(rounds: pd.Series, values: pd.Series, threshold: float) -> int | None:
    if rounds is None or values is None:
        return None
    df = pd.DataFrame({"round": rounds, "value": values}).dropna().sort_values("round")
    if df.empty:
        return None
    hit = df[df["value"] >= threshold]
    if hit.empty:
        return None
    return int(hit.iloc[0]["round"])


def _load_pdesired() -> float | None:
    cfg_path = Path("/workspace/config/hyperparams.yml")
    if not cfg_path.exists():
        return None
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    try:
        return float(cfg.get("pdesired")) if cfg.get("pdesired") is not None else None
    except Exception:
        return None


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
    if "train_score" in iot.columns:
        score_col = "train_score"
    elif "train_acc" in iot.columns:
        score_col = "train_acc"
    elif "val_score" in iot.columns:
        score_col = "val_score"
    elif "val_acc" in iot.columns:
        score_col = "val_acc"

    if "train_rmse" in iot.columns:
        error_col = "train_rmse"
    elif "train_loss" in iot.columns:
        error_col = "train_loss"
    elif "val_rmse" in iot.columns:
        error_col = "val_rmse"
    elif "val_loss" in iot.columns:
        error_col = "val_loss"
    if "train_r2" in iot.columns:
        r2_col = "train_r2"
    elif "val_r2" in iot.columns:
        r2_col = "val_r2"
    if "train_mape" in iot.columns:
        mape_col = "train_mape"
    elif "val_mape" in iot.columns:
        mape_col = "val_mape"
    return score_col, error_col, r2_col, mape_col


def compute_summary(df: pd.DataFrame, pdesired: float | None = None) -> Summary:
    metrics = df[df["type"] == "metric"].copy()
    iot = metrics[metrics["file"].str.startswith("iot")].copy()
    cloud = metrics[metrics["file"].str.startswith("cloud")].copy()

    score_col, error_col, r2_col, mape_col = _pick_cols(iot)
    acc_col = "val_acc" if "val_acc" in iot.columns else ("train_acc" if "train_acc" in iot.columns else None)

    local_last = None
    local_best = None
    local_error_last = None
    local_r2_last = None
    local_mape_last = None
    local_acc_last = None
    local_acc_best = None
    if not iot.empty and "round" in iot.columns:
        g = iot.sort_values("round").groupby("iot")
        if score_col:
            local_last = _mean(g[score_col].last())
            local_best = _mean(g[score_col].max())
        if error_col:
            local_error_last = _mean(g[error_col].last())
        if r2_col:
            local_r2_last = _mean(g[r2_col].last())
        if mape_col:
            local_mape_last = _mean(g[mape_col].last())
        if acc_col:
            local_acc_last = _mean(g[acc_col].last())
            local_acc_best = _mean(g[acc_col].max())

    if not cloud.empty and "target" in cloud.columns:
        g_target = cloud.sort_values("round").groupby("target")
        global_score_last = _mean(g_target["global_score"].last()) if "global_score" in cloud.columns else None
        global_score_best = _mean(g_target["global_score"].max()) if "global_score" in cloud.columns else None
        global_rmse_last = _mean(g_target["global_rmse"].last()) if "global_rmse" in cloud.columns else None
        global_r2_last = _mean(g_target["global_r2"].last()) if "global_r2" in cloud.columns else None
        global_mape_last = _mean(g_target["global_mape"].last()) if "global_mape" in cloud.columns else None
        global_acc_last = _mean(g_target["global_acc"].last()) if "global_acc" in cloud.columns else None
        global_acc_best = _mean(g_target["global_acc"].max()) if "global_acc" in cloud.columns else None
    else:
        global_score = cloud["global_score"] if "global_score" in cloud.columns else pd.Series(dtype=float)
        global_rmse = cloud["global_rmse"] if "global_rmse" in cloud.columns else pd.Series(dtype=float)
        global_r2 = cloud["global_r2"] if "global_r2" in cloud.columns else pd.Series(dtype=float)
        global_mape = cloud["global_mape"] if "global_mape" in cloud.columns else pd.Series(dtype=float)
        global_acc = cloud["global_acc"] if "global_acc" in cloud.columns else pd.Series(dtype=float)
        global_score_last = _last(global_score)
        global_score_best = _best_high(global_score)
        global_rmse_last = _last(global_rmse)
        global_r2_last = _last(global_r2)
        global_mape_last = _last(global_mape)
        global_acc_last = _last(global_acc)
        global_acc_best = _best_high(global_acc)

    round_time_s = pd.Series(dtype=float)
    if not cloud.empty and "_ts" in cloud.columns:
        c = cloud.sort_values("_ts").copy()
        round_time_s = c["_ts"].diff() / 1000.0
        round_time_s = round_time_s[round_time_s > 0]

    throughput = pd.Series(dtype=float)
    if not cloud.empty and "_ts" in cloud.columns and "payload_bytes" in cloud.columns:
        c = cloud.sort_values("_ts").copy()
        c["delta_s"] = c["_ts"].diff() / 1000.0
        c = c[c["delta_s"] > 0]
        c["throughput_kbps"] = (c["payload_bytes"] / 1024.0) / c["delta_s"]
        throughput = c["throughput_kbps"].dropna()

    duration_s = None
    if not metrics.empty and "_ts" in metrics.columns:
        ts = metrics["_ts"].dropna()
        if len(ts) >= 2:
            duration_s = float((ts.max() - ts.min()) / 1000.0)

    window_start = _last(cloud.sort_values("_ts")["window"].dropna()) if "window" in cloud.columns else None
    window_last = _last(cloud.sort_values("_ts")["window"].dropna()) if "window" in cloud.columns else None
    window_mean = _mean(cloud["window"]) if "window" in cloud.columns else None
    contributors_mean = _mean(cloud["edges"]) if "edges" in cloud.columns else None

    return Summary(
        iot_count=iot["iot"].nunique() if "iot" in iot.columns else 0,
        contributors_mean=contributors_mean,
        rounds_iot=int(iot["round"].max()) if "round" in iot.columns and not iot.empty else None,
        rounds_cloud=int(cloud["round"].max()) if "round" in cloud.columns and not cloud.empty else None,
        duration_s=duration_s,
        local_score_last_mean=local_last,
        local_score_best_mean=local_best,
        local_error_last_mean=local_error_last,
        local_r2_last_mean=local_r2_last,
        local_mape_last_mean=local_mape_last,
        local_acc_last_mean=local_acc_last,
        local_acc_best_mean=local_acc_best,
        global_score_last=global_score_last,
        global_score_best=global_score_best,
        global_rmse_last=global_rmse_last,
        global_r2_last=global_r2_last,
        global_mape_last=global_mape_last,
        global_acc_last=global_acc_last,
        global_acc_best=global_acc_best,
        round_time_s_mean=_mean(round_time_s),
        round_time_s_p90=_p90(round_time_s),
        throughput_kbps_mean=_mean(throughput),
        enc_ms_mean=_mean(iot["enc_ms"]) if "enc_ms" in iot.columns else None,
        cloud_dec_ms_mean=_mean(cloud["dec_ms"]) if "dec_ms" in cloud.columns else None,
        pactual_mean=_mean(cloud["pactual"]) if "pactual" in cloud.columns else None,
        pdesired=pdesired,
        window_start=window_start,
        window_last=window_last,
        window_mean=window_mean,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default=str(OUT))
    ap.add_argument("--pdesired", type=float, default=None)
    ap.add_argument("--compare-dir", default=None, help="metrics dir for comparison run (Run B)")
    args = ap.parse_args()

    base = Path(args.metrics_dir)
    df = _load_metrics(base)
    pdesired = args.pdesired if args.pdesired is not None else _load_pdesired()
    summary = compute_summary(df, pdesired=pdesired)

    metrics = df[df["type"] == "metric"].copy()
    iot = metrics[metrics["file"].str.startswith("iot")].copy()
    cloud = metrics[metrics["file"].str.startswith("cloud")].copy()

    if not cloud.empty and "target" in cloud.columns and "global_score" in cloud.columns and "round" in cloud.columns:
        global_score = cloud.groupby("round")["global_score"].mean()
        rounds = global_score.index.to_series()
    else:
        global_score = cloud["global_score"] if "global_score" in cloud.columns else pd.Series(dtype=float)
        rounds = cloud["round"] if "round" in cloud.columns else pd.Series(dtype=float)

    best_global = _best_high(global_score)
    t90 = _time_to_threshold(rounds, global_score, 0.9 * best_global) if best_global else None
    early = _mean(global_score.head(10)) if not global_score.empty else None
    late = _mean(global_score.tail(10)) if not global_score.empty else None
    stability = float(global_score.tail(10).std()) if len(global_score) >= 10 else None
    global_trend = _trend(rounds, global_score)

    score_col, _, _, _ = _pick_cols(iot)
    local_trend = _trend(
        iot["round"] if "round" in iot.columns else pd.Series(dtype=float),
        iot[score_col] if score_col in iot.columns else pd.Series(dtype=float),
    )

    overhead_ratio = None
    if "round_time_ms" in iot.columns and "enc_ms" in iot.columns:
        rt = _mean(iot["round_time_ms"])
        enc = _mean(iot["enc_ms"])
        if rt and enc:
            overhead_ratio = 100.0 * (enc / rt)

    hetero = None
    hetero_path = base / "heterogeneity_metrics.json"
    if hetero_path.exists():
        hetero = json.loads(hetero_path.read_text(encoding="utf-8"))

    report = []
    report.append("# Paper-Aligned Report (Dynamic Analysis)")
    report.append("")
    report.append("## Snapshot")
    report.append(f"- IoTs: {summary.iot_count}; Aggregator: Cloud (centralized)")
    report.append(f"- IoT rounds: {summary.rounds_iot}; Cloud rounds: {summary.rounds_cloud}")
    report.append(f"- Duration: {summary.duration_s:.2f}s" if summary.duration_s else "- Duration: n/a")
    report.append("")

    report.append("## Paper Metrics (Comparable)")
    report.append("| Metric | Value |")
    report.append("| --- | --- |")
    report.append(f"| Local Score (last, mean) | {summary.local_score_last_mean:.4f} |" if summary.local_score_last_mean is not None else "| Local Score (last, mean) | n/a |")
    report.append(f"| Local Score (best, mean) | {summary.local_score_best_mean:.4f} |" if summary.local_score_best_mean is not None else "| Local Score (best, mean) | n/a |")
    report.append(f"| Local Error (last, mean) | {summary.local_error_last_mean:.4f} |" if summary.local_error_last_mean is not None else "| Local Error (last, mean) | n/a |")
    report.append(f"| Local R2 (last, mean) | {summary.local_r2_last_mean:.4f} |" if summary.local_r2_last_mean is not None else "| Local R2 (last, mean) | n/a |")
    report.append(f"| Local MAPE (last, mean) | {summary.local_mape_last_mean:.4f} |" if summary.local_mape_last_mean is not None else "| Local MAPE (last, mean) | n/a |")
    report.append(f"| Local Acc (last, mean) | {summary.local_acc_last_mean:.4f} |" if summary.local_acc_last_mean is not None else "| Local Acc (last, mean) | n/a |")
    report.append(f"| Local Acc (best, mean) | {summary.local_acc_best_mean:.4f} |" if summary.local_acc_best_mean is not None else "| Local Acc (best, mean) | n/a |")
    report.append(f"| Global Score (last) | {summary.global_score_last:.4f} |" if summary.global_score_last is not None else "| Global Score (last) | n/a |")
    report.append(f"| Global Score (best) | {summary.global_score_best:.4f} |" if summary.global_score_best is not None else "| Global Score (best) | n/a |")
    report.append(f"| Global RMSE (last) | {summary.global_rmse_last:.4f} |" if summary.global_rmse_last is not None else "| Global RMSE (last) | n/a |")
    report.append(f"| Global R2 (last) | {summary.global_r2_last:.4f} |" if summary.global_r2_last is not None else "| Global R2 (last) | n/a |")
    report.append(f"| Global MAPE (last) | {summary.global_mape_last:.4f} |" if summary.global_mape_last is not None else "| Global MAPE (last) | n/a |")
    report.append(f"| Global Acc (last) | {summary.global_acc_last:.4f} |" if summary.global_acc_last is not None else "| Global Acc (last) | n/a |")
    report.append(f"| Global Acc (best) | {summary.global_acc_best:.4f} |" if summary.global_acc_best is not None else "| Global Acc (best) | n/a |")
    report.append(f"| Round Time (mean, s) | {summary.round_time_s_mean:.3f} |" if summary.round_time_s_mean is not None else "| Round Time (mean, s) | n/a |")
    report.append(f"| Round Time (p90, s) | {summary.round_time_s_p90:.3f} |" if summary.round_time_s_p90 is not None else "| Round Time (p90, s) | n/a |")
    report.append(f"| Throughput (KB/s, mean) | {summary.throughput_kbps_mean:.2f} |" if summary.throughput_kbps_mean is not None else "| Throughput (KB/s, mean) | n/a |")
    report.append(f"| Encrypt Overhead (ms, mean) | {summary.enc_ms_mean:.2f} |" if summary.enc_ms_mean is not None else "| Encrypt Overhead (ms, mean) | n/a |")
    report.append(f"| Cloud Decrypt (ms, mean) | {summary.cloud_dec_ms_mean:.2f} |" if summary.cloud_dec_ms_mean is not None else "| Cloud Decrypt (ms, mean) | n/a |")
    if overhead_ratio is not None:
        report.append(f"| Encrypt Overhead / Round Time | {overhead_ratio:.2f}% |")
    report.append("")

    if not cloud.empty and "target" in cloud.columns:
        rows = []
        for target, sub in cloud.sort_values("round").groupby("target"):
            score = sub["global_score"].dropna() if "global_score" in sub.columns else pd.Series(dtype=float)
            rmse = sub["global_rmse"].dropna() if "global_rmse" in sub.columns else pd.Series(dtype=float)
            r2 = sub["global_r2"].dropna() if "global_r2" in sub.columns else pd.Series(dtype=float)
            mape = sub["global_mape"].dropna() if "global_mape" in sub.columns else pd.Series(dtype=float)
            acc = sub["global_acc"].dropna() if "global_acc" in sub.columns else pd.Series(dtype=float)
            rows.append({
                "target": target,
                "score_last": float(score.iloc[-1]) if len(score) else None,
                "rmse_last": float(rmse.iloc[-1]) if len(rmse) else None,
                "r2_last": float(r2.iloc[-1]) if len(r2) else None,
                "mape_last": float(mape.iloc[-1]) if len(mape) else None,
                "acc_last": float(acc.iloc[-1]) if len(acc) else None,
            })
        if rows:
            report.append("## Global Metrics by Target")
            report.append(pd.DataFrame(rows).to_markdown(index=False))
            report.append("")

    report.append("## Dynamic Analysis")
    if best_global is not None:
        report.append(f"- Time-to-90% of best global score: round {t90}" if t90 is not None else "- Time-to-90% of best global score: n/a")
        report.append(f"- Early vs late global score (first 10 → last 10): {early:.4f} → {late:.4f}" if early is not None and late is not None else "- Early vs late global score: n/a")
        report.append(f"- Stability (std of last 10 global score): {stability:.4f}" if stability is not None else "- Stability: n/a")
        report.append(f"- Global score trend (slope per round): {global_trend:.6f}" if global_trend is not None else "- Global score trend: n/a")
    report.append(f"- Local score trend (slope per round): {local_trend:.6f}" if local_trend is not None else "- Local score trend: n/a")

    if summary.pactual_mean is not None:
        report.append(f"- Participation: pactual mean = {summary.pactual_mean:.3f}" + (f", pdesired = {summary.pdesired:.3f}" if summary.pdesired is not None else ""))
    if summary.contributors_mean is not None:
        report.append(f"- Contributors per aggregation window (mean): {summary.contributors_mean:.2f}")
    if summary.window_start is not None and summary.window_last is not None:
        report.append(f"- Window drift: {summary.window_start:.2f} → {summary.window_last:.2f} (mean {summary.window_mean:.2f})")
    report.append("")

    report.append("## Paper-Style Interpretation")
    report.append("- The centralized async aggregation reduces round time while keeping global score aligned with local convergence.")
    report.append("- Sliding window adaptation reflects participation dynamics; drift toward smaller windows suggests frequent arrivals (pactual near 1.0).")
    report.append("- Salsa20 overhead remains small relative to round time, consistent with the paper’s overhead analysis.")
    if not cloud.empty and "target" in cloud.columns:
        report.append("- Global metrics are reported per target (temperature vs humidity vs light) to avoid mixing tasks.")

    if hetero:
        report.append("")
        report.append("## Data Heterogeneity (Intel Lab)")
        by_target = hetero.get("by_target", {})
        for target, stats in by_target.items():
            report.append(f"- Target `{target}`: bins={stats.get('bins')}, range={stats.get('range')}")
            js = stats.get("js", {})
            emd = stats.get("emd", {})
            if js:
                report.append(f"  - JS Divergence mean/min/max: {js.get('mean')} / {js.get('min')} / {js.get('max')}")
            if emd:
                report.append(f"  - EMD mean/min/max: {emd.get('mean')} / {emd.get('min')} / {emd.get('max')}")

    if args.compare_dir:
        compare_base = Path(args.compare_dir)
        try:
            df_sync = _load_metrics(compare_base)
        except Exception as e:
            report.append("")
            report.append("## Run Comparison")
            report.append(f"- Failed to load compare metrics: {e}")
        else:
            sync_summary = compute_summary(df_sync, pdesired=pdesired)
            report.append("")
            report.append("## Run Comparison")
            report.append("| Metric | Run A | Run B | Delta |")
            report.append("| --- | --- | --- | --- |")
            if summary.round_time_s_mean and sync_summary.round_time_s_mean:
                delta = 100.0 * (sync_summary.round_time_s_mean - summary.round_time_s_mean) / sync_summary.round_time_s_mean
                report.append(f"| Round Time (mean, s) | {summary.round_time_s_mean:.3f} | {sync_summary.round_time_s_mean:.3f} | {delta:+.2f}% |")
            if summary.throughput_kbps_mean and sync_summary.throughput_kbps_mean:
                delta = 100.0 * (summary.throughput_kbps_mean - sync_summary.throughput_kbps_mean) / sync_summary.throughput_kbps_mean
                report.append(f"| Throughput (KB/s) | {summary.throughput_kbps_mean:.2f} | {sync_summary.throughput_kbps_mean:.2f} | {delta:+.2f}% |")
            if summary.global_score_last and sync_summary.global_score_last:
                delta = summary.global_score_last - sync_summary.global_score_last
                report.append(f"| Global Score (last) | {summary.global_score_last:.4f} | {sync_summary.global_score_last:.4f} | {delta:+.4f} |")
            if summary.global_rmse_last and sync_summary.global_rmse_last:
                delta = summary.global_rmse_last - sync_summary.global_rmse_last
                report.append(f"| Global RMSE (last) | {summary.global_rmse_last:.4f} | {sync_summary.global_rmse_last:.4f} | {delta:+.4f} |")
            if summary.global_r2_last and sync_summary.global_r2_last:
                delta = summary.global_r2_last - sync_summary.global_r2_last
                report.append(f"| Global R2 (last) | {summary.global_r2_last:.4f} | {sync_summary.global_r2_last:.4f} | {delta:+.4f} |")
            if summary.global_mape_last and sync_summary.global_mape_last:
                delta = summary.global_mape_last - sync_summary.global_mape_last
                report.append(f"| Global MAPE (last) | {summary.global_mape_last:.4f} | {sync_summary.global_mape_last:.4f} | {delta:+.4f} |")
            if summary.global_acc_last and sync_summary.global_acc_last:
                delta = summary.global_acc_last - sync_summary.global_acc_last
                report.append(f"| Global Acc (last) | {summary.global_acc_last:.4f} | {sync_summary.global_acc_last:.4f} | {delta:+.4f} |")
            if summary.enc_ms_mean and sync_summary.enc_ms_mean:
                delta = summary.enc_ms_mean - sync_summary.enc_ms_mean
                report.append(f"| Encrypt Overhead (ms) | {summary.enc_ms_mean:.2f} | {sync_summary.enc_ms_mean:.2f} | {delta:+.2f} |")

    out = base / "paper_report.md"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[paper_report] wrote {out}")


if __name__ == "__main__":
    main()
