from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from project.analysis.paths import OUT

CSV = OUT / "metrics_all.csv"
PLOTS = OUT / "paper_plots"


def _safe_mean(series):
    if series is None or series.empty:
        return None
    return float(series.mean())


def _write_csv(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def _plot_line(df, x, y, title, filename, hue=None):
    if df.empty:
        return
    plt.figure()
    if hue and hue in df.columns:
        for key, sub in df.groupby(hue):
            plt.plot(sub[x], sub[y], label=str(key))
        plt.legend()
    else:
        plt.plot(df[x], df[y])
    plt.title(title)
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / filename
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    plt.savefig(tmp)
    tmp.replace(out)
    plt.close()


def _plot_bar(df, x, y, title, filename):
    if df.empty:
        return
    plt.figure()
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / filename
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    plt.savefig(tmp)
    tmp.replace(out)
    plt.close()


def main():
    if not CSV.exists():
        print("Run extract_metrics first.")
        return

    df = pd.read_csv(CSV)
    metrics = df[df["type"] == "metric"].copy()

    iot = metrics[metrics["file"].str.startswith("iot")].copy()
    edge = metrics[metrics["file"].str.startswith("edge")].copy()
    cloud = metrics[metrics["file"].str.startswith("cloud")].copy()

    # 1) IoT training time (per device)
    iot_time = pd.DataFrame()
    if not iot.empty and "train_time_ms" in iot.columns:
        iot_time = (
            iot.groupby("iot", dropna=False)["train_time_ms"]
            .mean()
            .reset_index()
            .rename(columns={"train_time_ms": "train_time_ms_mean"})
        )
        _write_csv(iot_time, "paper_iot_train_time.csv")
        _plot_bar(iot_time, "iot", "train_time_ms_mean", "IoT Training Time (ms)", "paper_iot_train_time.png")

    # 2) Global RMSE (cloud)
    cloud_rmse = pd.DataFrame()
    if not cloud.empty and "global_rmse" in cloud.columns:
        cols = ["round", "global_rmse"]
        if "target" in cloud.columns:
            cols.append("target")
        cloud_rmse = cloud[cols].dropna().sort_values("round")
        _write_csv(cloud_rmse, "paper_global_rmse.csv")
        _plot_line(cloud_rmse, "round", "global_rmse", "Global RMSE", "paper_global_rmse.png", hue="target" if "target" in cloud.columns else None)

    # 2b) Global Score (cloud)
    cloud_score = pd.DataFrame()
    if not cloud.empty and "global_score" in cloud.columns:
        cols = ["round", "global_score"]
        if "target" in cloud.columns:
            cols.append("target")
        cloud_score = cloud[cols].dropna().sort_values("round")
        _write_csv(cloud_score, "paper_global_score.csv")
        _plot_line(cloud_score, "round", "global_score", "Global Score", "paper_global_score.png", hue="target" if "target" in cloud.columns else None)

    # 2c) Global R2 / MAPE (cloud)
    cloud_r2 = pd.DataFrame()
    if not cloud.empty and "global_r2" in cloud.columns:
        cols = ["round", "global_r2"]
        if "target" in cloud.columns:
            cols.append("target")
        cloud_r2 = cloud[cols].dropna().sort_values("round")
        _write_csv(cloud_r2, "paper_global_r2.csv")
        _plot_line(cloud_r2, "round", "global_r2", "Global R2", "paper_global_r2.png", hue="target" if "target" in cloud.columns else None)

    cloud_mape = pd.DataFrame()
    if not cloud.empty and "global_mape" in cloud.columns:
        cols = ["round", "global_mape"]
        if "target" in cloud.columns:
            cols.append("target")
        cloud_mape = cloud[cols].dropna().sort_values("round")
        _write_csv(cloud_mape, "paper_global_mape.csv")
        _plot_line(cloud_mape, "round", "global_mape", "Global MAPE", "paper_global_mape.png", hue="target" if "target" in cloud.columns else None)

    # 3) Round time (cloud _ts delta)
    round_time = pd.DataFrame()
    if not cloud.empty and "_ts" in cloud.columns:
        c = cloud.sort_values("_ts").copy()
        c["round_time_s"] = c["_ts"].diff() / 1000.0
        round_time = c[["round", "round_time_s"]].dropna()
        _write_csv(round_time, "paper_round_time.csv")
        _plot_line(round_time, "round", "round_time_s", "Round Time (s)", "paper_round_time.png")

    # 4) Throughput (edge payload_bytes / delta_t)
    throughput = pd.DataFrame()
    if not edge.empty and "_ts" in edge.columns and "payload_bytes" in edge.columns:
        rows = []
        for edge_id, sub in edge.sort_values("_ts").groupby("edge"):
            sub = sub.copy()
            sub["delta_s"] = sub["_ts"].diff() / 1000.0
            sub = sub[sub["delta_s"] > 0]
            sub["throughput_kbps"] = (sub["payload_bytes"] / 1024.0) / sub["delta_s"]
            tmp = sub[["edge", "throughput_kbps"]].dropna()
            rows.append(tmp)
        if rows:
            throughput = pd.concat(rows, ignore_index=True)
            _write_csv(throughput, "paper_throughput.csv")
            _plot_bar(
                throughput.groupby("edge", dropna=False)["throughput_kbps"].mean().reset_index(),
                "edge",
                "throughput_kbps",
                "Throughput (KB/s)",
                "paper_throughput.png",
            )

    # 5) Overhead (enc/dec)
    overhead_rows = []
    if not iot.empty and "enc_ms" in iot.columns:
        overhead_rows.append({"layer": "iot_encrypt_ms", "mean_ms": _safe_mean(iot["enc_ms"])})
    if not edge.empty and "dec_ms_mean" in edge.columns:
        overhead_rows.append({"layer": "edge_decrypt_ms", "mean_ms": _safe_mean(edge["dec_ms_mean"])})
    if not cloud.empty and "dec_ms" in cloud.columns:
        overhead_rows.append({"layer": "cloud_decrypt_ms", "mean_ms": _safe_mean(cloud["dec_ms"])})
    overhead = pd.DataFrame(overhead_rows)
    if not overhead.empty:
        _write_csv(overhead, "paper_overhead.csv")
        _plot_bar(overhead, "layer", "mean_ms", "Crypto Overhead (ms)", "paper_overhead.png")

    # Summary
    summary = {
        "iot_train_time_ms_mean": _safe_mean(iot["train_time_ms"]) if "train_time_ms" in iot.columns else None,
        "global_rmse_last": float(cloud_rmse["global_rmse"].dropna().iloc[-1]) if not cloud_rmse.empty else None,
        "global_score_last": float(cloud_score["global_score"].dropna().iloc[-1]) if not cloud_score.empty else None,
        "global_r2_last": float(cloud_r2["global_r2"].dropna().iloc[-1]) if not cloud_r2.empty else None,
        "global_mape_last": float(cloud_mape["global_mape"].dropna().iloc[-1]) if not cloud_mape.empty else None,
        "round_time_s_mean": _safe_mean(round_time["round_time_s"]) if not round_time.empty else None,
        "throughput_kbps_mean": _safe_mean(throughput["throughput_kbps"]) if not throughput.empty else None,
        "iot_enc_ms_mean": _safe_mean(iot["enc_ms"]) if "enc_ms" in iot.columns else None,
        "edge_dec_ms_mean": _safe_mean(edge["dec_ms_mean"]) if "dec_ms_mean" in edge.columns else None,
        "cloud_dec_ms_mean": _safe_mean(cloud["dec_ms"]) if "dec_ms" in cloud.columns else None,
    }
    summary_df = pd.DataFrame([summary])
    _write_csv(summary_df, "paper_summary.csv")
    print("[paper_metrics] wrote paper_* outputs to", OUT)


if __name__ == "__main__":
    main()
