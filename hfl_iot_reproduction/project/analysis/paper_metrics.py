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

    # 2) Global accuracy (cloud)
    cloud_acc = pd.DataFrame()
    if not cloud.empty and "global_acc" in cloud.columns:
        cloud_acc = cloud[["round", "global_acc"]].dropna().sort_values("round")
        _write_csv(cloud_acc, "paper_global_accuracy.csv")
        _plot_line(cloud_acc, "round", "global_acc", "Global Accuracy", "paper_global_accuracy.png")

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
        "global_acc_last": float(cloud_acc["global_acc"].dropna().iloc[-1]) if not cloud_acc.empty else None,
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
