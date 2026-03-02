import pandas as pd

from project.analysis.explain_results import generate_results_explanation
from project.analysis.paths import OUT

CSV = OUT / "metrics_all.csv"

def _write_csv_replace(df: pd.DataFrame, target, index: bool = True):
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(target)

def main():
    if not CSV.exists():
        print("Run extract_metrics first.")
        return

    df = pd.read_csv(CSV)

    # IoT summary
    iot = df[(df["type"] == "metric") & (df["file"].str.startswith("iot"))].copy()
    if not iot.empty:
        acc_col = "train_acc" if "train_acc" in iot.columns else "val_acc"
        loss_col = "train_loss" if "train_loss" in iot.columns else "val_loss"
        rows = []
        for iot_id, sub in iot.groupby("iot"):
            sub = sub.sort_values("round")
            rows.append(
                {
                    "iot": iot_id,
                    "rounds": int(sub["round"].max()) if "round" in sub.columns else len(sub),
                    "acc_mean": float(sub[acc_col].mean()) if acc_col in sub.columns else None,
                    "acc_first": float(sub[acc_col].dropna().iloc[0]) if acc_col in sub.columns and sub[acc_col].notna().any() else None,
                    "acc_last": float(sub[acc_col].dropna().iloc[-1]) if acc_col in sub.columns and sub[acc_col].notna().any() else None,
                    "acc_best": float(sub[acc_col].max()) if acc_col in sub.columns else None,
                    "loss_last": float(sub[loss_col].dropna().iloc[-1]) if loss_col in sub.columns and sub[loss_col].notna().any() else None,
                }
            )
        g = pd.DataFrame(rows).set_index("iot").sort_index()

        # only include round_time_ms when available
        if "round_time_ms" in iot.columns:
            rt = iot.groupby("iot", dropna=False)["round_time_ms"].mean()
            g["round_time_ms_mean"] = rt
            if "payload_bytes" in iot.columns:
                pb = iot.groupby("iot", dropna=False)["payload_bytes"].mean()
                g["throughput_kbps_mean"] = (pb / 1024.0) / (rt / 1000.0)

        print("== IoT Summary ==")
        print(g.to_string())
        _write_csv_replace(g, OUT / "iot_summary.csv")

    # Edge summary
    edge = df[(df["type"] == "metric") & (df["file"].str.startswith("edge"))].copy()
    if not edge.empty:
        rows = []
        for edge_id, sub in edge.groupby("edge"):
            sub = sub.sort_values("_ts")
            row = {"edge": edge_id}
            if "window" in sub.columns and sub["window"].notna().any():
                row["window_start"] = float(sub["window"].dropna().iloc[0])
                row["window_last"] = float(sub["window"].dropna().iloc[-1])
                row["window_mean"] = float(sub["window"].mean())
            if "pactual" in sub.columns and sub["pactual"].notna().any():
                row["pactual_mean"] = float(sub["pactual"].mean())
                row["pactual_half_ratio"] = float((sub["pactual"] == 0.5).mean())
                row["pactual_full_ratio"] = float((sub["pactual"] == 1.0).mean())
            if "qcurrent" in sub.columns and sub["qcurrent"].notna().any():
                row["qcurrent_mean"] = float(sub["qcurrent"].mean())
            if "buf" in sub.columns and sub["buf"].notna().any():
                row["buf_mean"] = float(sub["buf"].mean())
            rows.append(row)

        g = pd.DataFrame(rows).set_index("edge").sort_index()
        print("\n== Edge Summary ==")
        print(g.to_string())
        _write_csv_replace(g, OUT / "edge_summary.csv")

    # Cloud summary
    cloud = df[(df["type"] == "metric") & (df["file"].str.startswith("cloud"))].copy()
    if not cloud.empty:
        row = {
            "rows": len(cloud),
            "round_min": int(cloud["round"].min()) if "round" in cloud.columns and cloud["round"].notna().any() else None,
            "round_max": int(cloud["round"].max()) if "round" in cloud.columns and cloud["round"].notna().any() else None,
            "edges_mean": float(cloud["edges"].mean()) if "edges" in cloud.columns and cloud["edges"].notna().any() else None,
            "edges_min": float(cloud["edges"].min()) if "edges" in cloud.columns and cloud["edges"].notna().any() else None,
            "edges_max": float(cloud["edges"].max()) if "edges" in cloud.columns and cloud["edges"].notna().any() else None,
            "beta_cloud_mode": float(cloud["beta_cloud"].dropna().mode().iloc[0]) if "beta_cloud" in cloud.columns and cloud["beta_cloud"].notna().any() else None,
            "global_acc_last": float(cloud["global_acc"].dropna().iloc[-1]) if "global_acc" in cloud.columns and cloud["global_acc"].notna().any() else None,
            "global_acc_best": float(cloud["global_acc"].max()) if "global_acc" in cloud.columns else None,
        }
        g = pd.DataFrame([row])
        print("\n== Cloud Summary ==")
        print(g.to_string(index=False))
        _write_csv_replace(g, OUT / "cloud_summary.csv", index=False)

    report_path = generate_results_explanation(df)
    print(f"\n[analysis] wrote {report_path}")

if __name__ == "__main__":
    main()
