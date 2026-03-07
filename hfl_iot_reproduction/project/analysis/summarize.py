import pandas as pd

from project.analysis.explain_results import generate_results_explanation
from project.analysis.paths import OUT

CSV = OUT / "metrics_all.csv"


def _write_csv_replace(df: pd.DataFrame, target, index: bool = True):
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(target)


def _pick_cols(df: pd.DataFrame):
    score_col = None
    error_col = None
    r2_col = None
    mape_col = None
    acc_col = None

    if "train_score" in df.columns:
        score_col = "train_score"
    elif "train_acc" in df.columns:
        score_col = "train_acc"
    elif "val_score" in df.columns:
        score_col = "val_score"
    elif "val_acc" in df.columns:
        score_col = "val_acc"

    if "train_rmse" in df.columns:
        error_col = "train_rmse"
    elif "train_loss" in df.columns:
        error_col = "train_loss"
    elif "val_rmse" in df.columns:
        error_col = "val_rmse"
    elif "val_loss" in df.columns:
        error_col = "val_loss"

    if "train_r2" in df.columns:
        r2_col = "train_r2"
    elif "val_r2" in df.columns:
        r2_col = "val_r2"

    if "train_mape" in df.columns:
        mape_col = "train_mape"
    elif "val_mape" in df.columns:
        mape_col = "val_mape"

    if "val_acc" in df.columns:
        acc_col = "val_acc"
    elif "train_acc" in df.columns:
        acc_col = "train_acc"

    return score_col, error_col, r2_col, mape_col, acc_col


def main():
    if not CSV.exists():
        print("Run extract_metrics first.")
        return

    df = pd.read_csv(CSV)

    # IoT summary
    iot = df[(df["type"] == "metric") & (df["file"].str.startswith("iot"))].copy()
    if not iot.empty:
        score_col, error_col, r2_col, mape_col, acc_col = _pick_cols(iot)
        rows = []
        for iot_id, sub in iot.groupby("iot"):
            sub = sub.sort_values("round")
            rounds = int(sub["round"].max()) if "round" in sub.columns else len(sub)

            score = sub[score_col].dropna() if score_col in sub.columns else pd.Series(dtype=float)
            error = sub[error_col].dropna() if error_col in sub.columns else pd.Series(dtype=float)
            r2 = sub[r2_col].dropna() if r2_col and r2_col in sub.columns else pd.Series(dtype=float)
            mape = sub[mape_col].dropna() if mape_col and mape_col in sub.columns else pd.Series(dtype=float)
            acc = sub[acc_col].dropna() if acc_col and acc_col in sub.columns else pd.Series(dtype=float)
            distill = sub["distill_rmse"].dropna() if "distill_rmse" in sub.columns else pd.Series(dtype=float)

            rows.append(
                {
                    "iot": iot_id,
                    "target": sub["target"].dropna().iloc[-1] if "target" in sub.columns and sub["target"].notna().any() else None,
                    "rounds": rounds,
                    "score_mean": float(score.mean()) if len(score) else None,
                    "score_first": float(score.iloc[0]) if len(score) else None,
                    "score_last": float(score.iloc[-1]) if len(score) else None,
                    "score_best": float(score.max()) if len(score) else None,
                    "error_last": float(error.iloc[-1]) if len(error) else None,
                    "error_best": float(error.min()) if len(error) else None,
                    "r2_last": float(r2.iloc[-1]) if len(r2) else None,
                    "r2_best": float(r2.max()) if len(r2) else None,
                    "mape_last": float(mape.iloc[-1]) if len(mape) else None,
                    "mape_best": float(mape.min()) if len(mape) else None,
                    "acc_last": float(acc.iloc[-1]) if len(acc) else None,
                    "acc_best": float(acc.max()) if len(acc) else None,
                    "distill_rmse_last": float(distill.iloc[-1]) if len(distill) else None,
                    "distill_rmse_best": float(distill.min()) if len(distill) else None,
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

    # Cloud aggregation summary (centralized async)
    cloud = df[(df["type"] == "metric") & (df["file"].str.startswith("cloud"))].copy()
    if not cloud.empty:
        rows = []
        groups = cloud.groupby("target") if "target" in cloud.columns else [("global", cloud)]
        for target, sub in groups:
            sub = sub.sort_values("_ts")
            row = {"target": target}
            if "window" in sub.columns and sub["window"].notna().any():
                row["window_start"] = float(sub["window"].dropna().iloc[0])
                row["window_last"] = float(sub["window"].dropna().iloc[-1])
                row["window_mean"] = float(sub["window"].mean())
                row["window_min"] = float(sub["window"].min())
                row["window_max"] = float(sub["window"].max())
            if "pactual" in sub.columns and sub["pactual"].notna().any():
                row["pactual_mean"] = float(sub["pactual"].mean())
                row["pactual_half_ratio"] = float((sub["pactual"] == 0.5).mean())
                row["pactual_full_ratio"] = float((sub["pactual"] == 1.0).mean())
            if "edges" in sub.columns and sub["edges"].notna().any():
                row["contributors_mean"] = float(sub["edges"].mean())
                row["contributors_min"] = float(sub["edges"].min())
                row["contributors_max"] = float(sub["edges"].max())
            if "buf" in sub.columns and sub["buf"].notna().any():
                row["buf_mean"] = float(sub["buf"].mean())
            rows.append(row)

        g = pd.DataFrame(rows).set_index("target").sort_index()
        print("\n== Cloud Aggregation Summary ==")
        print(g.to_string())
        _write_csv_replace(g, OUT / "cloud_agg_summary.csv")

    # Cloud summary
    if not cloud.empty:
        rows = []
        groups = cloud.groupby("target") if "target" in cloud.columns else [("global", cloud)]
        for target, sub in groups:
            row = {
                "target": target,
                "rows": len(sub),
                "round_min": int(sub["round"].min()) if "round" in sub.columns and sub["round"].notna().any() else None,
                "round_max": int(sub["round"].max()) if "round" in sub.columns and sub["round"].notna().any() else None,
                "edges_mean": float(sub["edges"].mean()) if "edges" in sub.columns and sub["edges"].notna().any() else None,
                "edges_min": float(sub["edges"].min()) if "edges" in sub.columns and sub["edges"].notna().any() else None,
                "edges_max": float(sub["edges"].max()) if "edges" in sub.columns and sub["edges"].notna().any() else None,
                "beta_cloud_mode": float(sub["beta_cloud"].dropna().mode().iloc[0]) if "beta_cloud" in sub.columns and sub["beta_cloud"].notna().any() else None,
                "global_score_last": float(sub["global_score"].dropna().iloc[-1]) if "global_score" in sub.columns and sub["global_score"].notna().any() else None,
                "global_score_best": float(sub["global_score"].max()) if "global_score" in sub.columns else None,
                "global_rmse_last": float(sub["global_rmse"].dropna().iloc[-1]) if "global_rmse" in sub.columns and sub["global_rmse"].notna().any() else None,
                "global_acc_last": float(sub["global_acc"].dropna().iloc[-1]) if "global_acc" in sub.columns and sub["global_acc"].notna().any() else None,
                "global_acc_best": float(sub["global_acc"].max()) if "global_acc" in sub.columns else None,
                "global_r2_last": float(sub["global_r2"].dropna().iloc[-1]) if "global_r2" in sub.columns and sub["global_r2"].notna().any() else None,
                "global_mape_last": float(sub["global_mape"].dropna().iloc[-1]) if "global_mape" in sub.columns and sub["global_mape"].notna().any() else None,
                "server_ft_rmse_last": float(sub["server_ft_rmse"].dropna().iloc[-1]) if "server_ft_rmse" in sub.columns and sub["server_ft_rmse"].notna().any() else None,
                "server_ft_score_last": float(sub["server_ft_score"].dropna().iloc[-1]) if "server_ft_score" in sub.columns and sub["server_ft_score"].notna().any() else None,
            }
            rows.append(row)
        g = pd.DataFrame(rows)
        print("\n== Cloud Summary ==")
        print(g.to_string(index=False))
        _write_csv_replace(g, OUT / "cloud_summary.csv", index=False)

    report_path = generate_results_explanation(df)
    print(f"\n[analysis] wrote {report_path}")


if __name__ == "__main__":
    main()
