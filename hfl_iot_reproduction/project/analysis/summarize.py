from pathlib import Path
import pandas as pd

OUT = Path("/workspace/outputs")
CSV = OUT/"metrics_all.csv"

def main():
    if not CSV.exists():
        print("Run extract_metrics first.")
        return

    df = pd.read_csv(CSV)

    # IoT summary
    iot = df[(df["type"]=="metric") & (df["file"].str.startswith("iot"))]
    if not iot.empty:
        agg_dict = {
            "rounds": ("round","max"),
            "val_acc_mean": ("val_acc","mean"),
            "val_acc_last": ("val_acc","last"),
        }
        # 🔹 só adiciona round_time_ms se existir
        if "round_time_ms" in iot.columns:
            agg_dict["round_time_ms_mean"] = ("round_time_ms","mean")

        g = iot.groupby("iot").agg(**agg_dict)
        print("== IoT Summary ==")
        print(g.to_string())
        g.to_csv(OUT/"iot_summary.csv")

    # Edge summary
    edge = df[(df["type"]=="metric") & (df["file"].str.startswith("edge"))]
    if not edge.empty:
        agg_dict = {}
        if "window" in edge.columns:
            agg_dict["window_mean"] = ("window","mean")
        if "pactual" in edge.columns:
            agg_dict["pactual_mean"] = ("pactual","mean")

        g = edge.groupby("edge").agg(**agg_dict)
        print("\n== Edge Summary ==")
        print(g.to_string())
        g.to_csv(OUT/"edge_summary.csv")

    # Cloud summary
    cloud = df[(df["type"]=="metric") & (df["file"].str.startswith("cloud"))]
    if not cloud.empty and "round" in cloud.columns:
        mx = cloud["round"].max()
        print(f"\n== Cloud Summary ==\nrounds: {mx}")

if __name__ == "__main__":
    main()
