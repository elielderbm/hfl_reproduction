import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/workspace/outputs")
OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT/"metrics_all.csv"

def plot():
    df = pd.read_csv(CSV)
    # IoT curves: val_acc/val_loss by round
    iot = df[(df["type"]=="metric") & (df["file"].str.startswith("iot"))].copy()
    if not iot.empty and "round" in iot:
        g = iot.groupby(["iot","round"]).mean(numeric_only=True).reset_index()
        for key, sub in g.groupby("iot"):
            plt.figure()
            sub.plot(x="round", y="val_acc")
            plt.title(f"IoT {key} - Val Accuracy")
            plt.savefig(OUT/f"iot_{key}_val_acc.png"); plt.close()

            plt.figure()
            sub.plot(x="round", y="val_loss")
            plt.title(f"IoT {key} - Val Loss")
            plt.savefig(OUT/f"iot_{key}_val_loss.png"); plt.close()

            # 🔹 Só plota round_time_ms se existir
            if "round_time_ms" in sub.columns:
                plt.figure()
                sub.plot(x="round", y="round_time_ms")
                plt.title(f"IoT {key} - Round Time (ms)")
                plt.savefig(OUT/f"iot_{key}_round_time.png"); plt.close()

    # Edge: window and pactual over time
    edge = df[(df["type"]=="metric") & (df["file"].str.startswith("edge"))].copy()
    if not edge.empty:
        for key, sub in edge.groupby("edge"):
            plt.figure()
            sub.plot(y=["window","pactual"])
            plt.title(f"Edge {key} - Window & Participation")
            plt.savefig(OUT/f"edge_{key}_win_p.png"); plt.close()

    # Cloud: round vs edges
    cloud = df[(df["type"]=="metric") & (df["file"].str.startswith("cloud"))].copy()
    if not cloud.empty:
        plt.figure()
        cloud.plot(x="round", y="edges")
        plt.title("Cloud - Edges contributing")
        plt.savefig(OUT/f"cloud_edges.png"); plt.close()

    print("[analysis] plots saved to outputs/*.png")

if __name__ == "__main__":
    plot()
