import pandas as pd
import matplotlib.pyplot as plt
from project.analysis.paths import OUT

OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT/"metrics_all.csv"

def _save_current_fig(path):
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    plt.savefig(tmp)
    tmp.replace(path)

def plot():
    if not CSV.exists():
        print("Run extract_metrics first.")
        return

    df = pd.read_csv(CSV)
    # IoT curves: train_acc/train_loss by round (fallback para val_*)
    iot = df[(df["type"]=="metric") & (df["file"].str.startswith("iot"))].copy()
    if not iot.empty and "round" in iot:
        g = iot.groupby(["iot","round"]).mean(numeric_only=True).reset_index()
        acc_col = "train_acc" if "train_acc" in g.columns else "val_acc"
        loss_col = "train_loss" if "train_loss" in g.columns else "val_loss"
        for key, sub in g.groupby("iot"):
            plt.figure()
            sub.plot(x="round", y=acc_col)
            plt.title(f"IoT {key} - Accuracy")
            _save_current_fig(OUT/f"iot_{key}_acc.png"); plt.close()

            plt.figure()
            sub.plot(x="round", y=loss_col)
            plt.title(f"IoT {key} - Loss")
            _save_current_fig(OUT/f"iot_{key}_loss.png"); plt.close()

            # 🔹 Só plota round_time_ms se existir
            if "round_time_ms" in sub.columns:
                plt.figure()
                sub.plot(x="round", y="round_time_ms")
                plt.title(f"IoT {key} - Round Time (ms)")
                _save_current_fig(OUT/f"iot_{key}_round_time.png"); plt.close()

    # Edge: window and pactual over time
    edge = df[(df["type"]=="metric") & (df["file"].str.startswith("edge"))].copy()
    if not edge.empty:
        for key, sub in edge.groupby("edge"):
            plt.figure()
            sub.plot(y=["window","pactual"])
            plt.title(f"Edge {key} - Window & Participation")
            _save_current_fig(OUT/f"edge_{key}_win_p.png"); plt.close()

    # Cloud: round vs edges
    cloud = df[(df["type"]=="metric") & (df["file"].str.startswith("cloud"))].copy()
    if not cloud.empty:
        plt.figure()
        cloud.plot(x="round", y="edges")
        plt.title("Cloud - Edges contributing")
        _save_current_fig(OUT/f"cloud_edges.png"); plt.close()

    print("[analysis] plots saved to outputs/*.png")

if __name__ == "__main__":
    plot()
