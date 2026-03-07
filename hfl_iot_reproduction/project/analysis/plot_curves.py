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
    # IoT curves: score and error by round
    iot = df[(df["type"]=="metric") & (df["file"].str.startswith("iot"))].copy()
    if not iot.empty and "round" in iot:
        g = iot.groupby(["iot","round"]).mean(numeric_only=True).reset_index()
        if "train_score" in g.columns:
            score_col = "train_score"
        elif "val_score" in g.columns:
            score_col = "val_score"
        elif "train_acc" in g.columns:
            score_col = "train_acc"
        elif "val_acc" in g.columns:
            score_col = "val_acc"
        else:
            score_col = None
        error_col = None
        error_label = None
        if "train_rmse" in g.columns:
            error_col = "train_rmse"
            error_label = "RMSE"
        elif "train_loss" in g.columns:
            error_col = "train_loss"
            error_label = "Loss (MSE)"
        elif "val_rmse" in g.columns:
            error_col = "val_rmse"
            error_label = "RMSE"
        elif "val_loss" in g.columns:
            error_col = "val_loss"
            error_label = "Loss (MSE)"

        for key, sub in g.groupby("iot"):
            if score_col:
                plt.figure()
                sub.plot(x="round", y=score_col)
                plt.title(f"IoT {key} - Score")
                _save_current_fig(OUT/f"iot_{key}_score.png"); plt.close()

            if error_col:
                plt.figure()
                sub.plot(x="round", y=error_col)
                plt.title(f"IoT {key} - {error_label}")
                _save_current_fig(OUT/f"iot_{key}_error.png"); plt.close()

            r2_col = "train_r2" if "train_r2" in sub.columns else ("val_r2" if "val_r2" in sub.columns else None)
            if r2_col:
                plt.figure()
                sub.plot(x="round", y=r2_col)
                plt.title(f"IoT {key} - R2")
                _save_current_fig(OUT/f"iot_{key}_r2.png"); plt.close()

            mape_col = "train_mape" if "train_mape" in sub.columns else ("val_mape" if "val_mape" in sub.columns else None)
            if mape_col:
                plt.figure()
                sub.plot(x="round", y=mape_col)
                plt.title(f"IoT {key} - MAPE")
                _save_current_fig(OUT/f"iot_{key}_mape.png"); plt.close()

            acc_col = "val_acc" if "val_acc" in sub.columns else ("train_acc" if "train_acc" in sub.columns else None)
            if acc_col:
                plt.figure()
                sub.plot(x="round", y=acc_col)
                plt.title(f"IoT {key} - Accuracy")
                _save_current_fig(OUT/f"iot_{key}_acc.png"); plt.close()

            if "distill_rmse" in sub.columns:
                plt.figure()
                sub.plot(x="round", y="distill_rmse")
                plt.title(f"IoT {key} - Distill RMSE")
                _save_current_fig(OUT/f"iot_{key}_distill_rmse.png"); plt.close()

            # 🔹 Só plota round_time_ms se existir
            if "round_time_ms" in sub.columns:
                plt.figure()
                sub.plot(x="round", y="round_time_ms")
                plt.title(f"IoT {key} - Round Time (ms)")
                _save_current_fig(OUT/f"iot_{key}_round_time.png"); plt.close()

    # Cloud: window/pactual and global error if available
    cloud = df[(df["type"]=="metric") & (df["file"].str.startswith("cloud"))].copy()
    if not cloud.empty:
        if "window" in cloud.columns and "pactual" in cloud.columns:
            plt.figure()
            cloud.plot(x="round", y=["window","pactual"])
            plt.title("Cloud - Window & Participation (IoTs)")
            _save_current_fig(OUT/"cloud_window_pactual.png"); plt.close()

        if "edges" in cloud.columns:
            plt.figure()
            cloud.plot(x="round", y="edges")
            plt.title("Cloud - IoTs contributing")
            _save_current_fig(OUT/"cloud_iots.png"); plt.close()

        if "target" in cloud.columns:
            for t, sub in cloud.groupby("target"):
                if "global_rmse" in sub.columns:
                    plt.figure()
                    sub.plot(x="round", y="global_rmse")
                    plt.title(f"Cloud - Global RMSE ({t})")
                    _save_current_fig(OUT/f"cloud_global_rmse_{t}.png"); plt.close()
                if "global_score" in sub.columns:
                    plt.figure()
                    sub.plot(x="round", y="global_score")
                    plt.title(f"Cloud - Global Score ({t})")
                    _save_current_fig(OUT/f"cloud_global_score_{t}.png"); plt.close()
                if "global_acc" in sub.columns:
                    plt.figure()
                    sub.plot(x="round", y="global_acc")
                    plt.title(f"Cloud - Global Acc ({t})")
                    _save_current_fig(OUT/f"cloud_global_acc_{t}.png"); plt.close()
                if "global_r2" in sub.columns:
                    plt.figure()
                    sub.plot(x="round", y="global_r2")
                    plt.title(f"Cloud - Global R2 ({t})")
                    _save_current_fig(OUT/f"cloud_global_r2_{t}.png"); plt.close()
                if "global_mape" in sub.columns:
                    plt.figure()
                    sub.plot(x="round", y="global_mape")
                    plt.title(f"Cloud - Global MAPE ({t})")
                    _save_current_fig(OUT/f"cloud_global_mape_{t}.png"); plt.close()
        else:
            if "global_rmse" in cloud.columns:
                plt.figure()
                cloud.plot(x="round", y="global_rmse")
                plt.title("Cloud - Global RMSE")
                _save_current_fig(OUT/f"cloud_global_rmse.png"); plt.close()

            if "global_score" in cloud.columns:
                plt.figure()
                cloud.plot(x="round", y="global_score")
                plt.title("Cloud - Global Score")
                _save_current_fig(OUT/f"cloud_global_score.png"); plt.close()

            if "global_acc" in cloud.columns:
                plt.figure()
                cloud.plot(x="round", y="global_acc")
                plt.title("Cloud - Global Acc")
                _save_current_fig(OUT/f"cloud_global_acc.png"); plt.close()

            if "global_r2" in cloud.columns:
                plt.figure()
                cloud.plot(x="round", y="global_r2")
                plt.title("Cloud - Global R2")
                _save_current_fig(OUT/f"cloud_global_r2.png"); plt.close()

        if "global_mape" in cloud.columns:
            plt.figure()
            cloud.plot(x="round", y="global_mape")
            plt.title("Cloud - Global MAPE")
            _save_current_fig(OUT/f"cloud_global_mape.png"); plt.close()
        if "server_ft_rmse" in cloud.columns:
            plt.figure()
            cloud.plot(x="round", y="server_ft_rmse")
            plt.title("Cloud - Server FT RMSE")
            _save_current_fig(OUT/f"cloud_server_ft_rmse.png"); plt.close()
        if "server_ft_score" in cloud.columns:
            plt.figure()
            cloud.plot(x="round", y="server_ft_score")
            plt.title("Cloud - Server FT Score")
            _save_current_fig(OUT/f"cloud_server_ft_score.png"); plt.close()

        # Per-target curves (global_<target>_rmse/score)
        for col in cloud.columns:
            if col in ("global_rmse", "global_score"):
                continue
            if col.startswith("global_") and col.endswith("_rmse"):
                plt.figure()
                cloud.plot(x="round", y=col)
                plt.title(f"Cloud - {col.replace('global_', '').replace('_', ' ').upper()} RMSE")
                _save_current_fig(OUT/f"cloud_{col}.png"); plt.close()
            if col.startswith("global_") and col.endswith("_score"):
                plt.figure()
                cloud.plot(x="round", y=col)
                plt.title(f"Cloud - {col.replace('global_', '').replace('_', ' ').upper()} Score")
                _save_current_fig(OUT/f"cloud_{col}.png"); plt.close()
            if col.startswith("global_") and col.endswith("_r2"):
                plt.figure()
                cloud.plot(x="round", y=col)
                plt.title(f"Cloud - {col.replace('global_', '').replace('_', ' ').upper()} R2")
                _save_current_fig(OUT/f"cloud_{col}.png"); plt.close()
            if col.startswith("global_") and col.endswith("_mape"):
                plt.figure()
                cloud.plot(x="round", y=col)
                plt.title(f"Cloud - {col.replace('global_', '').replace('_', ' ').upper()} MAPE")
                _save_current_fig(OUT/f"cloud_{col}.png"); plt.close()
            if col.startswith("global_") and col.endswith("_acc"):
                plt.figure()
                cloud.plot(x="round", y=col)
                plt.title(f"Cloud - {col.replace('global_', '').replace('_', ' ').upper()} ACC")
                _save_current_fig(OUT/f"cloud_{col}.png"); plt.close()
            if col.endswith("_teacher_rmse"):
                plt.figure()
                cloud.plot(x="round", y=col)
                plt.title(f"Cloud - {col.replace('_teacher_rmse','').upper()} Teacher RMSE")
                _save_current_fig(OUT/f"cloud_{col}.png"); plt.close()
            if col.endswith("_teacher_score"):
                plt.figure()
                cloud.plot(x="round", y=col)
                plt.title(f"Cloud - {col.replace('_teacher_score','').upper()} Teacher Score")
                _save_current_fig(OUT/f"cloud_{col}.png"); plt.close()

    print("[analysis] plots saved to outputs/*.png")

if __name__ == "__main__":
    plot()
