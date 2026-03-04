from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project.analysis.paths import OUT, DATA, CONFIG
from project.common.data_utils import load_client_split, unscale_target
from project.common.dataset_config import load_dataset_config, task_for_target
from project.common.model import build_model, compile_model, set_weights_vector
from project.common.metrics import regression_metrics, classification_metrics, score_from_rmse


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower()).strip("_")


def _save_fig(path: Path):
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    plt.savefig(tmp)
    tmp.replace(path)
    plt.close()


def _load_weights(weights_dir: Path, every: int = 1):
    files = sorted(weights_dir.glob("cloud_*.npy"))
    for path in files:
        stem = path.stem
        m = re.search(r"cloud_([a-zA-Z0-9_]+)_round_(\\d+)", stem)
        if m:
            target = m.group(1)
            round_id = int(m.group(2))
        else:
            m = re.search(r"cloud_round_(\\d+)", stem)
            if not m:
                continue
            target = "global"
            round_id = int(m.group(1))
        if every > 1 and round_id % every != 0:
            continue
        yield round_id, target, path


def _build_target_sets(targets_map: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for iot, target in targets_map.items():
        (Xtr, ytr), (Xv, yv), (Xt, yt) = load_client_split(iot)
        if len(yt) == 0:
            continue
        bucket = out.setdefault(target, [])
        bucket.append((Xt, yt))

    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for target, parts in out.items():
        Xs = [p[0] for p in parts]
        ys = [p[1] for p in parts]
        merged[target] = (np.vstack(Xs), np.concatenate(ys))
    return merged


def _summarize_target(df: pd.DataFrame, window: int) -> pd.DataFrame:
    rows = []
    for target, sub in df.groupby("target"):
        sub = sub.sort_values("round")
        row = {
            "target": target,
            "rounds": int(sub["round"].max()) if "round" in sub.columns else len(sub),
            "score_first": float(sub["score"].iloc[0]),
            "score_last": float(sub["score"].iloc[-1]),
            "score_best": float(sub["score"].max()),
            "rmse_first": float(sub["rmse"].iloc[0]),
            "rmse_last": float(sub["rmse"].iloc[-1]),
            "rmse_best": float(sub["rmse"].min()),
            "r2_last": float(sub["r2"].iloc[-1]) if "r2" in sub.columns else None,
            "mape_last": float(sub["mape"].iloc[-1]) if "mape" in sub.columns else None,
        }
        if len(sub) >= 2 * window:
            row["score_delta_last_window"] = float(sub["score"].tail(window).mean() - sub["score"].iloc[-2 * window:-window].mean())
            row["rmse_delta_last_window"] = float(sub["rmse"].tail(window).mean() - sub["rmse"].iloc[-2 * window:-window].mean())
        else:
            row["score_delta_last_window"] = None
            row["rmse_delta_last_window"] = None
        rows.append(row)
    return pd.DataFrame(rows).set_index("target").sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default=str(OUT))
    ap.add_argument("--weights-dir", default=str(OUT / "weights"))
    ap.add_argument("--every", type=int, default=1, help="evaluate every N rounds")
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        print(f"[target_report] missing {weights_dir}. Enable SAVE_GLOBAL_WEIGHTS=1 and rerun.")
        return

    cfg = load_dataset_config()
    targets_map = cfg.get("targets", {})
    target_sets = _build_target_sets(targets_map)
    if not target_sets:
        print("[target_report] no target test sets found.")
        return

    models = {}

    rows = []
    for round_id, target, path in _load_weights(weights_dir, every=args.every):
        if target not in target_sets:
            continue
        w = np.load(path)
        task_name = task_for_target(target, cfg)
        model = models.get(task_name)
        if model is None:
            model = build_model(kind="student", task=task_name)
            compile_model(model, lr=0.01, loss="binary_crossentropy" if task_name == "classification" else "mse", task=task_name)
            models[task_name] = model
        set_weights_vector(model, w)
        Xt, yt = target_sets[target]
        preds = model.predict(Xt, verbose=0).reshape(-1)
        if task_name == "classification":
            m = classification_metrics(yt, preds)
            rows.append({
                "round": round_id,
                "target": target,
                "rmse": m.get("brier_rmse"),
                "mae": m.get("mae"),
                "score": m.get("acc"),
                "loss": m.get("bce"),
                "acc": m.get("acc"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
            })
        else:
            yt_u = unscale_target(yt, target)
            preds_u = unscale_target(preds, target)
            mse, mae, rmse, r2, mape = regression_metrics(yt_u, preds_u)
            score = score_from_rmse(rmse)
            rows.append({
                "round": round_id,
                "target": target,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape": mape,
                "score": score,
            })

    if not rows:
        print("[target_report] no metrics computed.")
        return

    df = pd.DataFrame(rows)
    csv_path = metrics_dir / "target_round_metrics.csv"
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv_path)

    summary = _summarize_target(df, args.window)
    report_path = metrics_dir / "target_report.md"

    lines = []
    lines.append("# Target-Specific Cloud Evaluation")
    lines.append("")
    lines.append("This report evaluates the **global model** separately on each target (e.g., temp vs humidity vs light).")
    lines.append("It uses **saved global weights per round** (enable `SAVE_GLOBAL_WEIGHTS=1`).")
    lines.append("For multi-target runs, weights are stored as `cloud_<target>_round_XXXX.npy`.")
    lines.append("")
    lines.append("## Summary (per target)")
    lines.append(summary.to_markdown())
    lines.append("")

    # Target comparison (last round)
    if len(summary) >= 2:
        last_scores = summary["score_last"].sort_values(ascending=False)
        last_rmses = summary["rmse_last"].sort_values(ascending=True)
        lines.append("## Cross-Target Comparison (Last Round)")
        lines.append(f"- Best score: `{last_scores.index[0]}` ({last_scores.iloc[0]:.4f})")
        lines.append(f"- Lowest RMSE: `{last_rmses.index[0]}` ({last_rmses.iloc[0]:.4f})")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Plots
    for target, sub in df.groupby("target"):
        sub = sub.sort_values("round")
        tslug = _slug(target)
        if "rmse" in sub.columns:
            plt.figure(); plt.plot(sub["round"], sub["rmse"]); plt.title(f"{target} RMSE (Cloud)")
            _save_fig(metrics_dir / f"target_{tslug}_rmse.png")
        if "score" in sub.columns:
            plt.figure(); plt.plot(sub["round"], sub["score"]); plt.title(f"{target} Score (Cloud)")
            _save_fig(metrics_dir / f"target_{tslug}_score.png")
        if "r2" in sub.columns:
            plt.figure(); plt.plot(sub["round"], sub["r2"]); plt.title(f"{target} R2 (Cloud)")
            _save_fig(metrics_dir / f"target_{tslug}_r2.png")
        if "mape" in sub.columns:
            plt.figure(); plt.plot(sub["round"], sub["mape"]); plt.title(f"{target} MAPE (Cloud)")
            _save_fig(metrics_dir / f"target_{tslug}_mape.png")

    print(f"[target_report] wrote {report_path}")
    print(f"[target_report] wrote {csv_path}")


if __name__ == "__main__":
    main()
