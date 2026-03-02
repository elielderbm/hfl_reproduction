from __future__ import annotations

import json
import sys
from itertools import combinations
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

# Garante que /workspace esteja no sys.path quando executado via script direto
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.common.dataset_config import load_dataset_config

CLIENTS_DIR = Path("/workspace/data/clients")
OUT = Path("/workspace/outputs/heterogeneity_metrics.json")


def _load_targets(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=float)
    df = pd.read_csv(path)
    if "y" not in df.columns:
        return np.array([], dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").dropna().astype(float)
    return y.to_numpy()


def _client_values(iot: str) -> np.ndarray:
    values = []
    for split in ("train", "val", "test"):
        values.append(_load_targets(CLIENTS_DIR / iot / f"{split}.csv"))
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values) if any(len(v) for v in values) else np.array([], dtype=float)


def _entropy(p: np.ndarray) -> float:
    return float(-np.sum([v * log(v) for v in p if v > 0.0]))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    out = 0.0
    for pi, qi in zip(p, q):
        if pi > 0 and qi > 0:
            out += pi * log(pi / qi)
    return float(out)


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _emd(p: np.ndarray, q: np.ndarray) -> float:
    # 1D Earth Mover's Distance for equal-width bins
    return float(np.sum(np.abs(np.cumsum(p - q))))


def main():
    if not CLIENTS_DIR.exists():
        raise FileNotFoundError(f"Clientes não encontrados em {CLIENTS_DIR}. Execute data/prepare_intel_lab.py primeiro.")

    cfg = load_dataset_config()
    targets = cfg.get("targets", {})
    bins = int(cfg.get("heterogeneity_bins", 20))

    per_client = {}
    grouped = {}
    for iot_dir in sorted(CLIENTS_DIR.iterdir()):
        if not iot_dir.is_dir():
            continue
        iot = iot_dir.name
        y = _client_values(iot)
        target = targets.get(iot, "unknown")
        grouped.setdefault(target, {})[iot] = y
        per_client[iot] = {
            "target": target,
            "count": int(len(y)),
            "mean": float(np.mean(y)) if len(y) else None,
            "std": float(np.std(y)) if len(y) else None,
            "min": float(np.min(y)) if len(y) else None,
            "max": float(np.max(y)) if len(y) else None,
        }

    summary = {"per_client": per_client, "by_target": {}}

    for target, series_map in grouped.items():
        values = [v for v in series_map.values() if len(v) > 0]
        if not values:
            summary["by_target"][target] = {"entropy": {}, "js": None, "emd": None}
            continue

        all_vals = np.concatenate(values)
        vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
        if vmin == vmax:
            summary["by_target"][target] = {"entropy": {}, "js": None, "emd": None}
            continue

        edges = np.linspace(vmin, vmax, bins + 1)
        dists = {}
        entropies = {}
        for iot, vals in series_map.items():
            if len(vals) == 0:
                continue
            hist, _ = np.histogram(vals, bins=edges)
            p = hist.astype(float)
            p = p / p.sum() if p.sum() > 0 else p
            dists[iot] = p
            entropies[iot] = _entropy(p) if p.sum() > 0 else None

        js_vals = []
        emd_vals = []
        for (a, pa), (b, pb) in combinations(dists.items(), 2):
            js_vals.append(_js(pa, pb))
            emd_vals.append(_emd(pa, pb))

        summary["by_target"][target] = {
            "entropy": entropies,
            "js": {
                "mean": float(np.mean(js_vals)) if js_vals else None,
                "min": float(np.min(js_vals)) if js_vals else None,
                "max": float(np.max(js_vals)) if js_vals else None,
            },
            "emd": {
                "mean": float(np.mean(emd_vals)) if emd_vals else None,
                "min": float(np.min(emd_vals)) if emd_vals else None,
                "max": float(np.max(emd_vals)) if emd_vals else None,
            },
            "bins": bins,
            "range": {"min": vmin, "max": vmax},
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[heterogeneity] per-client stats:")
    for k, v in per_client.items():
        print(f"  {k}: count={v['count']} mean={v['mean']} std={v['std']}")
    print(f"[heterogeneity] wrote {OUT}")


if __name__ == "__main__":
    main()
