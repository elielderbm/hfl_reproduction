from __future__ import annotations

import json
from itertools import combinations
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

CLIENTS_DIR = Path("/workspace/data/clients")
OUT = Path("/workspace/outputs/heterogeneity_metrics.json")


def _load_labels(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=int)
    df = pd.read_csv(path)
    if "label" in df.columns:
        labels = df["label"]
    elif "y" in df.columns:
        labels = df["y"]
    else:
        return np.array([], dtype=int)
    y = pd.to_numeric(labels, errors="coerce").dropna().astype(int)
    y = y[(y >= 1) & (y <= 6)]
    return y.to_numpy()


def _client_distribution(iot: str) -> np.ndarray:
    labels = []
    for split in ("train", "val", "test"):
        labels.append(_load_labels(CLIENTS_DIR / iot / f"{split}.csv"))
    if not labels:
        return np.zeros(6, dtype=float)
    y = np.concatenate(labels) if any(len(x) for x in labels) else np.array([], dtype=int)
    if len(y) == 0:
        return np.zeros(6, dtype=float)
    counts = np.bincount(y - 1, minlength=6).astype(float)
    return counts / counts.sum()


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
        raise FileNotFoundError(f"Clientes não encontrados em {CLIENTS_DIR}. Execute data/prepare_har.py primeiro.")

    dists = {}
    entropies = {}
    for iot_dir in sorted(CLIENTS_DIR.iterdir()):
        if not iot_dir.is_dir():
            continue
        iot = iot_dir.name
        p = _client_distribution(iot)
        dists[iot] = p.tolist()
        entropies[iot] = _entropy(p) if p.sum() > 0 else None

    js_vals = []
    emd_vals = []
    for (a, pa), (b, pb) in combinations(dists.items(), 2):
        p = np.array(pa, dtype=float)
        q = np.array(pb, dtype=float)
        if p.sum() == 0 or q.sum() == 0:
            continue
        js_vals.append(_js(p, q))
        emd_vals.append(_emd(p, q))

    summary = {
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
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[heterogeneity] entropy per client:")
    for k, v in entropies.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: n/a")
    print("[heterogeneity] JS mean/min/max:", summary["js"])
    print("[heterogeneity] EMD mean/min/max:", summary["emd"])
    print(f"[heterogeneity] wrote {OUT}")


if __name__ == "__main__":
    main()
