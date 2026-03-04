from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yaml

# Garante que /workspace esteja no sys.path quando executado via script direto
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.common.dataset_config import load_dataset_config

DATA_URL = "https://db.csail.mit.edu/labdata/data.txt.gz"
DATA_DIR = Path("/workspace/data/intel_lab")
RAW_GZ = DATA_DIR / "data.txt.gz"
RAW_TXT = DATA_DIR / "data.txt"
CLIENTS_DIR = Path("/workspace/data/clients")
CLIENTS_MAP_PATH = Path("/workspace/config/clients.yml")
META_PATH = DATA_DIR / "meta.json"
RESOLVED_CLIENTS_PATH = DATA_DIR / "clients_resolved.yml"
SCALER_PATH = DATA_DIR / "scaler.json"

IOT_IDS = ["iot1", "iot2", "iot3", "iot4", "iot5", "iot6"]

VALID_RANGES = {
    "temp": (-40.0, 80.0),
    "humidity": (0.0, 100.0),
    "light": (0.0, 1.0e6),
    "voltage": (0.0, 10.0),
}


def ensure_raw():
    if RAW_TXT.exists():
        print(f"[data_prep] Dataset já presente em {RAW_TXT}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_GZ.exists():
        print("[data_prep] Baixando Intel Lab Sensor Data...")
        with requests.get(DATA_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(RAW_GZ, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    print(f"[data_prep] Extraindo {RAW_GZ}...")
    with gzip.open(RAW_GZ, "rb") as src, open(RAW_TXT, "wb") as dst:
        dst.write(src.read())


def _parse_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    ncols = df.shape[1]
    if ncols >= 8:
        df = df.iloc[:, :8]
        df.columns = ["date", "time", "epoch", "moteid", "temp", "humidity", "light", "voltage"]
    elif ncols == 7:
        # Fallback: sem coluna de tensão (voltage)
        df.columns = ["date", "time", "epoch", "moteid", "temp", "humidity", "light"]
    else:
        raise ValueError(f"Formato inesperado com {ncols} colunas no arquivo {path}")

    # Conversões numéricas
    for col in ("epoch", "moteid", "temp", "humidity", "light", "voltage"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Limpeza básica
    df = df.replace([np.inf, -np.inf], np.nan)
    required = ["epoch", "moteid", "temp", "humidity", "light"]
    if "voltage" in df.columns:
        required.append("voltage")
    df = df.dropna(subset=required)

    # Filtra por intervalos plausíveis
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]

    df["moteid"] = df["moteid"].astype(int)
    df = df.sort_values("epoch")
    return df


def _load_clients_map(df: pd.DataFrame) -> Dict[str, int]:
    if not CLIENTS_MAP_PATH.exists():
        raise FileNotFoundError(f"Arquivo de mapeamento não encontrado: {CLIENTS_MAP_PATH}")
    with open(CLIENTS_MAP_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    mapping: Dict[str, int] = {}
    auto = []
    for iot in IOT_IDS:
        v = raw.get(iot)
        if v is None:
            auto.append(iot)
            continue
        if isinstance(v, str) and v.strip().lower() == "auto":
            auto.append(iot)
            continue
        mapping[iot] = int(v)

    if auto:
        counts = df["moteid"].value_counts().sort_values(ascending=False)
        available = [int(mid) for mid in counts.index if int(mid) not in mapping.values()]
        for iot in auto:
            if not available:
                raise ValueError("Não há motes suficientes para auto-atribuição")
            mapping[iot] = available.pop(0)

    # Persistir mapa resolvido para rastreabilidade
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESOLVED_CLIENTS_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(mapping, f, sort_keys=True)

    return mapping


def _build_windows(df: pd.DataFrame, features: List[str], target: str, window: int, delta: int) -> Tuple[np.ndarray, np.ndarray]:
    values = df[features].to_numpy(dtype=np.float32)
    target_vals = df[target].to_numpy(dtype=np.float32)
    n = len(df)
    n_samples = n - window - delta + 1
    if n_samples <= 0:
        return np.zeros((0, window * len(features)), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    X = np.empty((n_samples, window * len(features)), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.float32)
    for i in range(n_samples):
        end = i + window
        X[i] = values[i:end].reshape(-1)
        y[i] = target_vals[end - 1 + delta]
    return X, y


def _split_time(X: np.ndarray, y: np.ndarray, splits: dict) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    ntrain = int(n * splits.get("train", 0.6))
    nval = int(n * splits.get("val", 0.2))
    train = (X[:ntrain], y[:ntrain])
    val = (X[ntrain:ntrain + nval], y[ntrain:ntrain + nval])
    test = (X[ntrain + nval:], y[ntrain + nval:])
    return train, val, test


def _scale_features(train_sets, val_sets, test_sets, method: str):
    if method != "standard":
        return train_sets, val_sets, test_sets, None

    Xtr_all = [x for (x, _) in train_sets if x.size > 0]
    if not Xtr_all:
        return train_sets, val_sets, test_sets, None

    Xtr = np.vstack(Xtr_all)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std == 0] = 1.0

    def _apply(split_sets):
        out = []
        for X, y in split_sets:
            if X.size == 0:
                out.append((X, y))
            else:
                out.append(((X - mean) / std, y))
        return out

    return _apply(train_sets), _apply(val_sets), _apply(test_sets), {"mean": mean, "std": std}


def _scale_targets(train_sets, val_sets, test_sets, targets_by_iot, tasks_by_target):
    # Escalonamento por target (temp/umidade)
    by_target = {}
    for (X, y), target in zip(train_sets, targets_by_iot):
        if tasks_by_target.get(target, "regression") == "classification":
            continue
        if y.size == 0:
            continue
        by_target.setdefault(target, []).append(y)

    if not by_target:
        return train_sets, val_sets, test_sets, None

    scalers = {}
    for target, ys in by_target.items():
        ytr = np.concatenate(ys)
        mean = float(ytr.mean())
        std = float(ytr.std()) if float(ytr.std()) != 0.0 else 1.0
        scalers[target] = {"mean": mean, "std": std}

    def _apply(split_sets):
        out = []
        for (X, y), target in zip(split_sets, targets_by_iot):
            if y.size == 0:
                out.append((X, y))
                continue
            if tasks_by_target.get(target, "regression") == "classification":
                out.append((X, y))
                continue
            scaler = scalers.get(target)
            if scaler is None:
                out.append((X, y))
            else:
                out.append((X, (y - scaler["mean"]) / scaler["std"]))
        return out

    return _apply(train_sets), _apply(val_sets), _apply(test_sets), scalers


def _resolve_threshold(raw, values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if raw is None:
        return float(np.median(values))
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower()
    if s in ("median", "mediana"):
        return float(np.median(values))
    if s in ("mean", "media"):
        return float(np.mean(values))
    try:
        return float(s)
    except ValueError:
        return float(np.median(values))


def _binarize(y: np.ndarray, threshold: float) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float32)
    return (y >= threshold).astype(np.float32)


def main():
    cfg = load_dataset_config()
    features = cfg["features"]
    targets = cfg["targets"]
    tasks = cfg.get("tasks", {})
    classification_cfg = cfg.get("classification", {}) or {}
    window = int(cfg["window_size"])
    delta = int(cfg["delta_steps"])
    splits = cfg.get("splits", {"train": 0.6, "val": 0.2, "test": 0.2})
    scaling = cfg.get("scaling", {"method": "standard", "with_target": False})
    if window <= 0 or delta < 0:
        raise ValueError("window_size deve ser > 0 e delta_steps >= 0")
    if splits.get("train", 0) + splits.get("val", 0) > 1.0:
        raise ValueError("splits inválidos: train + val deve ser <= 1.0")

    ensure_raw()
    df = _parse_raw(RAW_TXT)
    print(f"[data_prep] Intel Lab carregado: {df.shape} linhas/colunas")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features ausentes no dataset: {missing}")
    missing_targets = {t for t in targets.values() if t not in df.columns}
    if missing_targets:
        raise ValueError(f"Alvos ausentes no dataset: {sorted(missing_targets)}")

    mapping = _load_clients_map(df)

    # Preparar por IoT
    train_sets = []
    val_sets = []
    test_sets = []
    per_iot = {}
    class_targets = {t for t, task in tasks.items() if str(task).strip().lower() == "classification"}
    class_train_by_target: dict[str, list[np.ndarray]] = {}

    for iot in IOT_IDS:
        mote_id = mapping.get(iot)
        target = targets.get(iot, "temp")
        task = tasks.get(target, "regression")
        if mote_id is None:
            print(f"[data_prep] IoT {iot} sem mote_id definido.")
            X = np.zeros((0, window * len(features)), dtype=np.float32)
            y = np.zeros((0,), dtype=np.float32)
        else:
            sub = df[df["moteid"] == mote_id].copy()
            sub = sub.sort_values("epoch")
            if sub.empty:
                X = np.zeros((0, window * len(features)), dtype=np.float32)
                y = np.zeros((0,), dtype=np.float32)
                print(f"[data_prep] IoT {iot} (mote={mote_id}) sem dados válidos.")
            else:
                X, y = _build_windows(sub, features, target, window, delta)
                print(f"[data_prep] IoT {iot} (mote={mote_id}, target={target}) janelas: {len(X)}")

        (Xtr, ytr), (Xv, yv), (Xt, yt) = _split_time(X, y, splits)
        train_sets.append((Xtr, ytr))
        val_sets.append((Xv, yv))
        test_sets.append((Xt, yt))
        if target in class_targets and ytr.size > 0:
            class_train_by_target.setdefault(target, []).append(ytr)
        per_iot[iot] = {
            "mote_id": mote_id,
            "target": target,
            "task": str(task).strip().lower(),
            "n_windows": int(len(X)),
            "n_train": int(len(Xtr)),
            "n_val": int(len(Xv)),
            "n_test": int(len(Xt)),
        }

    thresholds = {}
    for target in class_targets:
        ys = class_train_by_target.get(target, [])
        if not ys:
            continue
        y_all = np.concatenate(ys)
        thresh_cfg = (classification_cfg.get(target) or {}).get("threshold", "median")
        thresholds[target] = _resolve_threshold(thresh_cfg, y_all)

    if thresholds:
        for i, iot in enumerate(IOT_IDS):
            target = per_iot[iot]["target"]
            thr = thresholds.get(target)
            if thr is None:
                continue
            Xtr, ytr = train_sets[i]
            Xv, yv = val_sets[i]
            Xt, yt = test_sets[i]
            train_sets[i] = (Xtr, _binarize(ytr, thr))
            val_sets[i] = (Xv, _binarize(yv, thr))
            test_sets[i] = (Xt, _binarize(yt, thr))

    # Escalonamento (features)
    train_sets, val_sets, test_sets, scaler = _scale_features(
        train_sets, val_sets, test_sets, method=scaling.get("method", "standard")
    )
    target_scaler = None
    if scaling.get("with_target"):
        targets_by_iot = [per_iot[iot]["target"] for iot in IOT_IDS]
        train_sets, val_sets, test_sets, target_scaler = _scale_targets(
            train_sets, val_sets, test_sets, targets_by_iot, tasks
        )

    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, iot in enumerate(IOT_IDS):
        out_dir = CLIENTS_DIR / iot
        out_dir.mkdir(parents=True, exist_ok=True)

        def _to_df(X, y):
            if X.size == 0:
                cols = [f"x{i}" for i in range(window * len(features))] + ["y"]
                return pd.DataFrame(columns=cols)
            cols = [f"x{i}" for i in range(X.shape[1])] + ["y"]
            return pd.DataFrame(np.column_stack([X, y]), columns=cols)

        _to_df(*train_sets[i]).to_csv(out_dir / "train.csv", index=False)
        _to_df(*val_sets[i]).to_csv(out_dir / "val.csv", index=False)
        _to_df(*test_sets[i]).to_csv(out_dir / "test.csv", index=False)

        print(
            f"[data_prep] Cliente {iot}: {per_iot[iot]['n_train']}/"
            f"{per_iot[iot]['n_val']}/{per_iot[iot]['n_test']} amostras (train/val/test)."
        )

    # Salvar meta
    meta = {
        "dataset": cfg.get("dataset", "intel_lab"),
        "download_url": DATA_URL,
        "window_size": window,
        "delta_steps": delta,
        "features": features,
        "targets": targets,
        "tasks": tasks,
        "classification": classification_cfg,
        "thresholds": thresholds,
        "splits": splits,
        "scaling": scaling,
        "clients": per_iot,
        "input_dim": window * len(features),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if scaler is not None or target_scaler is not None:
        scaler_payload = {}
        if scaler is not None:
            scaler_payload["features"] = {"mean": scaler["mean"].tolist(), "std": scaler["std"].tolist()}
        if target_scaler is not None:
            scaler_payload["target"] = target_scaler
        if thresholds:
            scaler_payload["classification"] = {"thresholds": thresholds}
        SCALER_PATH.write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")
        print(f"[data_prep] Scaler salvo em {SCALER_PATH}")

    print(f"[data_prep] Meta salvo em {META_PATH}")


if __name__ == "__main__":
    main()
