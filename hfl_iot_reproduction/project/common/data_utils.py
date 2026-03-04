from pathlib import Path
import os
import json

import numpy as np
import pandas as pd

from project.common.dataset_config import load_dataset_config, task_for_target


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _prepare_split(df: pd.DataFrame, target: str | None = None):
    cfg = load_dataset_config()
    input_dim = int(cfg.get("input_dim", 0)) or 1

    if df.empty:
        return np.zeros((0, input_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if "y" not in df.columns:
        raise KeyError("Nenhuma coluna de alvo encontrada (esperado: 'y')")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = pd.to_numeric(df["y"], errors="coerce").astype("float32")
    X = df.drop(columns=["y"], errors="ignore").values.astype("float32")

    if X.shape[1] != input_dim:
        # fallback: ajusta input_dim a partir do CSV
        input_dim = X.shape[1]

    y_out = y.to_numpy()
    if task_for_target(target, cfg) == "classification":
        # ensure binary 0/1 (defensive for legacy data)
        y_out = (y_out >= 0.5).astype("float32")

    return X, y_out


def load_client_split(iot_id: str):
    base = Path(f"/workspace/data/clients/{iot_id}")
    train = _read_csv(base / "train.csv")
    val = _read_csv(base / "val.csv")
    test = _read_csv(base / "test.csv")

    cfg = load_dataset_config()
    target = (cfg.get("targets") or {}).get(iot_id)
    Xtr, ytr = _prepare_split(train, target)
    Xv, yv = _prepare_split(val, target)
    Xt, yt = _prepare_split(test, target)

    return (Xtr, ytr), (Xv, yv), (Xt, yt)


def load_global_test():
    cfg = load_dataset_config()
    input_dim = int(cfg.get("input_dim", 0)) or 1

    base = Path("/workspace/data/clients")
    if not base.exists():
        raise FileNotFoundError("Clientes não encontrados em /workspace/data/clients")

    Xs = []
    ys = []
    for iot_dir in sorted(base.iterdir()):
        if not iot_dir.is_dir():
            continue
        df = _read_csv(iot_dir / "test.csv")
        target = (cfg.get("targets") or {}).get(iot_dir.name)
        X, y = _prepare_split(df, target)
        if len(y) == 0:
            continue
        Xs.append(X)
        ys.append(y)

    if not Xs:
        return np.zeros((0, input_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.vstack(Xs), np.concatenate(ys)


def load_global_test_by_target():
    cfg = load_dataset_config()
    targets = cfg.get("targets", {})
    base = Path("/workspace/data/clients")
    groups: dict[str, dict[str, list[np.ndarray]]] = {}

    if not base.exists():
        return {}

    for iot_dir in sorted(base.iterdir()):
        if not iot_dir.is_dir():
            continue
        iot = iot_dir.name
        target = targets.get(iot)
        if not target:
            continue
        df = _read_csv(iot_dir / "test.csv")
        X, y = _prepare_split(df, target)
        if len(y) == 0:
            continue
        bucket = groups.setdefault(target, {"X": [], "y": []})
        bucket["X"].append(X)
        bucket["y"].append(y)

    out = {}
    for target, vals in groups.items():
        if not vals["X"]:
            continue
        out[target] = (np.vstack(vals["X"]), np.concatenate(vals["y"]))
    return out


def _scaler_path() -> Path:
    cfg = load_dataset_config()
    dataset = cfg.get("dataset", "intel_lab")
    return Path(f"/workspace/data/{dataset}/scaler.json")


def load_target_scaler() -> dict | None:
    path = _scaler_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_classification_thresholds() -> dict:
    scaler = load_target_scaler() or {}
    classification = scaler.get("classification") or {}
    thresholds = classification.get("thresholds") or {}
    return {k: float(v) for k, v in thresholds.items() if v is not None}


def unscale_target(y: np.ndarray, target: str | None):
    cfg = load_dataset_config()
    if task_for_target(target, cfg) == "classification":
        return y
    scaler = load_target_scaler()
    if not scaler or "target" not in scaler:
        return y
    target_map = scaler.get("target") or {}
    stats = target_map.get(target)
    if stats is None and target_map:
        # fallback: primeiro target disponível
        stats = next(iter(target_map.values()))
    if not stats:
        return y
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0))
    return y * std + mean


def load_split_by_target(target: str, split: str = "train", max_samples: int | None = None, shuffle: bool = True):
    cfg = load_dataset_config()
    targets = cfg.get("targets", {})
    base = Path("/workspace/data/clients")
    if not base.exists():
        return np.zeros((0, cfg.get("input_dim", 1)), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    Xs = []
    ys = []
    for iot_dir in sorted(base.iterdir()):
        if not iot_dir.is_dir():
            continue
        iot = iot_dir.name
        if targets.get(iot) != target:
            continue
        df = _read_csv(iot_dir / f"{split}.csv")
        X, y = _prepare_split(df, target)
        if len(y) == 0:
            continue
        Xs.append(X)
        ys.append(y)

    if not Xs:
        return np.zeros((0, cfg.get("input_dim", 1)), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    if shuffle:
        rng = np.random.default_rng(int(os.getenv("GLOBAL_SEED", "42")))
        idx = rng.permutation(len(y))
        X = X[idx]
        y = y[idx]

    if max_samples is not None and max_samples > 0 and len(y) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    return X, y
