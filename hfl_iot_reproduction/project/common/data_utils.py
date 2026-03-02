from pathlib import Path

import numpy as np
import pandas as pd

from project.common.dataset_config import load_dataset_config


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _prepare_regression(df: pd.DataFrame):
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

    return X, y.to_numpy()


def load_client_split(iot_id: str):
    base = Path(f"/workspace/data/clients/{iot_id}")
    train = _read_csv(base / "train.csv")
    val = _read_csv(base / "val.csv")
    test = _read_csv(base / "test.csv")

    Xtr, ytr = _prepare_regression(train)
    Xv, yv = _prepare_regression(val)
    Xt, yt = _prepare_regression(test)

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
        X, y = _prepare_regression(df)
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
        X, y = _prepare_regression(df)
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
