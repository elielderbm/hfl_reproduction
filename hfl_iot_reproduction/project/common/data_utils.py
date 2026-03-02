from pathlib import Path

import numpy as np
import pandas as pd

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _clean_labels(df: pd.DataFrame):
    if df.empty:
        return np.zeros((0, 561), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    # Coluna pode ser "y" ou "label"
    if "y" in df.columns:
        labels = df["y"]
    elif "label" in df.columns:
        labels = df["label"]
    else:
        raise KeyError("Nenhuma coluna de label encontrada (esperado: 'y' ou 'label')")

    y = pd.to_numeric(labels, errors="coerce").fillna(-1).astype("int32") - 1
    mask = (y >= 0) & (y < 6)
    X = df.drop(columns=["y", "label", "subject"], errors="ignore").values[mask]
    return X, y[mask]


def load_client_split(iot_id: str):
    base = Path(f"/workspace/data/clients/{iot_id}")
    train = _read_csv(base / "train.csv")
    val = _read_csv(base / "val.csv")
    test = _read_csv(base / "test.csv")

    Xtr, ytr = _clean_labels(train)
    Xv, yv = _clean_labels(val)
    Xt, yt = _clean_labels(test)

    return (Xtr, ytr), (Xv, yv), (Xt, yt)


def load_global_test():
    base = Path("/workspace/data/clients")
    if not base.exists():
        raise FileNotFoundError("Clientes não encontrados em /workspace/data/clients")

    Xs = []
    ys = []
    for iot_dir in sorted(base.iterdir()):
        if not iot_dir.is_dir():
            continue
        df = _read_csv(iot_dir / "test.csv")
        X, y = _clean_labels(df)
        if len(y) == 0:
            continue
        Xs.append(X)
        ys.append(y)

    if not Xs:
        return np.zeros((0, 561), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    return np.vstack(Xs), np.concatenate(ys)
