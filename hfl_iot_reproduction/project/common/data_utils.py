import pandas as pd
from pathlib import Path

def load_client_split(iot_id: str):
    base = Path(f"/workspace/data/clients/{iot_id}")
    train = pd.read_csv(base / "train.csv")
    val   = pd.read_csv(base / "val.csv")
    test  = pd.read_csv(base / "test.csv")

    def clean_labels(df):
        # Coluna pode ser "y" ou "label"
        if "y" in df.columns:
            labels = df["y"]
        elif "label" in df.columns:
            labels = df["label"]
        else:
            raise KeyError("Nenhuma coluna de label encontrada (esperado: 'y' ou 'label')")

        y = pd.to_numeric(labels, errors="coerce").fillna(-1).astype("int32") - 1
        mask = (y >= 0) & (y < 6)

        # remove colunas extras
        X = df.drop(columns=["y", "label", "subject"], errors="ignore").values[mask]

        return X, y[mask]

    Xtr, ytr = clean_labels(train)
    Xv,  yv  = clean_labels(val)
    Xt,  yt  = clean_labels(test)

    return (Xtr, ytr), (Xv, yv), (Xt, yt)
