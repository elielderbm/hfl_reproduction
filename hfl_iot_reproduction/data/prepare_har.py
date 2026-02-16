import os
import zipfile
import requests
import pandas as pd
from pathlib import Path

HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATA_DIR = Path("/workspace/data/har")
ZIP_PATH = DATA_DIR / "UCI_HAR_Dataset.zip"
EXTRACTED = DATA_DIR / "UCI HAR Dataset"
CLIENTS_DIR = Path("/workspace/data/clients")

IOT_IDS = ["iot1", "iot2", "iot3", "iot4"]

def ensure_extracted():
    """Garante que a pasta final UCI HAR Dataset exista"""
    if (EXTRACTED / "train" / "X_train.txt").exists():
        print(f"[data_prep] Dataset já presente em {EXTRACTED}")
        return

    # tenta extrair de qualquer zip que encontrar
    zips = list(DATA_DIR.glob("*.zip"))
    if not zips:
        print("[data_prep] Nenhum .zip local, baixando...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        r = requests.get(HAR_URL)
        with open(ZIP_PATH, "wb") as f:
            f.write(r.content)
        zips = [ZIP_PATH]

    for z in zips:
        print(f"[data_prep] Extraindo {z}...")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(DATA_DIR)

    if not (EXTRACTED / "train" / "X_train.txt").exists():
        raise FileNotFoundError(f"Não encontrei {EXTRACTED}/train/X_train.txt após extração.")

def load_split(split: str):
    root = EXTRACTED
    X = pd.read_csv(root/f"{split}/X_{split}.txt", sep=r"\s+", header=None)
    y = pd.read_csv(root/f"{split}/y_{split}.txt", header=None)
    subj = pd.read_csv(root/f"{split}/subject_{split}.txt", header=None)
    df = X.copy()
    df["label"] = y   # 🔹 nome fixo "label"
    df["subject"] = subj
    return df

def partition_clients(df: pd.DataFrame):
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)

    for iot in IOT_IDS:
        (CLIENTS_DIR / iot).mkdir(parents=True, exist_ok=True)

    subjects = df["subject"].unique()
    subjects.sort()

    mapping = {}
    for i, subj in enumerate(subjects):
        mapping[subj] = IOT_IDS[i % len(IOT_IDS)]

    for iot in IOT_IDS:
        client_df = df[df["subject"].map(lambda s: mapping[s] == iot)]
        out_dir = CLIENTS_DIR / iot
        if client_df.empty:
            pd.DataFrame().to_csv(out_dir/"train.csv", index=False)
            pd.DataFrame().to_csv(out_dir/"val.csv", index=False)
            pd.DataFrame().to_csv(out_dir/"test.csv", index=False)
            print(f"[data_prep] Cliente {iot} não tem dados (CSVs vazios criados).")
            continue

        client_df = client_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(client_df)
        ntrain = int(0.6*n)
        nval   = int(0.2*n)

        train = client_df.iloc[:ntrain]
        val   = client_df.iloc[ntrain:ntrain+nval]
        test  = client_df.iloc[ntrain+nval:]

        train.to_csv(out_dir/"train.csv", index=False)
        val.to_csv(out_dir/"val.csv", index=False)
        test.to_csv(out_dir/"test.csv", index=False)

        print(f"[data_prep] Cliente {iot}: {len(train)}/{len(val)}/{len(test)} instâncias.")

def main():
    ensure_extracted()
    train = load_split("train")
    test = load_split("test")
    df = pd.concat([train, test], ignore_index=True)
    print("[data_prep] Dataset combinado, shape:", df.shape)
    partition_clients(df)

if __name__ == "__main__":
    main()
