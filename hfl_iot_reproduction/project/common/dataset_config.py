from __future__ import annotations

import os
from pathlib import Path

import yaml

DATASET_CFG_PATH = Path(os.getenv("DATASET_CONFIG", "/workspace/config/dataset.yml"))

DEFAULTS = {
    "dataset": "intel_lab",
    "window_size": 24,
    "delta_steps": 1,
    "features": ["temp", "humidity", "light", "voltage"],
    "targets": {
        "iot1": "temp",
        "iot2": "temp",
        "iot3": "humidity",
        "iot4": "humidity",
    },
    "tasks": {
        "temp": "regression",
        "humidity": "regression",
        "light": "regression",
    },
    "classification": {},
    "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
    "scaling": {"method": "standard", "with_target": False},
    "heterogeneity_bins": 20,
}


def load_dataset_config() -> dict:
    if not DATASET_CFG_PATH.exists():
        return _with_derived(DEFAULTS.copy())
    with open(DATASET_CFG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = DEFAULTS.copy()
    cfg.update({k: v for k, v in raw.items() if v is not None})

    # nested merges
    splits = DEFAULTS["splits"].copy()
    splits.update(raw.get("splits") or {})
    cfg["splits"] = splits

    scaling = DEFAULTS["scaling"].copy()
    scaling.update(raw.get("scaling") or {})
    cfg["scaling"] = scaling

    targets = DEFAULTS["targets"].copy()
    targets.update(raw.get("targets") or {})
    cfg["targets"] = targets

    tasks = DEFAULTS["tasks"].copy()
    tasks.update(raw.get("tasks") or {})
    cfg["tasks"] = tasks

    classification = DEFAULTS["classification"].copy()
    classification.update(raw.get("classification") or {})
    cfg["classification"] = classification

    return _with_derived(cfg)


def _with_derived(cfg: dict) -> dict:
    features = cfg.get("features") or []
    window_size = int(cfg.get("window_size", DEFAULTS["window_size"]))
    cfg["window_size"] = window_size
    cfg["delta_steps"] = int(cfg.get("delta_steps", DEFAULTS["delta_steps"]))
    cfg["n_features"] = int(len(features)) if features else 0
    cfg["input_dim"] = int(window_size) * int(len(features)) if features else 0
    # normalize task names
    tasks = cfg.get("tasks") or {}
    cfg["tasks"] = {k: str(v).strip().lower() for k, v in tasks.items() if v is not None}
    return cfg


def task_for_target(target: str | None, cfg: dict | None = None) -> str:
    if cfg is None:
        cfg = load_dataset_config()
    tasks = cfg.get("tasks") or {}
    if not target:
        return "regression"
    return str(tasks.get(target, "regression")).strip().lower()


def task_for_iot(iot: str, cfg: dict | None = None) -> str:
    if cfg is None:
        cfg = load_dataset_config()
    targets = cfg.get("targets", {})
    target = targets.get(iot)
    return task_for_target(target, cfg)
