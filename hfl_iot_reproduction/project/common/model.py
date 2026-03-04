import os
from typing import Iterable

import numpy as np
import tensorflow as tf

from project.common.dataset_config import load_dataset_config


def _parse_int_list(raw: str | None, default: Iterable[int]) -> list[int]:
    if raw is None:
        return list(default)
    s = str(raw).strip()
    if not s:
        return list(default)
    out: list[int] = []
    for chunk in s.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError:
            continue
    return out or list(default)


def _input_shape(input_dim: int | None) -> tuple[int, int, int]:
    cfg = load_dataset_config()
    window = int(cfg.get("window_size", 1)) or 1
    n_features = int(len(cfg.get("features") or [])) or 1
    if input_dim is None or input_dim <= 0:
        input_dim = window * n_features
    if input_dim <= 0:
        input_dim = 1
    if window * n_features != input_dim:
        if input_dim % window == 0:
            n_features = max(1, int(input_dim // window))
        else:
            window = 1
            n_features = input_dim
    return window, n_features, input_dim


def _tcn_config(kind: str) -> dict:
    kind = (kind or "student").strip().lower()
    if kind in ("teacher", "cloud_teacher"):
        base = {
            "filters": int(os.getenv("CLOUD_TCN_FILTERS", os.getenv("TCN_TEACHER_FILTERS", "32"))),
            "kernel": int(os.getenv("CLOUD_TCN_KERNEL", os.getenv("TCN_TEACHER_KERNEL", "3"))),
            "dilations": _parse_int_list(os.getenv("CLOUD_TCN_DILATIONS", os.getenv("TCN_TEACHER_DILATIONS")), [1, 2, 4, 8]),
            "dropout": float(os.getenv("CLOUD_TCN_DROPOUT", os.getenv("TCN_TEACHER_DROPOUT", "0.2"))),
            "dense": int(os.getenv("CLOUD_TCN_DENSE", os.getenv("TCN_TEACHER_DENSE", "64"))),
        }
    elif kind == "edge_teacher":
        base = {
            "filters": int(os.getenv("EDGE_TCN_FILTERS", "32")),
            "kernel": int(os.getenv("EDGE_TCN_KERNEL", "3")),
            "dilations": _parse_int_list(os.getenv("EDGE_TCN_DILATIONS"), [1, 2, 4, 8]),
            "dropout": float(os.getenv("EDGE_TCN_DROPOUT", "0.2")),
            "dense": int(os.getenv("EDGE_TCN_DENSE", "64")),
        }
    else:
        base = {
            "filters": int(os.getenv("STUDENT_TCN_FILTERS", os.getenv("TCN_STUDENT_FILTERS", "16"))),
            "kernel": int(os.getenv("STUDENT_TCN_KERNEL", os.getenv("TCN_STUDENT_KERNEL", "3"))),
            "dilations": _parse_int_list(os.getenv("STUDENT_TCN_DILATIONS", os.getenv("TCN_STUDENT_DILATIONS")), [1, 2, 4]),
            "dropout": float(os.getenv("STUDENT_TCN_DROPOUT", os.getenv("TCN_STUDENT_DROPOUT", "0.1"))),
            "dense": int(os.getenv("STUDENT_TCN_DENSE", os.getenv("TCN_STUDENT_DENSE", "32"))),
        }
    return base


def _gru_config(kind: str) -> dict:
    kind = (kind or "student").strip().lower()
    if kind in ("teacher", "cloud_teacher"):
        return {
            "units": _parse_int_list(os.getenv("CLOUD_GRU_UNITS"), [64, 64]),
            "dropout": float(os.getenv("CLOUD_GRU_DROPOUT", "0.2")),
            "dense": int(os.getenv("CLOUD_GRU_DENSE", "64")),
        }
    if kind == "edge_teacher":
        return {
            "units": _parse_int_list(os.getenv("EDGE_GRU_UNITS"), [48, 48]),
            "dropout": float(os.getenv("EDGE_GRU_DROPOUT", "0.2")),
            "dense": int(os.getenv("EDGE_GRU_DENSE", "48")),
        }
    return {
        "units": _parse_int_list(os.getenv("STUDENT_GRU_UNITS"), [16, 16]),
        "dropout": float(os.getenv("STUDENT_GRU_DROPOUT", "0.1")),
        "dense": int(os.getenv("STUDENT_GRU_DENSE", "16")),
    }


def _mlp_config(kind: str) -> dict:
    kind = (kind or "student").strip().lower()
    if kind in ("teacher", "cloud_teacher"):
        return {
            "units": _parse_int_list(os.getenv("CLOUD_MLP_UNITS"), [256, 128]),
            "dropout": float(os.getenv("CLOUD_MLP_DROPOUT", "0.1")),
        }
    if kind == "edge_teacher":
        return {
            "units": _parse_int_list(os.getenv("EDGE_MLP_UNITS"), [64, 32]),
            "dropout": float(os.getenv("EDGE_MLP_DROPOUT", "0.1")),
        }
    return {
        "units": _parse_int_list(os.getenv("STUDENT_MLP_UNITS"), [32, 16]),
        "dropout": float(os.getenv("STUDENT_MLP_DROPOUT", "0.0")),
    }


def _model_type(kind: str) -> str:
    kind = (kind or "student").strip().lower()
    if kind in ("teacher", "cloud_teacher"):
        return os.getenv("CLOUD_TEACHER_MODEL_TYPE", "tcn").strip().lower()
    if kind == "edge_teacher":
        return os.getenv("EDGE_TEACHER_MODEL_TYPE", "tcn").strip().lower()
    return os.getenv("STUDENT_MODEL_TYPE", "tcn").strip().lower()


def build_model(input_dim: int | None = None, output_dim: int = 1, kind: str = "student", task: str | None = None):
    window, n_features, input_dim = _input_shape(input_dim)
    model_type = _model_type(kind)
    task_name = (task or "regression").strip().lower()
    output_activation = "sigmoid" if task_name == "classification" else "linear"

    if model_type == "mlp":
        cfg = _mlp_config(kind)
        units = cfg["units"] or [32]
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        x = inputs
        for u in units:
            x = tf.keras.layers.Dense(int(u), activation="relu")(x)
            if cfg["dropout"] > 0:
                x = tf.keras.layers.Dropout(cfg["dropout"])(x)
        outputs = tf.keras.layers.Dense(output_dim, activation=output_activation)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    if model_type == "gru":
        cfg = _gru_config(kind)
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.Reshape((window, n_features))(inputs)
        units = cfg["units"] or [16]
        for i, u in enumerate(units):
            return_seq = i < len(units) - 1
            x = tf.keras.layers.GRU(u, return_sequences=return_seq)(x)
            x = tf.keras.layers.Dropout(cfg["dropout"])(x)
        x = tf.keras.layers.Dense(cfg["dense"], activation="relu")(x)
        outputs = tf.keras.layers.Dense(output_dim, activation=output_activation)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    # default: TCN
    cfg = _tcn_config(kind)
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Reshape((window, n_features))(inputs)
    for d in cfg["dilations"]:
        x = tf.keras.layers.Conv1D(
            cfg["filters"],
            cfg["kernel"],
            padding="causal",
            dilation_rate=int(d),
            activation="relu",
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(cfg["dropout"])(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(cfg["dense"], activation="relu")(x)
    outputs = tf.keras.layers.Dense(output_dim, activation=output_activation)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _resolve_loss(loss: str):
    name = (loss or "mse").strip().lower()
    if name == "huber":
        delta = float(os.getenv("HUBER_DELTA", "1.0"))
        return tf.keras.losses.Huber(delta=delta)
    if name in ("mae", "mse"):
        return name
    return loss


def compile_model(m, lr=0.01, loss: str = "mse", task: str | None = None):
    # Stabilize training via optional gradient clipping and optimizer choice.
    opt_name = os.getenv("OPTIMIZER", "sgd").strip().lower()
    clip_norm = float(os.getenv("CLIP_NORM", "1.0"))
    if opt_name == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clip_norm)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0, clipnorm=clip_norm)

    task_name = (task or "regression").strip().lower()
    if task_name == "classification":
        use_loss = _resolve_loss(loss if loss not in (None, "", "auto") else "binary_crossentropy")
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    else:
        use_loss = _resolve_loss(loss)
        metrics = [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ]

    m.compile(
        optimizer=opt,
        loss=use_loss,
        metrics=metrics,
    )

def get_weights_vector(model):
    # flatten all weights to a single 1D float32 array
    w = []
    for v in model.trainable_variables:
        w.append(tf.reshape(v, [-1]))
    flat = tf.concat(w, axis=0).numpy().astype("float32")
    return flat

def set_weights_vector(model, vec):
    # set model weights from 1D array
    idx = 0
    new_vars = []
    for v in model.trainable_variables:
        shape = v.shape
        size = tf.size(v).numpy()
        part = vec[idx:idx+size]
        part = tf.reshape(tf.convert_to_tensor(part, dtype=v.dtype), shape)
        new_vars.append(part)
        idx += size
    for var, newv in zip(model.trainable_variables, new_vars):
        var.assign(newv)

def weights_zeros_like(model):
    import numpy as np
    z = []
    for v in model.trainable_variables:
        z.append(np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype))
    return z, np.concatenate([x.reshape(-1) for x in z]).astype("float32")
