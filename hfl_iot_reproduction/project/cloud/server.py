import os, asyncio, time, numpy as np, random, re
from collections import defaultdict
from pathlib import Path

import tensorflow as tf

from project.common.messaging import MQTTBus
from project.common.crypto import decrypt
from project.common.model import build_model, compile_model, get_weights_vector, set_weights_vector
from project.common.logging_utils import log_event, log_metric
from project.common.metrics import score_from_rmse, regression_metrics, classification_metrics
from project.common.config import load_hparams
from project.common.data_utils import load_global_test_by_target, load_split_by_target, unscale_target
from project.common.dataset_config import load_dataset_config, task_for_target

MQTT_BASE = os.getenv("MQTT_BASE_TOPIC", "hfl")
MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_QOS = int(os.getenv("MQTT_QOS", "1"))
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", "60"))
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)

hp = load_hparams()
beta_cloud = float(os.getenv("BETA_CLOUD", str(hp["beta_cloud"])))

cfg = load_dataset_config()
TARGETS = sorted({t for t in (cfg.get("targets", {}) or {}).values() if t})
if not TARGETS:
    TARGETS = ["temp"]
TARGET_TASKS = {t: task_for_target(t, cfg) for t in TARGETS}

# Readiness barrier (optional)
CLOUD_EXPECTED_EDGES = {e.strip() for e in os.getenv("CLOUD_EXPECTED_EDGES", "").split(",") if e.strip()}
CLOUD_WAIT_FOR_EDGES = os.getenv("CLOUD_WAIT_FOR_EDGES", "1") == "1"

# Teacher configuration
TEACHER_ENABLE = os.getenv("TEACHER_ENABLE", "1") == "1"
TEACHER_EPOCHS = int(os.getenv("TEACHER_EPOCHS", str(hp.get("teacher_epochs", 5))))
TEACHER_BATCH = int(os.getenv("TEACHER_BATCH", str(hp.get("teacher_batch", 64))))
TEACHER_LR = float(os.getenv("TEACHER_LR", str(hp.get("teacher_lr", 1e-3))))
TEACHER_MAX_SAMPLES = int(os.getenv("TEACHER_MAX_SAMPLES", str(hp.get("teacher_max_samples", 0))))
TEACHER_REFRESH_EVERY = int(os.getenv("TEACHER_REFRESH_EVERY", str(hp.get("teacher_refresh_every", 0))))
TEACHER_PUSH_INIT = os.getenv("TEACHER_PUSH_INIT", "0") == "1"
TEACHER_PUSH_EVERY = int(os.getenv("TEACHER_PUSH_EVERY", "0"))

# Server-side fine-tuning (student on proxy data)
SERVER_FT_EPOCHS = int(os.getenv("SERVER_FT_EPOCHS", "0"))
SERVER_FT_BATCH = int(os.getenv("SERVER_FT_BATCH", "128"))
SERVER_FT_EVERY = int(os.getenv("SERVER_FT_EVERY", "0"))
SERVER_FT_MAX_SAMPLES = int(os.getenv("SERVER_FT_MAX_SAMPLES", "0"))
SERVER_FT_ALPHA = float(os.getenv("SERVER_FT_ALPHA", "0.0"))

# State
edge_meta = defaultdict(lambda: defaultdict(lambda: {"p": 0.0, "q": 1.0}))
edge_target = {}  # edge_id -> target
w_global = {}     # target -> weights
round_ctr = 0

# Eval
eval_models = {}
X_test_by_target = {}

# Teacher
teacher_weights = {}
teacher_version = {}
teacher_metrics_by_target = {}
teacher_models = {}

# Proxy data for server-side fine-tuning
proxy_train_by_target = {}
ft_models = {}

# Edges ativos (via MQTT)
peers = set()  # edge_id -> conectado (hello recebido)
edge_last = defaultdict(dict)  # target -> {edge_id: weights}

SAVE_WEIGHTS = os.getenv("SAVE_GLOBAL_WEIGHTS", "0") == "1"
SAVE_EVERY = int(os.getenv("SAVE_GLOBAL_WEIGHTS_EVERY", "1"))
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", "/workspace/outputs/weights"))


def _cloud_ready() -> bool:
    if not CLOUD_WAIT_FOR_EDGES or not CLOUD_EXPECTED_EDGES:
        return True
    return CLOUD_EXPECTED_EDGES.issubset(peers)


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower()).strip("_")


def _cloud_down_topic(edge_id: str) -> str:
    return f"{MQTT_BASE}/cloud/down/{edge_id}"


def _edge_from_topic(topic: str) -> str | None:
    parts = topic.strip("/").split("/")
    for i in range(len(parts) - 2):
        if parts[i] == "cloud" and parts[i + 1] == "up":
            return parts[i + 2]
    return None


def edge_key_for(edge_id):
    if edge_id == "edge1":
        return os.getenv("EDGE1_KEY", os.getenv("EDGE_KEY_HEX", "00112233445566778899AABBCCDDEEFF"))
    if edge_id == "edge2":
        return os.getenv("EDGE2_KEY", os.getenv("EDGE_KEY_HEX", "0102030405060708090A0B0C0D0E0F10"))
    return os.getenv("EDGE_KEY_HEX", "00112233445566778899AABBCCDDEEFF")


def _task(target: str) -> str:
    return (TARGET_TASKS.get(target, "regression") or "regression").strip().lower()


def _loss_fn(task_name: str) -> str:
    if task_name == "classification":
        return os.getenv("LOSS_FN_CLASSIF", "binary_crossentropy")
    return os.getenv("LOSS_FN", "mse")


def _train_teacher(target: str):
    Xtr, ytr = load_split_by_target(target, split="train", max_samples=TEACHER_MAX_SAMPLES or None)
    if len(ytr) == 0:
        return None, None

    task_name = _task(target)
    model = build_model(kind="cloud_teacher", task=task_name)
    compile_model(model, lr=TEACHER_LR, loss=_loss_fn(task_name), task=task_name)
    model.fit(
        Xtr,
        ytr,
        batch_size=TEACHER_BATCH,
        epochs=TEACHER_EPOCHS,
        verbose=0,
        callbacks=[tf.keras.callbacks.TerminateOnNaN()],
    )

    # Avaliação do teacher no conjunto de teste (se existir)
    Xt, yt = load_split_by_target(target, split="test")
    metrics = {}
    if len(yt) > 0:
        preds = model.predict(Xt, verbose=0).reshape(-1)
        if np.isfinite(preds).all():
            if task_name == "classification":
                m = classification_metrics(yt, preds)
                metrics = {
                    "teacher_loss": m.get("bce"),
                    "teacher_mae": m.get("mae"),
                    "teacher_rmse": m.get("brier_rmse"),
                    "teacher_score": m.get("acc"),
                    "teacher_acc": m.get("acc"),
                    "teacher_precision": m.get("precision"),
                    "teacher_recall": m.get("recall"),
                    "teacher_f1": m.get("f1"),
                }
            else:
                yt_u = unscale_target(yt, target)
                preds_u = unscale_target(preds, target)
                mse, mae, rmse, r2, mape = regression_metrics(yt_u, preds_u)
                metrics = {
                    "teacher_loss": mse,
                    "teacher_mae": mae,
                    "teacher_rmse": rmse,
                    "teacher_r2": r2,
                    "teacher_mape": mape,
                    "teacher_score": score_from_rmse(rmse),
                }
    return model, metrics


def _teacher_predict(target: str, X: np.ndarray):
    if X is None or len(X) == 0:
        return None
    model = teacher_models.get(target)
    weights = teacher_weights.get(target)
    task_name = _task(target)
    if model is None or weights is None:
        if weights is None:
            return None
        model = build_model(kind="cloud_teacher", task=task_name)
        compile_model(model, lr=TEACHER_LR, loss=_loss_fn(task_name), task=task_name)
        set_weights_vector(model, weights)
        teacher_models[target] = model
    else:
        # ensure weights are up-to-date
        set_weights_vector(model, weights)
    try:
        preds = model.predict(X, verbose=0).reshape(-1)
    except Exception:
        return None
    if not np.isfinite(preds).all():
        return None
    return preds


def _server_finetune(target: str, weights: np.ndarray):
    global ft_models
    data = proxy_train_by_target.get(target)
    if not data:
        return weights, None
    Xp, yp = data
    if Xp is None or len(yp) == 0:
        return weights, None
    task_name = _task(target)
    model = ft_models.get(target)
    if model is None:
        model = build_model(kind="student", task=task_name)
        compile_model(model, lr=hp["eta"], loss=_loss_fn(task_name), task=task_name)
        ft_models[target] = model
    set_weights_vector(model, weights)

    y_train = yp
    alpha = max(0.0, min(1.0, SERVER_FT_ALPHA))
    if alpha > 0:
        tpred = _teacher_predict(target, Xp)
        if tpred is not None:
            y_train = (1.0 - alpha) * yp + alpha * tpred

    model.fit(
        Xp,
        y_train,
        batch_size=SERVER_FT_BATCH,
        epochs=max(1, SERVER_FT_EPOCHS),
        verbose=0,
        callbacks=[tf.keras.callbacks.TerminateOnNaN()],
    )
    new_w = get_weights_vector(model)

    # metrics on proxy train (unscaled)
    try:
        preds = model.predict(Xp, verbose=0).reshape(-1)
    except Exception:
        return new_w, None
    if not np.isfinite(preds).all():
        return new_w, None
    if task_name == "classification":
        m = classification_metrics(yp, preds)
        return new_w, {
            "server_ft_loss": m.get("bce"),
            "server_ft_mae": m.get("mae"),
            "server_ft_rmse": m.get("brier_rmse"),
            "server_ft_score": m.get("acc"),
            "server_ft_acc": m.get("acc"),
            "server_ft_precision": m.get("precision"),
            "server_ft_recall": m.get("recall"),
            "server_ft_f1": m.get("f1"),
            "server_ft_alpha": alpha,
        }
    yt_u = unscale_target(yp, target)
    preds_u = unscale_target(preds, target)
    mse, mae, rmse, r2, mape = regression_metrics(yt_u, preds_u)
    return new_w, {
        "server_ft_loss": mse,
        "server_ft_mae": mae,
        "server_ft_rmse": rmse,
        "server_ft_r2": r2,
        "server_ft_mape": mape,
        "server_ft_score": score_from_rmse(rmse),
        "server_ft_alpha": alpha,
    }


def _evaluate_target(target: str, weights: np.ndarray):
    data = X_test_by_target.get(target)
    if data is None:
        return None
    Xt, yt = data
    if len(yt) == 0:
        return None
    task_name = _task(target)
    model = eval_models.get(target)
    if model is None:
        model = build_model(kind="student", task=task_name)
        compile_model(model, lr=hp["eta"], loss=_loss_fn(task_name), task=task_name)
        eval_models[target] = model
    set_weights_vector(model, weights)
    try:
        preds = model.predict(Xt, verbose=0).reshape(-1)
    except Exception:
        return None
    if not np.isfinite(preds).all():
        return None
    if task_name == "classification":
        m = classification_metrics(yt, preds)
        return {
            "loss": m.get("bce"),
            "mae": m.get("mae"),
            "rmse": m.get("brier_rmse"),
            "score": m.get("acc"),
            "acc": m.get("acc"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "brier": m.get("brier"),
        }
    yt_u = unscale_target(yt, target)
    preds_u = unscale_target(preds, target)
    mse, mae, rmse, r2, mape = regression_metrics(yt_u, preds_u)
    return {
        "loss": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "score": score_from_rmse(rmse),
    }


async def mqtt_loop():
    """
    Loop principal do Cloud via MQTT. Processa hello/edge_model e publica init/model.
    """
    global round_ctr

    mqtt_bus = MQTTBus(client_id="cloud-mqtt", host=MQTT_HOST, port=MQTT_PORT)
    backoff = 1.0
    while True:
        try:
            await mqtt_bus.connect()
            break
        except Exception as e:
            print(f"[cloud] erro conectando MQTT ({MQTT_HOST}:{MQTT_PORT}): {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    mqtt_bus.subscribe(f"{MQTT_BASE}/cloud/up/#")
    print(f"[cloud] MQTT conectado em {MQTT_HOST}:{MQTT_PORT}, tópico={MQTT_BASE}/cloud/up/#")

    while True:
        topic, msg, meta = await mqtt_bus.recv_json()
        if not isinstance(msg, dict):
            continue
        edge = msg.get("edge") or _edge_from_topic(topic)
        if not edge:
            continue
        log_event(
            "cloud",
            event="mqtt_rx",
            edge=edge,
            topic=topic,
            msg_type=msg.get("type"),
            target=msg.get("target"),
            **(meta or {}),
        )

        mtype = msg.get("type")
        if mtype == "hello":
            target = msg.get("target") or TARGETS[0]
            if target not in TARGETS:
                print(f"[cloud] target desconhecido '{target}' de {edge}; usando {TARGETS[0]}")
                target = TARGETS[0]
            edge_target[edge] = target
            peers.add(edge)
            print(f"[cloud] novo edge conectado: {edge} (target={target})")

            # Envia init
            try:
                teacher_payload = teacher_weights.get(target) if (TEACHER_ENABLE and TEACHER_PUSH_INIT) else None
                tx_init = mqtt_bus.publish_json(_cloud_down_topic(edge), {
                    "type": "init",
                    "whex": w_global[target].tobytes().hex(),
                    "target": target,
                    "teacher_whex": teacher_payload.tobytes().hex() if teacher_payload is not None else None,
                    "teacher_version": teacher_version.get(target),
                    "teacher_score": (teacher_metrics_by_target.get(target, {}) or {}).get("teacher_score"),
                })
                log_event(
                    "cloud",
                    event="mqtt_tx",
                    edge=edge,
                    topic=_cloud_down_topic(edge),
                    msg_type="init",
                    target=target,
                    **(tx_init or {}),
                )
                print(f"[cloud] init enviado para {edge} (n={w_global[target].size})")
            except Exception as e:
                print(f"[cloud] erro enviando init para {edge}: {e}")
            continue

        if mtype != "edge_model":
            continue

        msg_target = msg.get("target") or edge_target.get(edge) or TARGETS[0]
        if msg_target not in TARGETS:
            print(f"[cloud] target inválido '{msg_target}' de {edge}. Ignorado.")
            continue

        ek = edge_key_for(edge)
        try:
            dec_start = time.time()
            dec = decrypt(ek, bytes.fromhex(msg["bytes"]))
            dec_ms = (time.time() - dec_start) * 1000.0
            we = np.frombuffer(dec, dtype=np.float32)
        except Exception as e:
            print(f"[cloud] erro decriptando modelo de {edge}: {e}")
            continue

        if we.shape != w_global[msg_target].shape:
            print(f"[cloud] SHAPE MISMATCH de {edge}: recebido={we.shape}, esperado={w_global[msg_target].shape}. Ignorado.")
            continue

        edge_meta[msg_target][edge]["p"] = float(msg.get("pactual", 0.0))
        edge_meta[msg_target][edge]["q"] = float(msg.get("qedge", 1.0))
        edge_last[msg_target][edge] = we

        # Aguarda todos os edges esperados antes de agregar
        if not _cloud_ready():
            try:
                send_teacher = TEACHER_ENABLE and TEACHER_PUSH_EVERY > 0 and round_ctr % TEACHER_PUSH_EVERY == 0
                teacher_payload = teacher_weights.get(msg_target) if send_teacher else None
                tx_wait = mqtt_bus.publish_json(_cloud_down_topic(edge), {
                    "type": "model",
                    "whex": w_global[msg_target].tobytes().hex(),
                    "round": round_ctr,
                    "target": msg_target,
                    "ve": float(beta_cloud * edge_meta[msg_target][edge]["p"] + (1.0 - beta_cloud) * edge_meta[msg_target][edge]["q"]),
                    "teacher_whex": teacher_payload.tobytes().hex() if teacher_payload is not None else None,
                    "teacher_version": teacher_version.get(msg_target),
                    "teacher_score": (teacher_metrics_by_target.get(msg_target, {}) or {}).get("teacher_score"),
                    "ready": False,
                    "expected_edges": sorted(CLOUD_EXPECTED_EDGES) if CLOUD_EXPECTED_EDGES else None,
                })
                log_event(
                    "cloud",
                    event="mqtt_tx",
                    edge=edge,
                    topic=_cloud_down_topic(edge),
                    msg_type="model",
                    target=msg_target,
                    ready=False,
                    **(tx_wait or {}),
                )
            except Exception:
                pass
            continue

        # Agregação por target
        num = np.zeros_like(w_global[msg_target], dtype=np.float32)
        den = 0.0
        for e_id, w in edge_last[msg_target].items():
            meta = edge_meta[msg_target][e_id]
            v = beta_cloud * meta["p"] + (1.0 - beta_cloud) * meta["q"]
            num += v * w
            den += v
        if den > 0:
            w_global[msg_target] = num / den

        round_ctr += 1

        # Avaliação por target
        target_metrics = {}
        for t in TARGETS:
            metrics = _evaluate_target(t, w_global[t])
            if not metrics:
                continue
            key = _slug(t)
            target_metrics[f"global_{key}_rmse"] = metrics["rmse"]
            target_metrics[f"global_{key}_mae"] = metrics["mae"]
            if "r2" in metrics:
                target_metrics[f"global_{key}_r2"] = metrics.get("r2")
            if "mape" in metrics:
                target_metrics[f"global_{key}_mape"] = metrics.get("mape")
            target_metrics[f"global_{key}_score"] = metrics["score"]
            target_metrics[f"global_{key}_loss"] = metrics["loss"]
            if "acc" in metrics:
                target_metrics[f"global_{key}_acc"] = metrics.get("acc")
            if "precision" in metrics:
                target_metrics[f"global_{key}_precision"] = metrics.get("precision")
            if "recall" in metrics:
                target_metrics[f"global_{key}_recall"] = metrics.get("recall")
            if "f1" in metrics:
                target_metrics[f"global_{key}_f1"] = metrics.get("f1")

        # Métricas globais compatíveis (do target atualizado)
        cur_key = _slug(msg_target)
        global_rmse = target_metrics.get(f"global_{cur_key}_rmse")
        global_mae = target_metrics.get(f"global_{cur_key}_mae")
        global_r2 = target_metrics.get(f"global_{cur_key}_r2")
        global_mape = target_metrics.get(f"global_{cur_key}_mape")
        global_score = target_metrics.get(f"global_{cur_key}_score")
        global_loss = target_metrics.get(f"global_{cur_key}_loss")
        global_acc = target_metrics.get(f"global_{cur_key}_acc")
        global_precision = target_metrics.get(f"global_{cur_key}_precision")
        global_recall = target_metrics.get(f"global_{cur_key}_recall")
        global_f1 = target_metrics.get(f"global_{cur_key}_f1")

        # Persistir pesos globais (opcional)
        if SAVE_WEIGHTS and round_ctr % max(1, SAVE_EVERY) == 0:
            try:
                WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
                fname = f"cloud_{cur_key}_round_{round_ctr:04d}.npy"
                np.save(WEIGHTS_DIR / fname, w_global[msg_target].astype("float32"))
            except Exception as e:
                print(f"[cloud] falha salvando pesos: {e}")

        # Server-side fine-tuning (student) on proxy data
        server_ft_metrics = {}
        if SERVER_FT_EVERY > 0 and SERVER_FT_EPOCHS > 0 and round_ctr % SERVER_FT_EVERY == 0:
            w_new, metrics_ft = _server_finetune(msg_target, w_global[msg_target])
            if w_new is not None:
                w_global[msg_target] = w_new
            if metrics_ft:
                server_ft_metrics.update(metrics_ft)

        # 🔹 Broadcast do modelo para todos os edges
        ve_map = {}
        for e_id, t in edge_target.items():
            meta = edge_meta[t].get(e_id, {"p": 0.0, "q": 1.0})
            ve_map[e_id] = float(beta_cloud * meta["p"] + (1.0 - beta_cloud) * meta["q"])

        send_teacher = TEACHER_ENABLE and TEACHER_PUSH_EVERY > 0 and round_ctr % TEACHER_PUSH_EVERY == 0
        for e_id, t in edge_target.items():
            try:
                teacher_payload = teacher_weights.get(t) if send_teacher else None
                tx_model = mqtt_bus.publish_json(_cloud_down_topic(e_id), {
                    "type": "model",
                    "whex": w_global[t].tobytes().hex(),
                    "round": round_ctr,
                    "target": t,
                    "ve": ve_map.get(e_id, 1.0),
                    "teacher_whex": teacher_payload.tobytes().hex() if teacher_payload is not None else None,
                    "teacher_version": teacher_version.get(t),
                    "teacher_score": (teacher_metrics_by_target.get(t, {}) or {}).get("teacher_score"),
                })
                log_event(
                    "cloud",
                    event="mqtt_tx",
                    edge=e_id,
                    topic=_cloud_down_topic(e_id),
                    msg_type="model",
                    target=t,
                    ready=True,
                    **(tx_model or {}),
                )
                print(f"[cloud] modelo global enviado -> {e_id}, round={round_ctr}, target={t}")
            except Exception as e:
                print(f"[cloud] erro enviando modelo para {e_id}: {e}")

        # Inclui métricas do teacher (constantes) se disponíveis
        teacher_metrics = {}
        if teacher_metrics_by_target:
            for t, vals in teacher_metrics_by_target.items():
                key = _slug(t)
                for mk, mv in vals.items():
                    teacher_metrics[f"{key}_{mk}"] = mv

        log_metric(
            "cloud",
            round=round_ctr,
            target=msg_target,
            task=_task(msg_target),
            edges=len(edge_last.get(msg_target, {})),
            beta_cloud=beta_cloud,
            global_loss=global_loss,
            global_mae=global_mae,
            global_rmse=global_rmse,
            global_r2=global_r2,
            global_mape=global_mape,
            global_score=global_score,
            global_acc=global_acc,
            global_precision=global_precision,
            global_recall=global_recall,
            global_f1=global_f1,
            dec_ms=dec_ms,
            payload_bytes=msg.get("payload_bytes"),
            **target_metrics,
            **teacher_metrics,
            **server_ft_metrics,
        )

        # Opcional: refresh do teacher
        if TEACHER_ENABLE and TEACHER_REFRESH_EVERY > 0 and round_ctr % TEACHER_REFRESH_EVERY == 0:
            model, metrics = _train_teacher(msg_target)
            if model is not None:
                teacher_weights[msg_target] = get_weights_vector(model)
                teacher_version[msg_target] = int(teacher_version.get(msg_target, 0)) + 1
                if metrics:
                    teacher_metrics_by_target[msg_target] = metrics
async def main():
    global w_global, eval_models, X_test_by_target, teacher_weights, teacher_version, teacher_metrics_by_target, proxy_train_by_target

    # Inicializa pesos globais (student) por target
    w_global = {}
    eval_models = {}
    edge_last.clear()
    for t in TARGETS:
        task_name = _task(t)
        model = build_model(kind="student", task=task_name)
        compile_model(model, lr=hp["eta"], loss=_loss_fn(task_name), task=task_name)
        w_global[t] = get_weights_vector(model).copy()
        eval_models[t] = model

    X_test_by_target = load_global_test_by_target()
    # proxy train sets for server-side fine-tuning
    if SERVER_FT_EPOCHS > 0:
        for t in TARGETS:
            Xp, yp = load_split_by_target(t, split="train", max_samples=SERVER_FT_MAX_SAMPLES or None)
            if len(yp) > 0:
                proxy_train_by_target[t] = (Xp, yp)

    # Treinar teacher por target (opcional)
    if TEACHER_ENABLE:
        for t in TARGETS:
            model, metrics = _train_teacher(t)
            if model is None:
                continue
            teacher_weights[t] = get_weights_vector(model)
            teacher_version[t] = int(teacher_version.get(t, 0)) + 1
            if metrics:
                teacher_metrics_by_target[t] = metrics
            teacher_models[t] = model
            log_event("cloud", event="teacher_trained", target=t, **(metrics or {}))

    log_event(
        "cloud",
        event="start",
        targets=TARGETS,
        tasks=TARGET_TASKS,
        mqtt_host=MQTT_HOST,
        mqtt_port=MQTT_PORT,
        mqtt_base=MQTT_BASE,
        mqtt_qos=MQTT_QOS,
        mqtt_keepalive=MQTT_KEEPALIVE,
    )
    any_weights = next(iter(w_global.values())) if w_global else np.array([], dtype=np.float32)
    print(f"[cloud] iniciado (MQTT) targets={TARGETS}, pesos iniciais n={any_weights.size}")
    await mqtt_loop()


if __name__ == "__main__":
    asyncio.run(main())
