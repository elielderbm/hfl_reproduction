import os, asyncio, json, time, numpy as np, random, re
from collections import defaultdict, deque
from pathlib import Path

from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
import tensorflow as tf

from project.common.messaging import ws_server, b64d
from project.common.crypto import decrypt
from project.common.model import build_model, compile_model, get_weights_vector, set_weights_vector
from project.common.logging_utils import log_event, log_metric
from project.common.metrics import score_from_rmse, regression_metrics, classification_metrics
from project.common.config import load_hparams
from project.common.data_utils import load_global_test_by_target, load_split_by_target, unscale_target
from project.common.dataset_config import load_dataset_config, task_for_target

PORT = 9000
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)

hp = load_hparams()
beta_cloud = float(os.getenv("BETA_CLOUD", str(hp["beta_cloud"])))

cfg = load_dataset_config()
targets_map = cfg.get("targets", {})
TARGETS = sorted({t for t in (cfg.get("targets", {}) or {}).values() if t})
if not TARGETS:
    TARGETS = ["temp"]
TARGET_TASKS = {t: task_for_target(t, cfg) for t in TARGETS}

# Async aggregation params (reuso dos hiperparâmetros do edge)
alpha_async = float(hp.get("alpha_edge", 0.7))
beta_async = float(hp.get("beta_edge", 0.3))
alpha_sw = float(hp.get("alpha_sw", 0.5))
pdesired = float(hp.get("pdesired", 0.75))
sw_init = float(hp.get("sw_init", 7))
WEIGHT_BY_SAMPLES = os.getenv("EDGE_WEIGHT_BY_SAMPLES", "0") == "1"

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
clients = set()
clients_by_target = defaultdict(set)  # target -> set(iot)
updates_by_target = defaultdict(deque)  # target -> deque[(ts, iot, round, w, score, dec_ms, payload_bytes, n_train)]
last_agg_ts = {}
window_by_target = {}
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

SAVE_WEIGHTS = os.getenv("SAVE_GLOBAL_WEIGHTS", "0") == "1"
SAVE_EVERY = int(os.getenv("SAVE_GLOBAL_WEIGHTS_EVERY", "1"))
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", "/workspace/outputs/weights"))


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower()).strip("_")


def iot_key_for(iot_id: str) -> str:
    env_key = f"{iot_id.upper()}_KEY"
    return os.getenv(env_key, os.getenv("IOT_KEY_HEX", "00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF"))


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


async def handler(ws, path):
    try:
        hello_raw = await ws.recv()
        hello = json.loads(hello_raw)
    except (ConnectionClosedOK, ConnectionClosedError):
        print("[cloud] conexão encerrada antes do hello")
        return
    except Exception as e:
        print(f"[cloud] erro recebendo hello: {e}")
        return

    if hello.get("type") != "hello":
        print("[cloud] mensagem inicial inválida")
        return

    iot = hello.get("iot")
    target = hello.get("target") or (targets_map.get(iot) if iot else None) or TARGETS[0]
    if target not in TARGETS:
        print(f"[cloud] target desconhecido '{target}' de {iot}; usando {TARGETS[0]}")
        target = TARGETS[0]

    if iot:
        clients.add(iot)
        clients_by_target[target].add(iot)

    print(f"[cloud] novo IoT conectado: {iot} (target={target})")

    # Envia init
    try:
        teacher_payload = teacher_weights.get(target) if (TEACHER_ENABLE and TEACHER_PUSH_INIT) else None
        await ws.send(json.dumps({
            "type": "init",
            "whex": w_global[target].tobytes().hex(),
            "target": target,
            "teacher_whex": teacher_payload.tobytes().hex() if teacher_payload is not None else None,
            "teacher_version": teacher_version.get(target),
            "teacher_score": (teacher_metrics_by_target.get(target, {}) or {}).get("teacher_score"),
        }))
        print(f"[cloud] init enviado para {iot} (n={w_global[target].size})")
    except Exception as e:
        print(f"[cloud] erro enviando init para {iot}: {e}")
        return

    try:
        while True:
            try:
                raw = await ws.recv()
                if raw is None:
                    print(f"[cloud] conexão fechada (sem msg) por {iot}")
                    break
                msg = json.loads(raw)
            except ConnectionClosedOK:
                print(f"[cloud] conexão encerrada OK ({iot})")
                break
            except ConnectionClosedError as e:
                print(f"[cloud] conexão encerrada erro ({iot}): {e}")
                break
            except Exception as e:
                print(f"[cloud] erro ao receber de {iot}: {e}")
                continue

            if msg.get("type") != "update":
                print(f"[cloud] ignorando msg de {iot} tipo={msg.get('type')}")
                continue

            msg_target = msg.get("target") or (targets_map.get(iot) if iot else None) or target
            if msg_target not in TARGETS:
                print(f"[cloud] target inválido '{msg_target}' de {iot}. Ignorado.")
                continue

            try:
                enc = b64d(msg["bytes"])
                dec_start = time.time()
                dec = decrypt(iot_key_for(iot or ""), enc)
                dec_ms = (time.time() - dec_start) * 1000.0
                w = np.frombuffer(dec, dtype=np.float32)
            except Exception as e:
                print(f"[cloud] erro decriptando update de {iot}: {e}")
                continue

            if w.shape != w_global[msg_target].shape:
                print(
                    f"[cloud] SHAPE MISMATCH de {iot}: recebido={w.shape}, esperado={w_global[msg_target].shape}. Ignorado."
                )
                continue

            train_score = msg.get("train_score")
            train_rmse = msg.get("train_rmse")
            if train_score is None and train_rmse is not None:
                try:
                    train_score = score_from_rmse(float(train_rmse))
                except Exception:
                    train_score = None

            clients.add(iot)
            clients_by_target[msg_target].add(iot)
            updates_by_target[msg_target].append(
                (
                    time.time(),
                    iot,
                    msg.get("round"),
                    w,
                    train_score,
                    dec_ms,
                    msg.get("payload_bytes"),
                    msg.get("n_train"),
                )
            )

            send_teacher = TEACHER_ENABLE and TEACHER_PUSH_EVERY > 0 and round_ctr % TEACHER_PUSH_EVERY == 0
            teacher_payload = teacher_weights.get(msg_target) if send_teacher else None
            await ws.send(json.dumps({
                "type": "model",
                "whex": w_global[msg_target].tobytes().hex(),
                "round": round_ctr,
                "target": msg_target,
                "teacher_whex": teacher_payload.tobytes().hex() if teacher_payload is not None else None,
                "teacher_version": teacher_version.get(msg_target),
                "teacher_score": (teacher_metrics_by_target.get(msg_target, {}) or {}).get("teacher_score"),
            }))

    finally:
        print(f"[cloud] handler encerrado para {iot}")


async def main():
    global w_global, eval_models, X_test_by_target, teacher_weights, teacher_version, teacher_metrics_by_target, proxy_train_by_target

    # Inicializa pesos globais (student) por target
    w_global = {}
    eval_models = {}
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

    for t in TARGETS:
        last_agg_ts[t] = time.time()
        window_by_target[t] = float(sw_init)

    log_event("cloud", event="start", port=PORT, targets=TARGETS, tasks=TARGET_TASKS)
    any_weights = next(iter(w_global.values())) if w_global else np.array([], dtype=np.float32)
    print(f"[cloud] iniciado na porta {PORT}, targets={TARGETS}, pesos iniciais n={any_weights.size}")

    async def aggregation_loop():
        global round_ctr, w_global
        while True:
            await asyncio.sleep(0.1)
            now = time.time()
            for target in TARGETS:
                updates = updates_by_target[target]
                if not updates:
                    continue

                last_ts = last_agg_ts.get(target, now)
                window = window_by_target.get(target, float(sw_init))
                if (now - last_ts) < max(1.0, float(window)):
                    continue

                # consumir updates desde o último ciclo
                while updates and updates[0][0] <= last_ts:
                    updates.popleft()
                batch = []
                while updates and updates[0][0] <= now:
                    batch.append(updates.popleft())

                if not batch:
                    last_agg_ts[target] = now
                    continue

                sum_w = np.zeros_like(w_global[target], dtype=np.float32)
                scores = []
                dec_ms_vals = []
                payload_bytes = 0
                n_eff = 0.0
                contrib = set()
                for (_ts, iot_id, _r, w, score, dec_ms, pbytes, n_train) in batch:
                    weight = float(n_train) if WEIGHT_BY_SAMPLES and n_train is not None else 1.0
                    sum_w += w * weight
                    n_eff += weight
                    contrib.add(iot_id)
                    if score is not None:
                        scores.append(float(score))
                    if dec_ms is not None:
                        dec_ms_vals.append(float(dec_ms))
                    if pbytes is not None:
                        payload_bytes += int(pbytes)

                denom = alpha_async + beta_async * (n_eff if WEIGHT_BY_SAMPLES else len(batch))
                if denom > 0:
                    w_global[target] = (alpha_async * w_global[target] + beta_async * sum_w) / denom

                total_clients = len(clients_by_target.get(target, set()))
                if total_clients <= 0:
                    total_clients = max(1, len(contrib))
                pactual = len(contrib) / max(1, total_clients)
                window = max(3.0, float(window) + alpha_sw * (pdesired - pactual))
                window_by_target[target] = window
                last_agg_ts[target] = now

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
                cur_key = _slug(target)
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
                        np.save(WEIGHTS_DIR / fname, w_global[target].astype("float32"))
                    except Exception as e:
                        print(f"[cloud] falha salvando pesos: {e}")

                # Server-side fine-tuning (student) on proxy data
                server_ft_metrics = {}
                if SERVER_FT_EVERY > 0 and SERVER_FT_EPOCHS > 0 and round_ctr % SERVER_FT_EVERY == 0:
                    w_new, metrics_ft = _server_finetune(target, w_global[target])
                    if w_new is not None:
                        w_global[target] = w_new
                    if metrics_ft:
                        server_ft_metrics.update(metrics_ft)

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
                    target=target,
                    task=_task(target),
                    edges=len(contrib),
                    beta_cloud=beta_cloud,
                    window=window,
                    pactual=pactual,
                    buf=len(batch),
                    n_eff=n_eff,
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
                    dec_ms=float(np.mean(dec_ms_vals)) if dec_ms_vals else None,
                    payload_bytes=payload_bytes if payload_bytes > 0 else None,
                    **target_metrics,
                    **teacher_metrics,
                    **server_ft_metrics,
                )

                # Opcional: refresh do teacher
                if TEACHER_ENABLE and TEACHER_REFRESH_EVERY > 0 and round_ctr % TEACHER_REFRESH_EVERY == 0:
                    model, metrics = _train_teacher(target)
                    if model is not None:
                        teacher_weights[target] = get_weights_vector(model)
                        teacher_version[target] = int(teacher_version.get(target, 0)) + 1
                        if metrics:
                            teacher_metrics_by_target[target] = metrics

    await asyncio.gather(
        ws_server("0.0.0.0", PORT, handler),
        aggregation_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
