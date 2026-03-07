import os, asyncio, json, time, numpy as np, random
import tensorflow as tf
from project.common.messaging import ws_connect, b64e
from project.common.crypto import encrypt
from project.common.model import build_model, compile_model, get_weights_vector, set_weights_vector
from project.common.logging_utils import log_event, log_metric
from project.common.metrics import score_from_rmse, regression_metrics, classification_metrics
from project.common.config import load_hparams
from project.common.data_utils import load_client_split, unscale_target
from project.common.dataset_config import load_dataset_config, task_for_target

IOT_ID = os.getenv("IOT_ID", "iot1")
EDGE_URI = os.getenv("EDGE_URI", "ws://cloud:9000")
IOT_KEY_HEX = os.getenv("IOT_KEY_HEX", "00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF")
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)

hp = load_hparams()
cfg = load_dataset_config()
TARGET = cfg.get("targets", {}).get(IOT_ID, "temp")
TASK = task_for_target(TARGET, cfg)
DISTILL_ALPHA = float(os.getenv("DISTILL_ALPHA", str(cfg.get("distill_alpha", 0.0) or 0.0)))
DISTILL_ALPHA = max(0.0, min(1.0, DISTILL_ALPHA))
DISTILL_ALPHA_DYNAMIC = os.getenv("DISTILL_ALPHA_DYNAMIC", "0") == "1"
IOT_WAIT_FOR_READY = os.getenv("IOT_WAIT_FOR_READY", "1") == "1"

# estado
round_ctr = 0
local_model = None
w_local = None
teacher_score = None


def _safe_predict(model, X):
    try:
        preds = model.predict(X, verbose=0).reshape(-1)
    except Exception:
        return None
    if not np.isfinite(preds).all():
        return None
    return preds


def _finite_ok(X, y) -> bool:
    if X is None or y is None:
        return False
    if len(y) == 0 or X.size == 0:
        return False
    return np.isfinite(X).all() and np.isfinite(y).all()


def _weight_stats(vec: np.ndarray) -> dict:
    if vec is None or vec.size == 0:
        return {}
    return {
        "w_norm": float(np.linalg.norm(vec)),
        "w_mean": float(np.mean(vec)),
        "w_std": float(np.std(vec)),
        "w_min": float(np.min(vec)),
        "w_max": float(np.max(vec)),
    }


def _predict_metrics(model, X, y, target: str | None = None, task: str | None = None):
    preds = _safe_predict(model, X)
    if preds is None:
        return {}
    y_true = y.reshape(-1)
    task_name = (task or "regression").strip().lower()
    if task_name != "classification" and target:
        y_true = unscale_target(y_true, target)
        preds = unscale_target(preds, target)
        mse, mae, rmse, r2, mape = regression_metrics(y_true, preds)
        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
    return classification_metrics(y_true, preds)


def _distill_alpha_effective(base: float, score: float | None) -> float:
    if not DISTILL_ALPHA_DYNAMIC or base <= 0:
        return base
    if score is None:
        return base * 0.5
    scale = max(0.0, min(1.0, (score - 0.2) / 0.8))
    return base * scale

async def run_iot():
    global round_ctr, local_model, w_local
    (Xtr, ytr), (Xv, yv), (Xt, yt) = load_client_split(IOT_ID)

    loss_fn = os.getenv("LOSS_FN_CLASSIF", "binary_crossentropy") if TASK == "classification" else os.getenv("LOSS_FN", "mse")
    local_model = build_model(kind="student", task=TASK)
    compile_model(local_model, lr=hp["eta"], loss=loss_fn, task=TASK)
    w_local = get_weights_vector(local_model)

    teacher_model = None
    teacher_weights = None
    teacher_version = None
    teacher_pred_cache = None
    teacher_score = None
    if DISTILL_ALPHA > 0:
        teacher_model = build_model(kind="cloud_teacher", task=TASK)
        compile_model(teacher_model, lr=hp["eta"], loss=loss_fn, task=TASK)

    log_event("iot", iot=IOT_ID, event="start", edge=EDGE_URI, target=TARGET)

    if len(ytr) == 0:
        log_event("iot", iot=IOT_ID, event="no_data")
        return
    if not _finite_ok(Xtr, ytr):
        log_event("iot", iot=IOT_ID, event="invalid_data", split="train")
        return

    # data stats (sanity check)
    log_event(
        "iot",
        iot=IOT_ID,
        event="data_stats",
        x_mean=float(np.mean(Xtr)),
        x_std=float(np.std(Xtr)),
        y_mean=float(np.mean(ytr)),
        y_std=float(np.std(ytr)),
        n_train=int(len(ytr)),
        n_val=int(len(yv)),
        n_test=int(len(yt)),
        target=TARGET,
        task=TASK,
        distill_alpha=DISTILL_ALPHA if DISTILL_ALPHA > 0 else None,
        distill_alpha_dynamic=DISTILL_ALPHA_DYNAMIC,
    )

    backoff = 1.0
    while round_ctr < hp["T"]:
        ws = None
        try:
            ws = await ws_connect(EDGE_URI)
            backoff = 1.0
            # hello
            await ws.send(json.dumps({"type": "hello", "iot": IOT_ID, "target": TARGET}))
            msg = json.loads(await ws.recv())
            assert msg["type"] == "init"
            if IOT_WAIT_FOR_READY and msg.get("ready") is False:
                while True:
                    wait_msg = json.loads(await ws.recv())
                    if wait_msg.get("type") == "start":
                        break
            wg = np.frombuffer(bytes.fromhex(msg["whex"]), dtype=np.float32)
            set_weights_vector(local_model, wg)
            w_local = wg.copy()
            if teacher_model is not None and msg.get("teacher_whex"):
                teacher_weights = np.frombuffer(bytes.fromhex(msg["teacher_whex"]), dtype=np.float32)
                set_weights_vector(teacher_model, teacher_weights)
                teacher_version = msg.get("teacher_version")
                teacher_pred_cache = None
                teacher_score = msg.get("teacher_score")

            while round_ctr < hp["T"]:
                round_ctr += 1
                round_start = time.time()

                # treino local
                t0 = time.time()
                y_train = ytr
                distill_rmse = None
                distill_mse = None
                teacher_preds = None
                alpha_eff = _distill_alpha_effective(DISTILL_ALPHA, teacher_score)
                if teacher_model is not None and teacher_weights is not None and alpha_eff > 0:
                    if teacher_pred_cache is None:
                        teacher_pred_cache = _safe_predict(teacher_model, Xtr)
                    teacher_preds = teacher_pred_cache
                    if teacher_preds is not None:
                        y_train = (1.0 - alpha_eff) * ytr + alpha_eff * teacher_preds
                local_model.fit(
                    Xtr,
                    y_train,
                    batch_size=hp["B"],
                    epochs=hp["E"],
                    verbose=0,
                    callbacks=[tf.keras.callbacks.TerminateOnNaN()],
                )
                train_time_ms = (time.time() - t0) * 1000.0
                w_new = get_weights_vector(local_model)

                # se pesos ficaram inválidos, reverte para o último válido
                if not np.isfinite(w_new).all():
                    log_event("iot", iot=IOT_ID, event="nan_weights", round=round_ctr)
                    set_weights_vector(local_model, w_local)
                    w_new = w_local.copy()
                    train_loss = train_mae = train_rmse = train_r2 = train_mape = train_score = None
                    train_acc = train_precision = train_recall = train_f1 = None
                else:
                    train_metrics = _predict_metrics(local_model, Xtr, ytr, TARGET, TASK)
                    if TASK == "classification":
                        train_loss = train_metrics.get("bce")
                        train_mae = train_metrics.get("mae")
                        train_rmse = train_metrics.get("brier_rmse")
                        train_acc = train_metrics.get("acc")
                        train_precision = train_metrics.get("precision")
                        train_recall = train_metrics.get("recall")
                        train_f1 = train_metrics.get("f1")
                        train_r2 = train_mape = None
                        train_score = train_acc
                    else:
                        train_loss = train_metrics.get("mse")
                        train_mae = train_metrics.get("mae")
                        train_rmse = train_metrics.get("rmse")
                        train_r2 = train_metrics.get("r2")
                        train_mape = train_metrics.get("mape")
                        if train_loss is not None and not np.isfinite(train_loss):
                            train_loss = train_mae = train_rmse = train_r2 = train_mape = None
                        train_score = score_from_rmse(train_rmse)
                        train_acc = train_precision = train_recall = train_f1 = None
                    if teacher_preds is not None:
                        student_preds = _safe_predict(local_model, Xtr)
                        if student_preds is not None:
                            distill_mse = float(np.mean((student_preds - teacher_preds) ** 2))
                            distill_rmse = float(np.sqrt(distill_mse))

                # opcional: validação se existir
                val_loss = None
                val_mae = None
                val_rmse = None
                val_r2 = None
                val_mape = None
                val_score = None
                val_acc = None
                val_precision = None
                val_recall = None
                val_f1 = None
                if len(yv) > 0 and _finite_ok(Xv, yv):
                    val_metrics = _predict_metrics(local_model, Xv, yv, TARGET, TASK)
                    if TASK == "classification":
                        val_loss = val_metrics.get("bce")
                        val_mae = val_metrics.get("mae")
                        val_rmse = val_metrics.get("brier_rmse")
                        val_acc = val_metrics.get("acc")
                        val_precision = val_metrics.get("precision")
                        val_recall = val_metrics.get("recall")
                        val_f1 = val_metrics.get("f1")
                        val_r2 = val_mape = None
                        val_score = val_acc
                    else:
                        val_loss = val_metrics.get("mse")
                        val_mae = val_metrics.get("mae")
                        val_rmse = val_metrics.get("rmse")
                        val_r2 = val_metrics.get("r2")
                        val_mape = val_metrics.get("mape")
                        if val_loss is not None and not np.isfinite(val_loss):
                            val_loss = val_mae = val_rmse = val_r2 = val_mape = None
                        val_score = score_from_rmse(val_rmse)

                # métricas principais por tipo de tarefa (para console + análise)
                reg_metric = None
                class_metric = None
                if TASK == "classification":
                    class_metric = val_acc if val_acc is not None else train_acc
                else:
                    reg_metric = val_rmse if val_rmse is not None else train_rmse

                def _fmt_metric(v):
                    if v is None:
                        return "n/a"
                    try:
                        fv = float(v)
                    except Exception:
                        return "n/a"
                    if not np.isfinite(fv):
                        return "n/a"
                    return f"{fv:.4f}"

                print(
                    f"[iot] {IOT_ID} round={round_ctr} target={TARGET} "
                    f"reg_rmse={_fmt_metric(reg_metric)} class_acc={_fmt_metric(class_metric)}"
                )

                # delay heterogêneo ~ N(mu, sigma)
                delay_s = max(0.0, random.gauss(hp["delay_mu"], hp["delay_sigma"]))
                if delay_s > 0:
                    await asyncio.sleep(delay_s)

                payload = w_new.astype("float32").tobytes()
                enc_start = time.time()
                enc = encrypt(IOT_KEY_HEX, payload)
                enc_ms = (time.time() - enc_start) * 1000.0
                await ws.send(json.dumps({
                    "type": "update",
                    "iot": IOT_ID,
                    "target": TARGET,
                    "round": round_ctr,
                    "bytes": b64e(enc),
                    "train_score": train_score,
                    "train_loss": train_loss,
                    "train_mae": train_mae,
                    "train_rmse": train_rmse,
                    "train_r2": train_r2,
                    "train_mape": train_mape,
                    "train_acc": train_acc,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1": train_f1,
                    "distill_alpha": alpha_eff if alpha_eff > 0 else None,
                    "distill_rmse": distill_rmse,
                    "distill_mse": distill_mse,
                    "n_train": int(len(ytr)),
                    "val_score": val_score,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2,
                    "val_mape": val_mape,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "payload_bytes": len(enc)
                }))

                # recebe modelo atualizado do edge
                reply = json.loads(await ws.recv())
                assert reply["type"] == "model"
                w_edge = np.frombuffer(bytes.fromhex(reply["whex"]), dtype=np.float32)
                set_weights_vector(local_model, w_edge)
                w_local = w_edge.copy()
                if teacher_model is not None and reply.get("teacher_whex"):
                    new_version = reply.get("teacher_version")
                    if new_version != teacher_version:
                        teacher_weights = np.frombuffer(bytes.fromhex(reply["teacher_whex"]), dtype=np.float32)
                        set_weights_vector(teacher_model, teacher_weights)
                        teacher_version = new_version
                        teacher_pred_cache = None
                        teacher_score = reply.get("teacher_score")

                round_time_ms = (time.time() - round_start) * 1000.0
                w_stats = _weight_stats(w_new)
                log_metric(
                    "iot",
                    iot=IOT_ID,
                    target=TARGET,
                    task=TASK,
                    round=round_ctr,
                    train_score=train_score,
                    train_loss=train_loss,
                    train_mae=train_mae,
                    train_rmse=train_rmse,
                    train_r2=train_r2,
                    train_mape=train_mape,
                    train_acc=train_acc,
                    train_precision=train_precision,
                    train_recall=train_recall,
                    train_f1=train_f1,
                    distill_alpha=alpha_eff if alpha_eff > 0 else None,
                    distill_rmse=distill_rmse,
                    distill_mse=distill_mse,
                    val_score=val_score,
                    val_loss=val_loss,
                    val_mae=val_mae,
                    val_rmse=val_rmse,
                    val_r2=val_r2,
                    val_mape=val_mape,
                    val_acc=val_acc,
                    val_precision=val_precision,
                    val_recall=val_recall,
                    val_f1=val_f1,
                    reg_metric=reg_metric,
                    class_metric=class_metric,
                    train_time_ms=train_time_ms,
                    enc_ms=enc_ms,
                    delay_ms=delay_s * 1000.0,
                    round_time_ms=round_time_ms,
                    payload_bytes=len(enc),
                    **w_stats,
                )
        except Exception as e:
            log_event("iot", iot=IOT_ID, event="reconnect", error=str(e))
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
        finally:
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass

def main():
    asyncio.run(run_iot())

if __name__ == "__main__":
    main()
