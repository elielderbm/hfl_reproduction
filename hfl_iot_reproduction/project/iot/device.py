import os, asyncio, json, time, numpy as np, random
import tensorflow as tf
from project.common.messaging import ws_connect, b64e
from project.common.crypto import encrypt
from project.common.model import build_model, compile_model, get_weights_vector, set_weights_vector
from project.common.logging_utils import log_event, log_metric
from project.common.metrics import score_from_rmse, regression_metrics
from project.common.config import load_hparams
from project.common.data_utils import load_client_split

IOT_ID = os.getenv("IOT_ID", "iot1")
EDGE_URI = os.getenv("EDGE_URI", "ws://edge1:8765")
IOT_KEY_HEX = os.getenv("IOT_KEY_HEX", "00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF")
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)

hp = load_hparams()

# estado
round_ctr = 0
local_model = None
w_local = None


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


def _predict_metrics(model, X, y):
    try:
        preds = model.predict(X, verbose=0).reshape(-1)
    except Exception:
        return None, None, None, None, None
    if not np.isfinite(preds).all():
        return None, None, None, None, None
    return regression_metrics(y.reshape(-1), preds)

async def run_iot():
    global round_ctr, local_model, w_local
    (Xtr, ytr), (Xv, yv), (Xt, yt) = load_client_split(IOT_ID)

    local_model = build_model()
    compile_model(local_model, lr=hp["eta"])
    w_local = get_weights_vector(local_model)

    log_event("iot", iot=IOT_ID, event="start", edge=EDGE_URI)

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
    )

    backoff = 1.0
    while round_ctr < hp["T"]:
        ws = None
        try:
            ws = await ws_connect(EDGE_URI)
            backoff = 1.0
            # hello
            await ws.send(json.dumps({"type": "hello", "iot": IOT_ID}))
            msg = json.loads(await ws.recv())
            assert msg["type"] == "init"
            wg = np.frombuffer(bytes.fromhex(msg["whex"]), dtype=np.float32)
            set_weights_vector(local_model, wg)
            w_local = wg.copy()

            while round_ctr < hp["T"]:
                round_ctr += 1
                round_start = time.time()

                # treino local
                t0 = time.time()
                local_model.fit(
                    Xtr,
                    ytr,
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
                else:
                    # métricas de treino (regressão)
                    train_loss, train_mae, train_rmse, train_r2, train_mape = _predict_metrics(local_model, Xtr, ytr)
                    if train_loss is not None and not np.isfinite(train_loss):
                        train_loss = train_mae = train_rmse = train_r2 = train_mape = None
                    train_score = score_from_rmse(train_rmse)

                # opcional: validação se existir
                val_loss = None
                val_mae = None
                val_rmse = None
                val_r2 = None
                val_mape = None
                val_score = None
                if len(yv) > 0 and _finite_ok(Xv, yv):
                    val_loss, val_mae, val_rmse, val_r2, val_mape = _predict_metrics(local_model, Xv, yv)
                    if val_loss is not None and not np.isfinite(val_loss):
                        val_loss = val_mae = val_rmse = val_r2 = val_mape = None
                    val_score = score_from_rmse(val_rmse)

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
                    "round": round_ctr,
                    "bytes": b64e(enc),
                    "train_score": train_score,
                    "train_loss": train_loss,
                    "train_mae": train_mae,
                    "train_rmse": train_rmse,
                    "train_r2": train_r2,
                    "train_mape": train_mape,
                    "val_score": val_score,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2,
                    "val_mape": val_mape,
                    "payload_bytes": len(enc)
                }))

                # recebe modelo atualizado do edge
                reply = json.loads(await ws.recv())
                assert reply["type"] == "model"
                w_edge = np.frombuffer(bytes.fromhex(reply["whex"]), dtype=np.float32)
                set_weights_vector(local_model, w_edge)
                w_local = w_edge.copy()

                round_time_ms = (time.time() - round_start) * 1000.0
                w_stats = _weight_stats(w_new)
                log_metric(
                    "iot",
                    iot=IOT_ID,
                    round=round_ctr,
                    train_score=train_score,
                    train_loss=train_loss,
                    train_mae=train_mae,
                    train_rmse=train_rmse,
                    train_r2=train_r2,
                    train_mape=train_mape,
                    val_score=val_score,
                    val_loss=val_loss,
                    val_mae=val_mae,
                    val_rmse=val_rmse,
                    val_r2=val_r2,
                    val_mape=val_mape,
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
