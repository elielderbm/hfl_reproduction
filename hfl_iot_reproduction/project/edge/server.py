import os, asyncio, json, time, numpy as np, random
from collections import deque

from project.common.messaging import ws_server, ws_connect, b64d
from project.common.crypto import encrypt, decrypt
from project.common.model import build_model, compile_model, get_weights_vector, set_weights_vector
from project.common.logging_utils import log_event, log_metric
from project.common.metrics import score_from_rmse, regression_metrics, classification_metrics
from project.common.config import load_hparams
from project.common.dataset_config import load_dataset_config, task_for_target
from project.common.data_utils import load_split_by_target, unscale_target

EDGE_ID = os.getenv("EDGE_ID","edge1")
EDGE_KEY_HEX = os.getenv("EDGE_KEY_HEX","00112233445566778899AABBCCDDEEFF")
EDGE_PORT = 8765
CLOUD_URI = "ws://cloud:9000"
SYNC_MODE = os.getenv("SYNC_MODE", "0") == "1"
EDGE_CLIENTS = {c.strip() for c in os.getenv("EDGE_CLIENTS", "").split(",") if c.strip()}
EDGE_TARGET = os.getenv("EDGE_TARGET")
EDGE_WEIGHT_BY_SAMPLES = os.getenv("EDGE_WEIGHT_BY_SAMPLES", "0") == "1"
EDGE_EXPECTED_IOTS = {c.strip() for c in os.getenv("EDGE_EXPECTED_IOTS", "").split(",") if c.strip()}
EDGE_WAIT_FOR_IOTS = os.getenv("EDGE_WAIT_FOR_IOTS", "1") == "1"
EDGE_WAIT_FOR_CLOUD = os.getenv("EDGE_WAIT_FOR_CLOUD", "1") == "1"
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED","42"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)

hp = load_hparams()
cfg = load_dataset_config()
targets_map = cfg.get("targets", {})

if not EDGE_TARGET:
    inferred = {targets_map.get(iot) for iot in EDGE_CLIENTS} if EDGE_CLIENTS else set()
    inferred.discard(None)
    if len(inferred) == 1:
        EDGE_TARGET = inferred.pop()
    else:
        EDGE_TARGET = "temp"
        print(f"[edge:{EDGE_ID}] EDGE_TARGET não definido e inferência ambígua; usando '{EDGE_TARGET}'.")
EDGE_TASK = task_for_target(EDGE_TARGET, cfg)

# State
clients = set()
pending_iot_ws = set()
cloud_ready = False
model = None
wvec0 = None
w_edge = None
pactual = 0.0
window = float(hp["sw_init"])
alpha_sw = float(hp["alpha_sw"])
alpha_edge = float(hp["alpha_edge"])
beta_edge_base = float(hp["beta_edge"])
beta_edge_eff = float(hp["beta_edge"])
gamma_edge = float(hp.get("gamma_edge", 0.9))
qedge = 1.0
ve_feedback = 1.0
last_agg_ts = 0.0
teacher_weights = None
teacher_version = None
teacher_score = None
edge_round_ctr = 0

# Edge fine-tuning (student on proxy data)
EDGE_FT_EPOCHS = int(os.getenv("EDGE_FT_EPOCHS", "0"))
EDGE_FT_BATCH = int(os.getenv("EDGE_FT_BATCH", "128"))
EDGE_FT_EVERY = int(os.getenv("EDGE_FT_EVERY", "0"))
EDGE_FT_MAX_SAMPLES = int(os.getenv("EDGE_FT_MAX_SAMPLES", "0"))
EDGE_FT_ALPHA = float(os.getenv("EDGE_FT_ALPHA", "0.0"))
edge_ft_data = None
edge_ft_model = None
EDGE_TEACHER_ENABLE = os.getenv("EDGE_TEACHER_ENABLE", "0") == "1"
EDGE_TEACHER_EPOCHS = int(os.getenv("EDGE_TEACHER_EPOCHS", "5"))
EDGE_TEACHER_BATCH = int(os.getenv("EDGE_TEACHER_BATCH", "128"))
EDGE_TEACHER_LR = float(os.getenv("EDGE_TEACHER_LR", "0.001"))
EDGE_TEACHER_MAX_SAMPLES = int(os.getenv("EDGE_TEACHER_MAX_SAMPLES", "0"))
EDGE_TEACHER_REFRESH_EVERY = int(os.getenv("EDGE_TEACHER_REFRESH_EVERY", "0"))
EDGE_DISTILL_SOURCE = os.getenv("EDGE_DISTILL_SOURCE", "auto").strip().lower()
if EDGE_DISTILL_SOURCE not in ("auto", "edge", "cloud"):
    EDGE_DISTILL_SOURCE = "auto"
edge_teacher_model = None
edge_teacher_metrics = {}
cloud_teacher_model = None

# Buffers
updates = deque()  # (ts, iot, round, w, train_score, train_loss, dec_ms, payload_bytes, n_train)
pending_by_round = {}
iot_train_score = {}


def _parse_q_fixed(raw):
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower()
    if s in ("none", "null", ""):
        return None
    return float(s)


Q_FIXED = _parse_q_fixed(hp.get("q_fixed"))

if not EDGE_EXPECTED_IOTS:
    EDGE_EXPECTED_IOTS = set(EDGE_CLIENTS)


def _edge_ready() -> bool:
    if not EDGE_WAIT_FOR_IOTS or not EDGE_EXPECTED_IOTS:
        iot_ready = True
    else:
        iot_ready = EDGE_EXPECTED_IOTS.issubset(clients)
    if EDGE_WAIT_FOR_CLOUD and not cloud_ready:
        return False
    return iot_ready


async def _notify_start():
    if not _edge_ready():
        return
    if not pending_iot_ws:
        return
    for ws in list(pending_iot_ws):
        try:
            await ws.send(json.dumps({"type": "start"}))
        except Exception:
            pass
        pending_iot_ws.discard(ws)


def _edge_finetune(weights: np.ndarray):
    global edge_ft_model, edge_teacher_model, cloud_teacher_model
    if EDGE_FT_EPOCHS <= 0 or not edge_ft_data:
        return weights, None
    Xp, yp = edge_ft_data
    if Xp is None or len(yp) == 0:
        return weights, None
    if edge_ft_model is None:
        edge_ft_model = build_model(kind="student", task=EDGE_TASK)
        loss_fn = os.getenv("LOSS_FN_CLASSIF", "binary_crossentropy") if EDGE_TASK == "classification" else os.getenv("LOSS_FN", "mse")
        compile_model(edge_ft_model, lr=hp["eta"], loss=loss_fn, task=EDGE_TASK)
    set_weights_vector(edge_ft_model, weights)

    y_train = yp
    alpha = max(0.0, min(1.0, EDGE_FT_ALPHA))
    if alpha > 0:
        tpred = None
        if EDGE_DISTILL_SOURCE in ("auto", "edge") and edge_teacher_model is not None:
            try:
                tpred = edge_teacher_model.predict(Xp, verbose=0).reshape(-1)
            except Exception:
                tpred = None
        if (tpred is None or not np.isfinite(tpred).all()) and EDGE_DISTILL_SOURCE in ("auto", "cloud"):
            if teacher_weights is not None:
                if cloud_teacher_model is None:
                    cloud_teacher_model = build_model(kind="cloud_teacher", task=EDGE_TASK)
                    loss_fn = os.getenv("LOSS_FN_CLASSIF", "binary_crossentropy") if EDGE_TASK == "classification" else os.getenv("LOSS_FN", "mse")
                    compile_model(cloud_teacher_model, lr=hp["eta"], loss=loss_fn, task=EDGE_TASK)
                set_weights_vector(cloud_teacher_model, teacher_weights)
                try:
                    tpred = cloud_teacher_model.predict(Xp, verbose=0).reshape(-1)
                except Exception:
                    tpred = None
        if tpred is not None and np.isfinite(tpred).all():
            y_train = (1.0 - alpha) * yp + alpha * tpred

    edge_ft_model.fit(
        Xp,
        y_train,
        batch_size=EDGE_FT_BATCH,
        epochs=max(1, EDGE_FT_EPOCHS),
        verbose=0,
    )
    new_w = get_weights_vector(edge_ft_model)

    try:
        preds = edge_ft_model.predict(Xp, verbose=0).reshape(-1)
    except Exception:
        return new_w, None
    if not np.isfinite(preds).all():
        return new_w, None
    if EDGE_TASK == "classification":
        metrics = classification_metrics(yp, preds)
        return new_w, {
            "edge_ft_loss": metrics.get("bce"),
            "edge_ft_mae": metrics.get("mae"),
            "edge_ft_rmse": metrics.get("brier_rmse"),
            "edge_ft_score": metrics.get("acc"),
            "edge_ft_acc": metrics.get("acc"),
            "edge_ft_precision": metrics.get("precision"),
            "edge_ft_recall": metrics.get("recall"),
            "edge_ft_f1": metrics.get("f1"),
            "edge_ft_alpha": alpha,
            "edge_ft_samples": int(len(yp)),
        }
    yt_u = unscale_target(yp, EDGE_TARGET)
    preds_u = unscale_target(preds, EDGE_TARGET)
    mse, mae, rmse, r2, mape = regression_metrics(yt_u, preds_u)
    return new_w, {
        "edge_ft_loss": mse,
        "edge_ft_mae": mae,
        "edge_ft_rmse": rmse,
        "edge_ft_r2": r2,
        "edge_ft_mape": mape,
        "edge_ft_score": score_from_rmse(rmse),
        "edge_ft_alpha": alpha,
        "edge_ft_samples": int(len(yp)),
    }


def _train_edge_teacher():
    Xtr, ytr = load_split_by_target(EDGE_TARGET, split="train", max_samples=EDGE_TEACHER_MAX_SAMPLES or None)
    if len(ytr) == 0:
        return None, None
    model = build_model(kind="edge_teacher", task=EDGE_TASK)
    loss_fn = os.getenv("LOSS_FN_CLASSIF", "binary_crossentropy") if EDGE_TASK == "classification" else os.getenv("LOSS_FN", "mse")
    compile_model(model, lr=EDGE_TEACHER_LR, loss=loss_fn, task=EDGE_TASK)
    model.fit(
        Xtr,
        ytr,
        batch_size=EDGE_TEACHER_BATCH,
        epochs=max(1, EDGE_TEACHER_EPOCHS),
        verbose=0,
    )
    # Avaliação no teste (se existir)
    Xt, yt = load_split_by_target(EDGE_TARGET, split="test")
    metrics = {}
    if len(yt) > 0:
        try:
            preds = model.predict(Xt, verbose=0).reshape(-1)
        except Exception:
            preds = None
        if preds is not None and np.isfinite(preds).all():
            if EDGE_TASK == "classification":
                m = classification_metrics(yt, preds)
                metrics = {
                    "edge_teacher_loss": m.get("bce"),
                    "edge_teacher_mae": m.get("mae"),
                    "edge_teacher_rmse": m.get("brier_rmse"),
                    "edge_teacher_score": m.get("acc"),
                    "edge_teacher_acc": m.get("acc"),
                    "edge_teacher_precision": m.get("precision"),
                    "edge_teacher_recall": m.get("recall"),
                    "edge_teacher_f1": m.get("f1"),
                }
            else:
                yt_u = unscale_target(yt, EDGE_TARGET)
                preds_u = unscale_target(preds, EDGE_TARGET)
                mse, mae, rmse, r2, mape = regression_metrics(yt_u, preds_u)
                metrics = {
                    "edge_teacher_loss": mse,
                    "edge_teacher_mae": mae,
                    "edge_teacher_rmse": rmse,
                    "edge_teacher_r2": r2,
                    "edge_teacher_mape": mape,
                    "edge_teacher_score": score_from_rmse(rmse),
                }
    return model, metrics

def iot_key_for(iot_id: str) -> str:
    """
    Retorna a chave hex para decriptar o update do IoT.
    Permite IOT1_KEY, IOT2_KEY... e fallback para IOT_KEY_HEX (compatível com o default do IoT).
    """
    env_key = f"{iot_id.upper()}_KEY"
    return os.getenv(env_key, os.getenv("IOT_KEY_HEX", "00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF"))

async def cloud_loop():
    """
    Conecta ao Cloud, realiza handshake, envia agregações do Edge periodicamente e
    recebe o modelo global atualizado. Reexecuta em loop se a conexão cair.
    """
    global w_edge, pactual, window, beta_edge_eff, ve_feedback, qedge, last_agg_ts, teacher_weights, teacher_version, teacher_score, edge_round_ctr, cloud_ready, edge_teacher_model, edge_teacher_metrics

    backoff = 1.0
    print(f"[edge:{EDGE_ID}] iniciado na porta {EDGE_PORT}")
    while True:
        try:
            print(f"[edge:{EDGE_ID}] tentando conectar ao Cloud em {CLOUD_URI}...")
            cw = await ws_connect(CLOUD_URI)
            print(f"[edge:{EDGE_ID}] conectado ao Cloud!")
            try:
                # Handshake: informar o edge e receber pesos iniciais
                await cw.send(json.dumps({"type": "hello", "edge": EDGE_ID, "target": EDGE_TARGET}))
                msg = json.loads(await cw.recv())
                assert msg["type"] == "init", f"Esperado 'init', recebido {msg.get('type')}"
                we = np.frombuffer(bytes.fromhex(msg["whex"]), dtype=np.float32)
                w_edge = we
                if msg.get("teacher_whex"):
                    teacher_weights = np.frombuffer(bytes.fromhex(msg["teacher_whex"]), dtype=np.float32)
                    teacher_version = msg.get("teacher_version")
                    teacher_score = msg.get("teacher_score")
                print(f"[edge:{EDGE_ID}] handshake concluído com Cloud, modelo inicial recebido ({w_edge.shape[0]} pesos)")
                cloud_ready = True
                await _notify_start()

                # Loop principal: agrega e envia para a cloud
                last_agg_ts = time.time()
                while True:
                    await asyncio.sleep(0.1)

                    A = []
                    now = time.time()

                    if not _edge_ready():
                        continue

                    if SYNC_MODE:
                        expected = EDGE_EXPECTED_IOTS or EDGE_CLIENTS or clients
                        if not expected:
                            continue
                        ready_rounds = [
                            r for r, bucket in pending_by_round.items()
                            if expected.issubset(bucket.keys())
                        ]
                        if not ready_rounds:
                            continue
                        r_pick = min(ready_rounds)
                        bucket = pending_by_round.pop(r_pick)
                        A = list(bucket.values())
                        pactual = 1.0
                    else:
                        if len(updates) == 0:
                            continue
                        if (now - last_agg_ts) < max(1.0, window):
                            continue

                        # consumir updates recebidos desde a última agregação
                        while updates and updates[0][0] <= last_agg_ts:
                            updates.popleft()
                        while updates and updates[0][0] <= now:
                            A.append(updates.popleft())
                        if len(A) == 0:
                            last_agg_ts = now
                            continue

                        # pactual = fração de IoTs distintos que contribuíram na janela
                        contrib = {u[1] for u in A}
                        if len(clients) > 0:
                            pactual = len(contrib) / max(1, len(clients))

                    agg_start = time.time()
                    sum_w = np.zeros_like(w_edge, dtype=np.float32)
                    scores = []
                    dec_ms_vals = []
                    payload_bytes = 0
                    n_eff = 0.0
                    for (_ts, _iot, _r, _w, _score, _loss, _dec_ms, _bytes, _n) in A:
                        weight = float(_n) if EDGE_WEIGHT_BY_SAMPLES and _n is not None else 1.0
                        sum_w += _w * weight
                        n_eff += weight
                        if _score is not None:
                            scores.append(float(_score))
                        if _dec_ms is not None:
                            dec_ms_vals.append(float(_dec_ms))
                        if _bytes is not None:
                            payload_bytes += int(_bytes)

                    # Estimativa de qualidade (q_current)
                    qcurrent = float(np.mean(scores)) if scores else 1.0
                    if Q_FIXED is not None:
                        qedge = float(Q_FIXED)
                    else:
                        qedge = gamma_edge * qedge + (1.0 - gamma_edge) * qcurrent

                    # Agregação: async (com histórico) ou sync (HierFAVG)
                    if SYNC_MODE:
                        denom = max(1.0, n_eff if EDGE_WEIGHT_BY_SAMPLES else len(A))
                        w_edge = sum_w / denom
                    else:
                        denom = alpha_edge + beta_edge_eff * (n_eff if EDGE_WEIGHT_BY_SAMPLES else len(A))
                        w_edge = (alpha_edge * w_edge + beta_edge_eff * sum_w) / denom
                    agg_ms = (time.time() - agg_start) * 1000.0

                    # Criptografar e enviar ao Cloud
                    payload = w_edge.astype("float32").tobytes()
                    enc = encrypt(EDGE_KEY_HEX, payload)
                    await cw.send(json.dumps({
                        "type": "edge_model",
                        "edge": EDGE_ID,
                        "target": EDGE_TARGET,
                        "pactual": pactual,
                        "qedge": qedge,
                        "bytes": enc.hex(),
                        "payload_bytes": len(enc)
                    }))

                    # Receber modelo global atualizado + feedback
                    reply = json.loads(await cw.recv())
                    assert reply["type"] == "model", f"Esperado 'model', recebido {reply.get('type')}"
                    wg = np.frombuffer(bytes.fromhex(reply["whex"]), dtype=np.float32)
                    w_edge = wg  # Edge segue o global
                    if reply.get("teacher_whex"):
                        new_version = reply.get("teacher_version")
                        if new_version != teacher_version:
                            teacher_weights = np.frombuffer(bytes.fromhex(reply["teacher_whex"]), dtype=np.float32)
                            teacher_version = new_version
                        teacher_score = reply.get("teacher_score")

                    ve_feedback = float(reply.get("ve", ve_feedback))
                    ve_feedback = max(0.1, min(2.0, ve_feedback))
                    beta_edge_eff = beta_edge_base * ve_feedback

                    edge_round_ctr += 1
                    edge_ft_metrics = {}
                    if EDGE_FT_EVERY > 0 and EDGE_FT_EPOCHS > 0 and edge_round_ctr % EDGE_FT_EVERY == 0:
                        w_new, metrics_ft = _edge_finetune(w_edge)
                        if w_new is not None:
                            w_edge = w_new
                        if metrics_ft:
                            edge_ft_metrics.update(metrics_ft)
                    if EDGE_TEACHER_ENABLE and EDGE_TEACHER_REFRESH_EVERY > 0 and edge_round_ctr % EDGE_TEACHER_REFRESH_EVERY == 0:
                        model_t, metrics_t = _train_edge_teacher()
                        if model_t is not None:
                            edge_teacher_model = model_t
                            edge_teacher_metrics = metrics_t or {}
                            log_event("edge", edge=EDGE_ID, event="edge_teacher_trained", target=EDGE_TARGET, **edge_teacher_metrics)

                    # Ajustar janela deslizante localmente (somente no modo assíncrono)
                    if not SYNC_MODE:
                        window = max(3.0, float(window + alpha_sw * (hp["pdesired"] - pactual)))
                    last_agg_ts = now

                    # Log de métricas do edge
                    log_metric(
                        "edge",
                        edge=EDGE_ID,
                        target=EDGE_TARGET,
                        task=EDGE_TASK,
                        window=window,
                        pactual=pactual,
                        qcurrent=qcurrent,
                        qedge=qedge,
                        ve=ve_feedback,
                        beta_edge=beta_edge_eff,
                        buf=len(A),
                        n_eff=n_eff,
                        payload_bytes=payload_bytes,
                        dec_ms_mean=float(np.mean(dec_ms_vals)) if dec_ms_vals else None,
                        agg_ms=agg_ms,
                        **edge_ft_metrics,
                    )

            finally:
                try:
                    await cw.close()
                    print(f"[edge:{EDGE_ID}] conexão fechada com Cloud")
                except Exception:
                    pass
                cloud_ready = False

            # Se saiu do loop sem exceção, resetar backoff
            backoff = 1.0

        except Exception as e:
            print(f"[edge:{EDGE_ID}] erro no loop Cloud: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)  # até 30s

async def iot_handler(ws, path):
    """
    Handler de IoTs conectando no Edge.
    Handshake: IoT manda hello; Edge responde com pesos iniciais.
    Depois recebe 'update' (pesos + métricas) e devolve modelo do edge (assíncrono).
    """
    global clients, w_edge

    # Handshake: aguarda hello do IoT e responde com init (pesos atuais do edge)
    raw = await ws.recv()
    hello = json.loads(raw)
    iot = hello.get("iot")
    msg_target = hello.get("target") or (targets_map.get(iot) if iot else None)
    if EDGE_TARGET and msg_target and msg_target != EDGE_TARGET:
        print(f"[edge:{EDGE_ID}] IoT {iot} target={msg_target} incompatível com EDGE_TARGET={EDGE_TARGET}")
    if iot:
        clients.add(iot)
    while w_edge is None:
        await asyncio.sleep(0.1)
    await ws.send(json.dumps({
        "type": "init",
        "whex": w_edge.tobytes().hex(),
        "target": EDGE_TARGET,
        "teacher_whex": teacher_weights.tobytes().hex() if teacher_weights is not None else None,
        "teacher_version": teacher_version,
        "teacher_score": teacher_score,
        "ready": _edge_ready(),
        "expected_iots": sorted(EDGE_EXPECTED_IOTS) if EDGE_EXPECTED_IOTS else None,
    }))
    if not _edge_ready():
        pending_iot_ws.add(ws)
    await _notify_start()
    print(f"[edge:{EDGE_ID}] IoT conectado")

    while True:
        msg = await ws.recv()
        M = json.loads(msg)
        if M["type"] == "update":
            iot = M["iot"]
            r = M["round"]
            msg_target = M.get("target") or targets_map.get(iot)
            if EDGE_TARGET and msg_target and msg_target != EDGE_TARGET:
                print(f"[edge:{EDGE_ID}] update descartado de {iot} (target {msg_target} != {EDGE_TARGET})")
                continue

            # 🔐 DECRIPTA o payload vindo do IoT (antes era lido direto!)
            try:
                enc = b64d(M["bytes"])
                key_hex = iot_key_for(iot)
                dec_start = time.time()
                dec = decrypt(key_hex, enc)
                dec_ms = (time.time() - dec_start) * 1000.0
                w = np.frombuffer(dec, dtype=np.float32)
            except Exception as e:
                print(f"[edge:{EDGE_ID}] erro decriptando update de {iot}: {e} (descartado)")
                continue

            # Checagem de shape (evita quebrar agregação)
            if w_edge is not None and w.shape != w_edge.shape:
                print(f"[edge:{EDGE_ID}] update inválido de {iot}: shape={w.shape} esperado={w_edge.shape} (descartado)")
                continue

            train_score = M.get("train_score")
            train_loss = M.get("train_loss")
            train_rmse = M.get("train_rmse")
            if train_score is None and train_rmse is not None:
                train_score = score_from_rmse(float(train_rmse))
            if train_score is None and train_loss is not None:
                # fallback: RMSE ~ sqrt(MSE)
                try:
                    train_score = score_from_rmse(float(train_loss) ** 0.5)
                except Exception:
                    train_score = None
            if train_score is None:
                train_score = M.get("val_score") or M.get("train_acc") or M.get("val_acc")
            if train_loss is None:
                train_loss = M.get("val_loss")
            clients.add(iot)
            n_train = M.get("n_train")
            item = (time.time(), iot, r, w, train_score, train_loss, dec_ms, M.get("payload_bytes"), n_train)
            if SYNC_MODE:
                bucket = pending_by_round.setdefault(r, {})
                bucket[iot] = item
            else:
                updates.append(item)
            iot_train_score[iot] = train_score
            if train_score is not None:
                print(f"[edge:{EDGE_ID}] update recebido de {iot}, round={r}, train_score={float(train_score):.3f}")
            else:
                print(f"[edge:{EDGE_ID}] update recebido de {iot}, round={r}")
            # Envia imediatamente o modelo do edge (pode estar "stale", pois é assíncrono)
            await ws.send(json.dumps({
                "type": "model",
                "whex": w_edge.tobytes().hex(),
                "teacher_whex": teacher_weights.tobytes().hex() if teacher_weights is not None else None,
                "teacher_version": teacher_version,
                "teacher_score": teacher_score,
            }))

async def main():
    global model, wvec0, w_edge, edge_ft_data, edge_teacher_model, edge_teacher_metrics
    # Inicializa um modelo para obter formas e pesos iniciais
    tmp = build_model(kind="student", task=EDGE_TASK)
    loss_fn = os.getenv("LOSS_FN_CLASSIF", "binary_crossentropy") if EDGE_TASK == "classification" else os.getenv("LOSS_FN", "mse")
    compile_model(tmp, lr=hp["eta"], loss=loss_fn, task=EDGE_TASK)
    wvec0 = get_weights_vector(tmp)
    w_edge = wvec0.copy()

    if EDGE_FT_EPOCHS > 0:
        Xp, yp = load_split_by_target(EDGE_TARGET, split="train", max_samples=EDGE_FT_MAX_SAMPLES or None)
        if len(yp) > 0:
            edge_ft_data = (Xp, yp)

    if EDGE_TEACHER_ENABLE:
        model_t, metrics_t = _train_edge_teacher()
        if model_t is not None:
            edge_teacher_model = model_t
            edge_teacher_metrics = metrics_t or {}
            log_event("edge", edge=EDGE_ID, event="edge_teacher_trained", target=EDGE_TARGET, **edge_teacher_metrics)

    log_event("edge", edge=EDGE_ID, event="start", port=EDGE_PORT, target=EDGE_TARGET, task=EDGE_TASK)

    await asyncio.gather(
        ws_server("0.0.0.0", EDGE_PORT, iot_handler),
        cloud_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
