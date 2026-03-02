import os, asyncio, json, time, numpy as np, random
from collections import deque

from project.common.messaging import ws_server, ws_connect, b64d
from project.common.crypto import encrypt, decrypt
from project.common.model import build_model, compile_model, get_weights_vector, set_weights_vector
from project.common.logging_utils import log_event, log_metric
from project.common.config import load_hparams

EDGE_ID = os.getenv("EDGE_ID","edge1")
EDGE_KEY_HEX = os.getenv("EDGE_KEY_HEX","00112233445566778899AABBCCDDEEFF")
EDGE_PORT = 8765
CLOUD_URI = "ws://cloud:9000"
SYNC_MODE = os.getenv("SYNC_MODE", "0") == "1"
EDGE_CLIENTS = {c.strip() for c in os.getenv("EDGE_CLIENTS", "").split(",") if c.strip()}
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED","42"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)

hp = load_hparams()

# State
clients = set()
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

# Buffers
updates = deque()  # (ts, iot, round, w, train_acc, train_loss, dec_ms, payload_bytes)
pending_by_round = {}
iot_train_acc = {}


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
    global w_edge, pactual, window, beta_edge_eff, ve_feedback, qedge, last_agg_ts

    backoff = 1.0
    print(f"[edge:{EDGE_ID}] iniciado na porta {EDGE_PORT}")
    while True:
        try:
            print(f"[edge:{EDGE_ID}] tentando conectar ao Cloud em {CLOUD_URI}...")
            cw = await ws_connect(CLOUD_URI)
            print(f"[edge:{EDGE_ID}] conectado ao Cloud!")
            try:
                # Handshake: informar o edge e receber pesos iniciais
                await cw.send(json.dumps({"type": "hello", "edge": EDGE_ID}))
                msg = json.loads(await cw.recv())
                assert msg["type"] == "init", f"Esperado 'init', recebido {msg.get('type')}"
                we = np.frombuffer(bytes.fromhex(msg["whex"]), dtype=np.float32)
                w_edge = we
                print(f"[edge:{EDGE_ID}] handshake concluído com Cloud, modelo inicial recebido ({w_edge.shape[0]} pesos)")

                # Loop principal: agrega e envia para a cloud
                last_agg_ts = time.time()
                while True:
                    await asyncio.sleep(0.1)

                    A = []
                    now = time.time()

                    if SYNC_MODE:
                        expected = EDGE_CLIENTS or clients
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
                    accs = []
                    dec_ms_vals = []
                    payload_bytes = 0
                    for (_ts, _iot, _r, _w, _acc, _loss, _dec_ms, _bytes) in A:
                        sum_w += _w
                        if _acc is not None:
                            accs.append(float(_acc))
                        if _dec_ms is not None:
                            dec_ms_vals.append(float(_dec_ms))
                        if _bytes is not None:
                            payload_bytes += int(_bytes)

                    # Estimativa de qualidade (q_current)
                    qcurrent = float(np.mean(accs)) if accs else 1.0
                    if Q_FIXED is not None:
                        qedge = float(Q_FIXED)
                    else:
                        qedge = gamma_edge * qedge + (1.0 - gamma_edge) * qcurrent

                    # Agregação: async (com histórico) ou sync (HierFAVG)
                    if SYNC_MODE:
                        w_edge = sum_w / max(1, len(A))
                    else:
                        denom = alpha_edge + beta_edge_eff * len(A)
                        w_edge = (alpha_edge * w_edge + beta_edge_eff * sum_w) / denom
                    agg_ms = (time.time() - agg_start) * 1000.0

                    # Criptografar e enviar ao Cloud
                    payload = w_edge.astype("float32").tobytes()
                    enc = encrypt(EDGE_KEY_HEX, payload)
                    await cw.send(json.dumps({
                        "type": "edge_model",
                        "edge": EDGE_ID,
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

                    ve_feedback = float(reply.get("ve", ve_feedback))
                    ve_feedback = max(0.1, min(2.0, ve_feedback))
                    beta_edge_eff = beta_edge_base * ve_feedback

                    # Ajustar janela deslizante localmente (somente no modo assíncrono)
                    if not SYNC_MODE:
                        window = max(3.0, float(window + alpha_sw * (hp["pdesired"] - pactual)))
                    last_agg_ts = now

                    # Log de métricas do edge
                    log_metric(
                        "edge",
                        edge=EDGE_ID,
                        window=window,
                        pactual=pactual,
                        qcurrent=qcurrent,
                        qedge=qedge,
                        ve=ve_feedback,
                        beta_edge=beta_edge_eff,
                        buf=len(A),
                        payload_bytes=payload_bytes,
                        dec_ms_mean=float(np.mean(dec_ms_vals)) if dec_ms_vals else None,
                        agg_ms=agg_ms,
                    )

            finally:
                try:
                    await cw.close()
                    print(f"[edge:{EDGE_ID}] conexão fechada com Cloud")
                except Exception:
                    pass

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
    await ws.recv()
    await ws.send(json.dumps({"type": "init", "whex": w_edge.tobytes().hex()}))
    print(f"[edge:{EDGE_ID}] IoT conectado")

    while True:
        msg = await ws.recv()
        M = json.loads(msg)
        if M["type"] == "update":
            iot = M["iot"]
            r = M["round"]

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

            train_acc = M.get("train_acc")
            train_loss = M.get("train_loss")
            if train_acc is None:
                train_acc = M.get("val_acc")
            if train_loss is None:
                train_loss = M.get("val_loss")
            clients.add(iot)
            item = (time.time(), iot, r, w, train_acc, train_loss, dec_ms, M.get("payload_bytes"))
            if SYNC_MODE:
                bucket = pending_by_round.setdefault(r, {})
                bucket[iot] = item
            else:
                updates.append(item)
            iot_train_acc[iot] = train_acc
            if train_acc is not None:
                print(f"[edge:{EDGE_ID}] update recebido de {iot}, round={r}, train_acc={float(train_acc):.3f}")
            else:
                print(f"[edge:{EDGE_ID}] update recebido de {iot}, round={r}")
            # Envia imediatamente o modelo do edge (pode estar "stale", pois é assíncrono)
            await ws.send(json.dumps({"type": "model", "whex": w_edge.tobytes().hex()}))

async def main():
    global model, wvec0, w_edge
    # Inicializa um modelo para obter formas e pesos iniciais
    tmp = build_model()
    compile_model(tmp, lr=hp["eta"])
    wvec0 = get_weights_vector(tmp)
    w_edge = wvec0.copy()

    log_event("edge", edge=EDGE_ID, event="start", port=EDGE_PORT)

    await asyncio.gather(
        ws_server("0.0.0.0", EDGE_PORT, iot_handler),
        cloud_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
