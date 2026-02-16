import os, asyncio, json, time, numpy as np, random
from collections import deque, defaultdict

from project.common.messaging import ws_server, ws_connect, b64d
from project.common.crypto import encrypt, decrypt
from project.common.model import build_model, compile_model, get_weights_vector, set_weights_vector
from project.common.logging_utils import log_event, log_metric
from project.common.config import load_hparams

EDGE_ID = os.getenv("EDGE_ID","edge1")
EDGE_KEY_HEX = os.getenv("EDGE_KEY_HEX","00112233445566778899AABBCCDDEEFF")
EDGE_PORT = 8765
CLOUD_URI = "ws://cloud:9000"
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED","42"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)

hp = load_hparams()

# State
clients = set()
model = None
wvec0 = None
w_edge = None
pactual = 0.0
window = int(hp["sw_init"])
alpha_sw = float(hp["alpha_sw"])
alpha_edge = float(hp["alpha_edge"])
beta_edge  = float(hp["beta_edge"])

# Buffers
updates = deque()  # (iot, round, w, val_acc, val_loss)
iot_round_seen = defaultdict(int)  # counts per round
iot_val_acc = {}

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
    global w_edge, pactual, window

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
                while True:
                    await asyncio.sleep(1.0)

                    if len(updates) == 0:
                        continue

                    # pactual = fração de IoTs distintos que contribuíram na janela recente
                    contrib = {u[0] for u in list(updates)[-window:]}
                    if len(clients) > 0:
                        pactual = len(contrib) / max(1, len(clients))

                    # Agregar atualizações do buffer
                    A = list(updates)
                    updates.clear()
                    if len(A) == 0:
                        continue

                    sum_w = np.zeros_like(w_edge, dtype=np.float32)
                    for (_iot, _r, _w, _acc, _loss) in A:
                        sum_w += _w

                    # w_edge = (alpha * w_edge + beta * sum(w_i)) / (alpha + beta*|A|)
                    w_edge = (alpha_edge * w_edge + beta_edge * sum_w) / (alpha_edge + beta_edge * len(A))

                    # Estimativa de qualidade (média das últimas val_acc reportadas)
                    qcurrent = float(np.mean(list(iot_val_acc.values()))) if iot_val_acc else 1.0

                    # Criptografar e enviar ao Cloud
                    payload = w_edge.astype("float32").tobytes()
                    enc = encrypt(EDGE_KEY_HEX, payload)
                    await cw.send(json.dumps({
                        "type": "edge_model",
                        "edge": EDGE_ID,
                        "pactual": pactual,
                        "qcurrent": qcurrent,
                        "bytes": enc.hex()
                    }))

                    # Receber modelo global atualizado
                    reply = json.loads(await cw.recv())
                    assert reply["type"] == "model", f"Esperado 'model', recebido {reply.get('type')}"
                    wg = np.frombuffer(bytes.fromhex(reply["whex"]), dtype=np.float32)
                    w_edge = wg  # Edge segue o global

                    # Ajustar janela deslizante localmente
                    window = max(3, int(window + alpha_sw * (hp["pdesired"] - pactual)))

                    # Log de métricas do edge
                    log_metric("edge", edge=EDGE_ID, window=window, pactual=pactual, qcurrent=qcurrent, buf=len(A))

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
                dec = decrypt(key_hex, enc)
                w = np.frombuffer(dec, dtype=np.float32)
            except Exception as e:
                print(f"[edge:{EDGE_ID}] erro decriptando update de {iot}: {e} (descartado)")
                continue

            # Checagem de shape (evita quebrar agregação)
            if w_edge is not None and w.shape != w_edge.shape:
                print(f"[edge:{EDGE_ID}] update inválido de {iot}: shape={w.shape} esperado={w_edge.shape} (descartado)")
                continue

            clients.add(iot)
            updates.append((iot, r, w, M["val_acc"], M["val_loss"]))
            iot_val_acc[iot] = M["val_acc"]
            print(f"[edge:{EDGE_ID}] update recebido de {iot}, round={r}, val_acc={M['val_acc']:.3f}")

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
