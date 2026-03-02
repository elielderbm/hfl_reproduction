import os, asyncio, json, time, numpy as np, random, re
from collections import defaultdict
from project.common.messaging import ws_server
from project.common.crypto import decrypt
from project.common.model import build_model, compile_model, get_weights_vector, set_weights_vector
from project.common.logging_utils import log_event, log_metric
from project.common.metrics import score_from_rmse, regression_metrics
from project.common.config import load_hparams
from project.common.data_utils import load_global_test, load_global_test_by_target
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from pathlib import Path

PORT = 9000
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED","42"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)

hp = load_hparams()
beta_cloud = float(os.getenv("BETA_CLOUD", str(hp["beta_cloud"])))

# State
edge_meta = defaultdict(lambda: {"p":0.0,"q":1.0})
w_global = None
round_ctr = 0
eval_model = None
X_test = None
y_test = None
X_test_by_target = {}
SAVE_WEIGHTS = os.getenv("SAVE_GLOBAL_WEIGHTS", "0") == "1"
SAVE_EVERY = int(os.getenv("SAVE_GLOBAL_WEIGHTS_EVERY", "1"))
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", "/workspace/outputs/weights"))

# Conexões ativas
peers = {}  # edge_id -> websocket

def edge_key_for(edge_id):
    if edge_id == "edge1":
        return os.getenv("EDGE1_KEY", os.getenv("EDGE_KEY_HEX","00112233445566778899AABBCCDDEEFF"))
    if edge_id == "edge2":
        return os.getenv("EDGE2_KEY", os.getenv("EDGE_KEY_HEX","0102030405060708090A0B0C0D0E0F10"))
    return os.getenv("EDGE_KEY_HEX","00112233445566778899AABBCCDDEEFF")

async def handler(ws, path):
    global w_global, round_ctr, peers, eval_model, X_test, y_test, X_test_by_target

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

    edge = hello["edge"]
    print(f"[cloud] novo edge conectado: {edge}")
    peers[edge] = ws

    # Envia init
    try:
        await ws.send(json.dumps({"type":"init","whex": w_global.tobytes().hex()}))
        print(f"[cloud] init enviado para {edge} (n={w_global.size})")
    except Exception as e:
        print(f"[cloud] erro enviando init para {edge}: {e}")
        peers.pop(edge, None)
        return

    if not hasattr(handler, "edge_last"):
        handler.edge_last = {}

    try:
        while True:
            try:
                raw = await ws.recv()
                if raw is None:
                    print(f"[cloud] conexão fechada (sem msg) por {edge}")
                    break
                msg = json.loads(raw)
            except ConnectionClosedOK:
                print(f"[cloud] conexão encerrada OK ({edge})")
                break
            except ConnectionClosedError as e:
                print(f"[cloud] conexão encerrada erro ({edge}): {e}")
                break
            except Exception as e:
                print(f"[cloud] erro ao receber de {edge}: {e}")
                continue

            if msg.get("type") != "edge_model":
                print(f"[cloud] ignorando msg de {edge} tipo={msg.get('type')}")
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

            if we.shape != w_global.shape:
                print(f"[cloud] SHAPE MISMATCH de {edge}: recebido={we.shape}, esperado={w_global.shape}. Ignorado.")
                continue

            edge_meta[edge]["p"] = float(msg.get("pactual", 0.0))
            edge_meta[edge]["q"] = float(msg.get("qedge", 1.0))
            handler.edge_last[edge] = we

            # Agregação
            num = np.zeros_like(w_global, dtype=np.float32)
            den = 0.0
            ve_map = {}
            for e_id, w in handler.edge_last.items():
                p = edge_meta[e_id]["p"]; q = edge_meta[e_id]["q"]
                v = beta_cloud * p + (1.0 - beta_cloud) * q
                ve_map[e_id] = float(v)
                num += v * w
                den += v
            if den > 0:
                w_global = num / den

            round_ctr += 1

            # Avaliação global no conjunto de teste
            global_loss = None
            global_mae = None
            global_rmse = None
            global_r2 = None
            global_mape = None
            global_score = None
            if X_test is not None and len(y_test) > 0:
                set_weights_vector(eval_model, w_global)
                try:
                    preds = eval_model.predict(X_test, verbose=0).reshape(-1)
                except Exception:
                    preds = None
                if preds is not None and np.isfinite(preds).all():
                    global_loss, global_mae, global_rmse, global_r2, global_mape = regression_metrics(y_test, preds)
                    global_score = score_from_rmse(global_rmse)

            # Avaliação por target (temp vs humidity, etc.)
            target_metrics = {}
            if X_test_by_target:
                for target, (Xt, yt) in X_test_by_target.items():
                    if yt is None or len(yt) == 0:
                        continue
                    try:
                        preds_t = eval_model.predict(Xt, verbose=0).reshape(-1)
                    except Exception:
                        continue
                    if not np.isfinite(preds_t).all():
                        continue
                    t_loss, t_mae, t_rmse, t_r2, t_mape = regression_metrics(yt, preds_t)
                    t_score = score_from_rmse(t_rmse)
                    key = re.sub(r"[^a-zA-Z0-9]+", "_", str(target).strip().lower()).strip("_")
                    target_metrics[f"global_{key}_rmse"] = t_rmse
                    target_metrics[f"global_{key}_mae"] = t_mae
                    target_metrics[f"global_{key}_r2"] = t_r2
                    target_metrics[f"global_{key}_mape"] = t_mape
                    target_metrics[f"global_{key}_score"] = t_score

            # Persistir pesos globais (opcional)
            if SAVE_WEIGHTS and round_ctr % max(1, SAVE_EVERY) == 0:
                try:
                    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
                    np.save(WEIGHTS_DIR / f"cloud_round_{round_ctr:04d}.npy", w_global.astype("float32"))
                except Exception as e:
                    print(f"[cloud] falha salvando pesos: {e}")

            # 🔹 Broadcast do modelo para todos os edges
            dead = []
            for e_id, peer_ws in list(peers.items()):
                try:
                    await peer_ws.send(json.dumps({
                        "type":"model",
                        "whex": w_global.tobytes().hex(),
                        "round": round_ctr,
                        "ve": ve_map.get(e_id, 1.0),
                    }))
                    print(f"[cloud] modelo global enviado -> {e_id}, round={round_ctr}")
                except Exception as e:
                    print(f"[cloud] erro enviando modelo para {e_id}: {e}")
                    dead.append(e_id)
            for e_id in dead:
                peers.pop(e_id, None)

            log_metric(
                "cloud",
                round=round_ctr,
                edges=len(handler.edge_last),
                beta_cloud=beta_cloud,
                global_loss=global_loss,
                global_mae=global_mae,
                global_rmse=global_rmse,
                global_r2=global_r2,
                global_mape=global_mape,
                global_score=global_score,
                dec_ms=dec_ms,
                payload_bytes=msg.get("payload_bytes"),
                **target_metrics,
            )

    finally:
        print(f"[cloud] handler encerrado para {edge}")
        peers.pop(edge, None)

async def main():
    global w_global, eval_model, X_test, y_test, X_test_by_target
    tmp = build_model()
    compile_model(tmp, lr=hp["eta"])
    w_global = get_weights_vector(tmp).copy()
    eval_model = build_model()
    compile_model(eval_model, lr=hp["eta"])
    X_test, y_test = load_global_test()
    X_test_by_target = load_global_test_by_target()
    log_event("cloud", event="start", port=PORT)
    print(f"[cloud] iniciado na porta {PORT}, pesos iniciais n={w_global.size}")
    await ws_server("0.0.0.0", PORT, handler)

if __name__ == "__main__":
    asyncio.run(main())
