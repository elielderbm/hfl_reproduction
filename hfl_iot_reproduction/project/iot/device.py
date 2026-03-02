import os, asyncio, json, time, numpy as np, random
from project.common.messaging import ws_connect, b64e
from project.common.crypto import encrypt
from project.common.model import build_model, compile_model, get_weights_vector, set_weights_vector
from project.common.logging_utils import log_event, log_metric
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
                local_model.fit(Xtr, ytr, batch_size=hp["B"], epochs=hp["E"], verbose=0)
                train_time_ms = (time.time() - t0) * 1000.0
                w_new = get_weights_vector(local_model)

                # métricas de treino (paper)
                train_loss, train_acc = local_model.evaluate(Xtr, ytr, verbose=0)
                train_acc = float(train_acc)
                train_loss = float(train_loss)

                # opcional: validação se existir
                val_acc = None
                val_loss = None
                if len(yv) > 0:
                    val_loss, val_acc = local_model.evaluate(Xv, yv, verbose=0)
                    val_acc = float(val_acc)
                    val_loss = float(val_loss)

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
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "payload_bytes": len(enc)
                }))

                # recebe modelo atualizado do edge
                reply = json.loads(await ws.recv())
                assert reply["type"] == "model"
                w_edge = np.frombuffer(bytes.fromhex(reply["whex"]), dtype=np.float32)
                set_weights_vector(local_model, w_edge)
                w_local = w_edge.copy()

                round_time_ms = (time.time() - round_start) * 1000.0
                log_metric(
                    "iot",
                    iot=IOT_ID,
                    round=round_ctr,
                    train_acc=train_acc,
                    train_loss=train_loss,
                    val_acc=val_acc,
                    val_loss=val_loss,
                    train_time_ms=train_time_ms,
                    enc_ms=enc_ms,
                    delay_ms=delay_s * 1000.0,
                    round_time_ms=round_time_ms,
                    payload_bytes=len(enc),
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
