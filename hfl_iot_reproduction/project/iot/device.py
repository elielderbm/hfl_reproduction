import os, asyncio, json, time, numpy as np, random
from project.common.messaging import ws_connect, b64e
from project.common.crypto import encrypt, decrypt
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

    ws = await ws_connect(EDGE_URI)
    try:
        # hello
        await ws.send(json.dumps({"type": "hello", "iot": IOT_ID}))
        msg = json.loads(await ws.recv())
        assert msg["type"] == "init"
        wg = np.frombuffer(bytes.fromhex(msg["whex"]), dtype=np.float32)
        set_weights_vector(local_model, wg)
        w_local = wg.copy()

        while True:
            round_ctr += 1
            # treino local
            local_model.fit(Xtr, ytr, batch_size=hp["bs"], epochs=hp["epochs"], verbose=0)
            w_new = get_weights_vector(local_model)
            # métricas de validação
            val_acc = float(local_model.evaluate(Xv, yv, verbose=0)[1])
            val_loss = float(local_model.evaluate(Xv, yv, verbose=0)[0])

            payload = w_new.astype("float32").tobytes()
            enc = encrypt(IOT_KEY_HEX, payload)
            await ws.send(json.dumps({
                "type": "update",
                "iot": IOT_ID,
                "round": round_ctr,
                "bytes": b64e(enc),
                "val_acc": val_acc,
                "val_loss": val_loss
            }))

            # recebe modelo atualizado do edge
            reply = json.loads(await ws.recv())
            assert reply["type"] == "model"
            w_edge = np.frombuffer(bytes.fromhex(reply["whex"]), dtype=np.float32)
            set_weights_vector(local_model, w_edge)
            w_local = w_edge.copy()

            log_metric("iot", iot=IOT_ID, round=round_ctr, val_acc=val_acc, val_loss=val_loss)
            await asyncio.sleep(hp["round_delay"])
    finally:
        await ws.close()

def main():
    asyncio.run(run_iot())

if __name__ == "__main__":
    main()
