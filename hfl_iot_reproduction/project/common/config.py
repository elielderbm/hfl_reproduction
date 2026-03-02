import os, yaml

def env_f(name, default: float): return float(os.getenv(name, str(default)))
def env_i(name, default: int): return int(os.getenv(name, str(default)))

def load_hparams():
    with open("/workspace/config/hyperparams.yml","r") as f:
        hp = yaml.safe_load(f)
    # override from env
    hp["T"] = env_i("SIM_ROUNDS", hp["T"])
    hp["E"] = env_i("LOCAL_EPOCHS", hp["E"])
    hp["eta"] = env_f("LR", hp["eta"])
    hp["B"] = env_i("BATCH_SIZE", hp["B"])
    hp["alpha_edge"] = env_f("ALPHA_EDGE", hp["alpha_edge"])
    hp["beta_edge"] = env_f("BETA_EDGE", hp["beta_edge"])
    hp["sw_init"] = env_i("SW_INIT", hp["sw_init"])
    hp["pdesired"] = env_f("PDESIRED", hp["pdesired"])
    hp["alpha_sw"] = env_f("ALPHA_SW", hp["alpha_sw"])
    hp["beta_cloud"] = env_f("BETA_CLOUD", 0.5)
    hp["gamma_edge"] = env_f("GAMMA_EDGE", hp.get("gamma_edge", 0.9))
    hp["q_fixed"] = os.getenv("Q_FIXED", str(hp.get("q_fixed", "1.0")))
    hp["delay_mu"] = env_f("IOT_DELAY_MU", hp.get("delay_mu", 0.0))
    hp["delay_sigma"] = env_f("IOT_DELAY_SIGMA", hp.get("delay_sigma", 0.0))

    # compat keys (used in older code paths)
    hp["epochs"] = hp["E"]
    hp["bs"] = hp["B"]
    return hp

def client_subject(iot_id: str) -> int:
    import yaml
    with open("/workspace/config/clients.yml","r") as f:
        m = yaml.safe_load(f)
    return int(m[iot_id])
