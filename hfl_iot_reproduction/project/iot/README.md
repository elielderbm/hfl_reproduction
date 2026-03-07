# IoT (Dispositivo)

- Treina localmente (**TensorFlow**, `E`, `B`, `eta`).
- Envia *update* (vetor de pesos) + métricas (`train_score`, `train_rmse`, `train_mae`, `r2`, `mape`) ao **Cloud**.
- Recebe modelo agregado do Cloud a cada envio (assíncrono).
- Quando habilitado, usa **distilação** do *teacher* (Cloud) para melhorar o student (`DISTILL_ALPHA`).

Execução: feita pelo `docker-compose` (serviços `iot1..iot6`). Logs em `logs/iot*.jsonl`.
