# IoT (Dispositivo)

- Treina localmente (**TensorFlow**, `E`, `B`, `eta`).
- Envia *update* (vetor de pesos) + métricas (`train_score`, `train_rmse`, `train_mae`) ao **Edge**.
- Recebe modelo agregado do Edge a cada envio (assíncrono).

Execução: feita pelo `docker-compose` (serviços `iot1..iot4`). Logs em `logs/iot*.jsonl`.
