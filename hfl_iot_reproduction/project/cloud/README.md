# Cloud (Agregação Global + Feedback)

- Recebe modelos dos Edges (criptografados).
- Pondera por `v_e = β·p_e + (1-β)·q_e` e agrega para `w_global`.
- Retorna `w_global` aos Edges. Registra métricas.

Execução: pelo `docker-compose` (serviço `cloud`). Logs em `logs/cloud.jsonl`.
