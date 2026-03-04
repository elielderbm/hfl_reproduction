# Cloud (Agregação Global + Feedback)

- Recebe modelos dos Edges (criptografados).
- Pondera por `v_e = β·p_e + (1-β)·q_e` e agrega para `w_global` **por target**.
- Treina um **teacher TCN** (opcional) com dados proxy e envia seus pesos aos Edges.
- Opcionalmente faz **fine-tuning do student** no Cloud com dados proxy (server-side).
- Retorna `w_global` aos Edges + *teacher* (quando habilitado). Registra métricas (`global_*`, `teacher_*`).

Execução: pelo `docker-compose` (serviço `cloud`). Logs em `logs/cloud.jsonl`.
