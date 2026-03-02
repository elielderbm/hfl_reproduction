# Edge (Agregação Assíncrona)

- Recebe atualizações dos IoTs conectados (WebSocket :8765).
- **Janela deslizante adaptativa**: `window = window + alpha_sw * (pdesired - pactual)`.
- Agregação: `w_e = (alpha_edge * w_e + beta_edge * sum(w_i)) / (alpha_edge + beta_edge*|A|)`.
- Envia modelo agregado (criptografado Salsa20) para a **Cloud** e recebe global + *feedback*.
- Baseline síncrono (HierFAVG): defina `SYNC_MODE=1` e `EDGE_CLIENTS=iot1,iot2` (ou `iot3,iot4`) para o edge esperar todos os clientes por round.

Execução: pelo `docker-compose` (serviços `edge1`, `edge2`). Logs em `logs/edge*.jsonl`.
