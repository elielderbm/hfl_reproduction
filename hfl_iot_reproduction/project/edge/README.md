# Edge (Agregação Assíncrona)

- Recebe atualizações dos IoTs conectados (WebSocket :8765).
- **Janela deslizante adaptativa**: `window = window + alpha_sw * (pdesired - pactual)`.
- Agregação: `w_e = (alpha_edge * w_e + beta_edge * sum(w_i)) / (alpha_edge + beta_edge*|A|)`.
- Envia modelo agregado (criptografado Salsa20) para a **Cloud** e recebe global + *feedback*.

Execução: pelo `docker-compose` (serviços `edge1`, `edge2`). Logs em `logs/edge*.jsonl`.
