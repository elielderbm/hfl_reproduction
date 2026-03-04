# Edge (Agregação Assíncrona)

- Recebe atualizações dos IoTs conectados (WebSocket :8765).
- **Janela deslizante adaptativa**: `window = window + alpha_sw * (pdesired - pactual)`.
- Agregação: `w_e = (alpha_edge * w_e + beta_edge * sum(w_i)) / (alpha_edge + beta_edge*|A|)`.
- Estima qualidade `q_current` a partir do **score** reportado pelos IoTs.
- Envia modelo agregado (criptografado Salsa20) para a **Cloud** e recebe global + *feedback*.
- Mantém **um modelo por target** (definido por `EDGE_TARGET`) e encaminha pesos do *teacher* aos IoTs.
- Pode ponderar agregação pelo número de amostras locais (`EDGE_WEIGHT_BY_SAMPLES=1`).
- Baseline síncrono (HierFAVG): defina `SYNC_MODE=1` e `EDGE_CLIENTS=iot1,iot2` (ou `iot3,iot4`) para o edge esperar todos os clientes por round.

Execução: pelo `docker-compose` (serviços `edge1`, `edge2`). Logs em `logs/edge*.jsonl`.
