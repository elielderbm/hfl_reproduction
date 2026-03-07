# Cloud (Agregação Centralizada + Feedback)

- Recebe updates dos IoTs (criptografados).
- Agrega de forma **assíncrona** com janela deslizante e histórico, produzindo `w_global` **por target**.
- Treina um **teacher TCN** (opcional) com dados proxy e envia seus pesos aos IoTs.
- Opcionalmente faz **fine-tuning do student** no Cloud com dados proxy (server-side).
- Retorna `w_global` aos IoTs + *teacher* (quando habilitado). Registra métricas (`global_*`, `teacher_*`).

Execução: pelo `docker-compose` (serviço `cloud`). Logs em `logs/cloud.jsonl`.
