# Data Pipeline

- Faz *download* do HAR (UCI) e re-particiona por **subject_id**.
- Usa `config/clients.yml` para mapear IoT→Participante (1 sujeito por IoT, conforme o artigo).
- Gera `data/clients/<iot>/{train,val,test}.csv`.
- Script opcional `heterogeneity_metrics.py` calcula Entropia, JS Divergence e EMD entre clientes.

Executa automaticamente via serviço `data_prep` no `docker-compose`.
