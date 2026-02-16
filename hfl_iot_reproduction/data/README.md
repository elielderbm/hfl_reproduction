# Data Pipeline

- Faz *download* do HAR (UCI) e re-particiona por **subject_id**.
- Usa `config/clients.yml` para mapear IoT→Participante.
- Gera `data/clients/<iot>/{train,val,test}.csv`.

Executa automaticamente via serviço `data_prep` no `docker-compose`.
