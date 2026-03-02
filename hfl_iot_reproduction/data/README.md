# Data Pipeline (Intel Lab Sensor Data)

- Faz **download** do Intel Lab Sensor Data (`data.txt.gz`) e extrai para `data/intel_lab/data.txt`.
- Limpa leituras inválidas (faixas plausíveis), ordena por **tempo (epoch)** e gera **janelas temporais**.
- Usa `config/clients.yml` para mapear **IoT → mote_id** (sensores). É possível usar `auto` e o mapa resolvido é salvo em `data/intel_lab/clients_resolved.yml`.
- Usa `config/dataset.yml` para definir **features**, **window_size**, **delta_steps** e **target** por IoT.
- Gera `data/clients/<iot>/{train,val,test}.csv` com colunas `x0..xN-1` e `y`.
- Faz **escalonamento global** das features (e opcionalmente do alvo), salvo em `data/intel_lab/scaler.json`.
- Salva meta em `data/intel_lab/meta.json` (dimensões, mapping, contagens).

Executa automaticamente via serviço `data_prep` no `docker-compose`.
