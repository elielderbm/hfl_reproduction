# HFL-IntelLab Docker Simulation (IoT → Edge → Cloud)

Reprodução **fiel** da metodologia do artigo *“Hierarchical Aggregation for Federated Learning in Heterogeneous IoT Scenarios: Enhancing Privacy and Communication Efficiency”*:
- Três camadas: **IoT** (dispositivos), **Edge** (pré-agregação assíncrona com janela deslizante), **Cloud** (agregação global ponderada por participação e qualidade).
- **Criptografia leve (Salsa20)** sobre *updates* do modelo.
- **Intel Lab Sensor Data** (MIT) com divisão por **mote** (não-IID por sensor).
- **Regressão temporal**: previsão de **temperatura/umidade em t+Δ** usando **janelas passadas**.
- **WebSocket** full-duplex entre camadas.
- Métricas e análises: *score* de qualidade, **RMSE/MAE**, round time, throughput, participação e qualidade.

> **Importante:** este repositório automatiza **download + preparação** do dataset, **treino federado** por 50 rodadas (configurável), geração de **logs** em JSONL e scripts de **análise** e **plots**.

## Visão Rápida
```
sudo docker compose up data_prep
sudo docker compose up --build
# aguarde a execução das rodadas (ou CTRL+C quando quiser parar)
sudo docker compose run --rm analyzer python -m project.analysis.extract_metrics
sudo docker compose run --rm analyzer python -m project.analysis.plot_curves
sudo docker compose run --rm analyzer python -m project.analysis.convergence_report --window 5 --show-rounds 5
sudo docker compose run --rm analyzer python -m project.analysis.target_report --every 1
```

## Estrutura
```
.
├── docker-compose.yml
├── .env                  # Ajuste se necessário
├── docker/               # Dockerfiles por serviço
├── data/                 # Scripts para obter e particionar Intel Lab
├── project/              # Código (IoT/Edge/Cloud) + análises
├── config/               # Mapas de clientes, dataset e hiperparâmetros
├── logs/                 # Logs JSONL por serviço/rodada
├── outputs/              # Saídas de análises (CSV/PNG)
└── scripts/              # Scripts utilitários
```

## Dataset: Intel Lab Sensor Data
Fonte e documentação:
- Download: `https://db.csail.mit.edu/labdata/data.txt.gz`
- Info: `https://db.csail.mit.edu/labdata/labdata.html`

Resumo (segundo a documentação):
- **54 sensores (Intel Mica2Dot)** em laboratório.
- **Coleta contínua** entre **28/02/2004 e 05/04/2004**.
- Cada linha inclui **timestamp/epoch**, **mote_id**, **temperatura**, **umidade**, **luminosidade** e **voltagem**.
- Volume aproximado de **~2,3 milhões de leituras**.

## Task de Regressão (t+Δ)
Para cada IoT:
- Selecionamos um **mote_id** (sensor) via `config/clients.yml`.
- É possível usar `auto` e o mapeamento resolvido fica em `data/intel_lab/clients_resolved.yml`.
- Definimos o **alvo** via `config/dataset.yml`:
  - `iot1`, `iot2`: **temperatura**
  - `iot3`, `iot4`: **umidade**
- Geramos janelas com `window_size` amostras passadas e **predizemos o valor no tempo t+Δ** (`delta_steps`).
- As features são concatenadas em vetor (ex.: `window_size * len(features)`).

**Score de qualidade (q):**
- Usamos **RMSE** como erro principal.
- Definimos um *score* ∈ (0,1] como `score = 1 / (1 + RMSE)`.
- Esse *score* alimenta a estimativa `q_e` no Edge, mantendo a lógica do paper (qualidade ↑ → peso ↑).

## Serviços
- **data_prep**: faz download e gera janelas temporais do Intel Lab para cada IoT:
```
sudo docker compose up data_prep
```
- **cloud**: agrega modelos dos edges com pesos `v_e = β·p_e + (1-β)·q_e`, guarda métricas globais e envia *feedback*.
- **edge1/edge2**: recebem atualizações de IoTs, agregam de forma **assíncrona** usando **janela deslizante** e suavização histórica.
- **iot1..iot4**: treinam localmente (TensorFlow), enviam atualização criptografada (Salsa20) + métricas locais.
- **analyzer**: consolida logs em CSV e gera gráficos.
- **baseline (opcional)**: `SYNC_MODE=1` nos edges habilita agregação síncrona (HierFAVG) para comparação.

## Como usar
1. **Dependências**: Docker + Docker Compose.
2. **Configurar env**: edite `.env` (ajuste se quiser).
3. **Configurar dataset**:
   - `config/dataset.yml` (window/Δ, features, targets)
   - `config/clients.yml` (motes por IoT)
   - (opcional) `SAVE_GLOBAL_WEIGHTS=1` no `.env` para habilitar análise por target
4. **Preparar dados**: `sudo docker compose up data_prep` (baixa e gera janelas).
5. **Subir**: `sudo docker compose up --build` (executa as rodadas).
6. **Analisar**:
   ```bash
    sudo docker compose run --rm analyzer python -m project.analysis.extract_metrics
    sudo docker compose run --rm analyzer python -m project.analysis.plot_curves
    sudo docker compose run --rm analyzer python -m project.analysis.summarize
    sudo docker compose run --rm analyzer python -m project.analysis.paper_metrics
    sudo docker compose run --rm analyzer python -m project.analysis.paper_report
    sudo docker compose run --rm analyzer python -m project.analysis.convergence_report --window 5 --show-rounds 5
    sudo docker compose run --rm analyzer python -m project.analysis.target_report --every 1
    sudo docker compose run --rm analyzer python -m project.analysis.compare_async_sync --async-dir outputs/experiments/<RUN_ID>/async --sync-dir outputs/experiments/<RUN_ID>/sync
    sudo docker compose run --rm analyzer python data/heterogeneity_metrics.py
   ```
7. **Limpar** (volumes/artefatos): `bash scripts/clean.sh`

> Dica: Logs em `./logs/*.jsonl`. Use `jq` para inspeção.

## Parâmetros Principais
- Rodadas: `SIM_ROUNDS` (default **50**)
- Épocas locais por rodada: `LOCAL_EPOCHS` (default **1**)
- LR/Batch: `LR`, `BATCH_SIZE`
- Otimizador/estabilidade: `OPTIMIZER` (sgd|adam), `CLIP_NORM`
- Janela deslizante: `SW_INIT`, `PDESIRED`, `ALPHA_SW`
- Pesos de agregação: `ALPHA_EDGE` (histórico), `BETA_EDGE` (updates), `BETA_CLOUD` (p vs q)
- Dataset (regressão temporal): `config/dataset.yml`
  - `window_size`, `delta_steps`, `features`, `targets`
- Mapa IoT→Mote: `config/clients.yml`
- Pesos globais (para análise por target): `SAVE_GLOBAL_WEIGHTS=1` e opcional `SAVE_GLOBAL_WEIGHTS_EVERY`

## Reprodutibilidade
- Scripts de **pré-processamento** reproduzem o cenário **não-IID por mote** (Intel Lab).
- **Assíncrono verdadeiro**: IoTs têm *delays* aleatórios (normal) por rodada, emulando heterogeneidade.
- Agregações e métricas replicam o **método do artigo** (com variáveis/nomes análogos).

## Experimentos Async vs Sync (HierFAVG)
Para rodar comparação automática e salvar logs/outputs separados:
```
bash scripts/run_experiments.sh
```
Os resultados ficam em `logs/experiments/<RUN_ID>/` e `outputs/experiments/<RUN_ID>/`.

## Licenças
- Código sob MIT.
- Intel Lab Sensor Data: conforme a licença/termos da base.
