# HFL-IntelLab Docker Simulation (IoT → Edge → Cloud)

Reprodução **fiel** da metodologia do artigo *“Hierarchical Aggregation for Federated Learning in Heterogeneous IoT Scenarios: Enhancing Privacy and Communication Efficiency”* **+ extensões** (contribuições desta dissertação):
- Três camadas: **IoT** (dispositivos), **Edge** (pré-agregação assíncrona com janela deslizante), **Cloud** (agregação global ponderada por participação e qualidade).
- **Criptografia leve (Salsa20)** sobre *updates* do modelo.
- **Intel Lab Sensor Data** (MIT) com divisão por **mote** (não-IID por sensor).
- **Regressão temporal**: previsão de **temperatura/umidade em t+Δ** usando **janelas passadas**.
- **Classificação binária**: inferência de **luminosidade** (2 estados) em t+Δ.
- **Modelos por camada**: *student* leve no IoT/Edge (MLP pequena) e *teacher* robusto no Cloud; Edge pode treinar um *teacher* moderado local para melhorar a inferência sem quebrar a agregação.
- **Distilação teacher → student** (opcional) para melhorar inferência sem expor dados.
- **Modelos globais por target** (temperatura, umidade e luminosidade separados).
- **WebSocket** full-duplex entre camadas.
- Métricas e análises: *score* de qualidade, **RMSE/MAE/R2/MAPE** (regressão), **Acurácia/Precisão/Recall/F1/BCE** (classificação), distillation error, round time, throughput, participação e qualidade.
- **Barreira de prontidão** (opcional) para iniciar somente quando Edge/IoT estiverem conectados.

> **Importante:** este repositório automatiza **download + preparação** do dataset, **treino federado** por 80 rodadas (configurável), geração de **logs** em JSONL e scripts de **análise** e **plots**.

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

## Tasks (Regressão e Classificação, t+Δ)
Para cada IoT:
- Selecionamos um **mote_id** (sensor) via `config/clients.yml`.
- É possível usar `auto` e o mapeamento resolvido fica em `data/intel_lab/clients_resolved.yml`.
- Definimos o **alvo** via `config/dataset.yml`:
  - `iot1`, `iot2`: **temperatura** (regressão)
  - `iot3`, `iot4`: **umidade** (regressão)
  - `iot5`, `iot6`: **luminosidade** (classificação binária)
- Geramos janelas com `window_size` amostras passadas e **predizemos o valor no tempo t+Δ** (`delta_steps`).
- Para **luminosidade**, binarizamos o alvo usando o **limiar** definido em `classification.light.threshold` (default: mediana do treino).
- As features são concatenadas em vetor (ex.: `window_size * len(features)`).
- O **model input** é reformatado internamente para `[janela, variáveis]` para as TCNs.

**Score de qualidade (q):**
- **Regressão**: usamos RMSE e definimos `score = 1 / (1 + RMSE)`.
- **Classificação**: usamos **acurácia** como *score*.
- Esse *score* alimenta a estimativa `q_e` no Edge, mantendo a lógica do paper (qualidade ↑ → peso ↑).

## Serviços
- **data_prep**: faz download e gera janelas temporais do Intel Lab para cada IoT:
```
sudo docker compose up data_prep
```
- **cloud**: agrega modelos dos edges com pesos `v_e = β·p_e + (1-β)·q_e`, mantém **modelo global por target**, treina *teacher* (opcional) com proxy público e envia *feedback*.
- **edge1/edge2/edge3**: recebem atualizações de IoTs, agregam de forma **assíncrona** usando **janela deslizante** e suavização histórica, **replicando pesos por target**.
- **iot1..iot6**: treinam localmente (TensorFlow), enviam atualização criptografada (Salsa20) + métricas locais, e podem usar **distilação** com *teacher*.
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
    sudo docker compose run --rm analyzer python -m project.analysis.compare_runs --base outputs/experiments/<RUN_ID>/async --new outputs
    sudo docker compose run --rm analyzer python data/heterogeneity_metrics.py
   ```
7. **Limpar** (volumes/artefatos): `bash scripts/clean.sh`

> Dica: Logs em `./logs/*.jsonl`. Use `jq` para inspeção.

## Parâmetros Principais
- Rodadas: `SIM_ROUNDS` (default **80**)
- Épocas locais por rodada: `LOCAL_EPOCHS` (default **1**)
- LR/Batch: `LR`, `BATCH_SIZE`
- Otimizador/estabilidade: `OPTIMIZER` (sgd|adam), `CLIP_NORM`
- Janela deslizante: `SW_INIT`, `PDESIRED`, `ALPHA_SW`
- Pesos de agregação: `ALPHA_EDGE` (histórico), `BETA_EDGE` (updates), `BETA_CLOUD` (p vs q)
- Targets por edge: `EDGE1_TARGET`, `EDGE2_TARGET`, `EDGE3_TARGET` (ex.: temp/humidity/light)
- Distilação: `DISTILL_ALPHA`
- Teacher (cloud): `TEACHER_ENABLE`, `TEACHER_EPOCHS`, `TEACHER_BATCH`, `TEACHER_LR`, `TEACHER_MAX_SAMPLES`, `TEACHER_REFRESH_EVERY`
- Push do teacher (dados mínimos): `TEACHER_PUSH_INIT`, `TEACHER_PUSH_EVERY` (0 = não enviar pesos do teacher)
- Fine-tuning no Cloud (student): `SERVER_FT_EPOCHS`, `SERVER_FT_BATCH`, `SERVER_FT_EVERY`, `SERVER_FT_MAX_SAMPLES`, `SERVER_FT_ALPHA`
- Fine-tuning no Edge (student): `EDGE_FT_EPOCHS`, `EDGE_FT_BATCH`, `EDGE_FT_EVERY`, `EDGE_FT_MAX_SAMPLES`, `EDGE_FT_ALPHA`
- Modelos por camada: `STUDENT_MODEL_TYPE` (tcn|gru|mlp), `CLOUD_TEACHER_MODEL_TYPE` (tcn|gru|mlp)
- MLP por camada: `STUDENT_MLP_*` (IoT/Edge, TinyML)
- Edge teacher: `EDGE_TEACHER_ENABLE`, `EDGE_TEACHER_MODEL_TYPE`, `EDGE_TEACHER_*`, `EDGE_DISTILL_SOURCE`, `EDGE_GRU_*` (tamanho moderado)
- TCN por camada: `STUDENT_TCN_*`, `CLOUD_TCN_*` (tamanho do modelo)
- Agregação por amostras no Edge: `EDGE_WEIGHT_BY_SAMPLES`
- Barreira de prontidão: `CLOUD_WAIT_FOR_EDGES`, `CLOUD_EXPECTED_EDGES`, `EDGE_WAIT_FOR_IOTS`, `EDGE_EXPECTED_IOTS`, `EDGE_WAIT_FOR_CLOUD`, `IOT_WAIT_FOR_READY`
- WebSocket tamanho máximo de mensagem: `WS_MAX_SIZE` (ex.: `16mb`); aumente se usar teacher muito grande
- Dataset (regressão temporal): `config/dataset.yml`
  - `window_size`, `delta_steps`, `features`, `targets`
  - `scaling.with_target` habilita normalização do alvo por target
  - `tasks` define regressão vs classificação por alvo
  - `classification.<target>.threshold` define o limiar binário (default: mediana do treino)
- Mapa IoT→Mote: `config/clients.yml`
- Pesos globais (para análise por target): `SAVE_GLOBAL_WEIGHTS=1` e opcional `SAVE_GLOBAL_WEIGHTS_EVERY`
- Função de perda: `LOSS_FN` (mse|mae|huber), `HUBER_DELTA` e `LOSS_FN_CLASSIF` (binary_crossentropy)

> **Nota importante:** o *student* (modelo agregado) **precisa ter a mesma arquitetura** em IoT/Edge/Cloud para que os vetores de pesos sejam compatíveis. Modelos maiores no Edge/Cloud devem ser usados como *teacher* e/ou em *fine‑tuning*, sem quebrar a agregação.
>
> Para **tráfego mínimo**, mantenha `TEACHER_PUSH_INIT=0` e `TEACHER_PUSH_EVERY=0`: só os pesos do *student* são transmitidos.

## Reprodutibilidade
- Scripts de **pré-processamento** reproduzem o cenário **não-IID por mote** (Intel Lab).
- **Assíncrono verdadeiro**: IoTs têm *delays* aleatórios (normal) por rodada, emulando heterogeneidade.
- Agregações e métricas replicam o **método do artigo** (com variáveis/nomes análogos).

## Extensões (Contribuições)
- **Modelos temporais (TCN)** em vez de MLP puro, preservando a metodologia HFL.
- **Teacher → Student Distillation** para melhorar a inferência local sem expor dados.
- **Modelos globais por target** (temperatura e umidade separados) para evitar interferência negativa.
- **Métricas ampliadas** (R2/MAPE e distillation error) e análises por target.
- **Edge fine-tuning** opcional (student no edge com proxy data) para melhorar inferência local.

## Experimentos Async vs Sync (HierFAVG)
Para rodar comparação automática e salvar logs/outputs separados:
```
bash scripts/run_experiments.sh
```
Os resultados ficam em `logs/experiments/<RUN_ID>/` e `outputs/experiments/<RUN_ID>/`.

## Licenças
- Código sob MIT.
- Intel Lab Sensor Data: conforme a licença/termos da base.
