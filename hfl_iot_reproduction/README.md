# FL-IntelLab Docker Simulation (IoT → Cloud)

Simulação de **FL centralizado assíncrono** (IoT → Cloud) com **extensões** aplicadas ao cenário IoT:
- Duas camadas: **IoT** (dispositivos) e **Cloud** (agregação assíncrona com janela deslizante e histórico).
- **Criptografia leve (Salsa20)** sobre *updates* do modelo.
- **Intel Lab Sensor Data** (MIT) com divisão por **mote** (não-IID por sensor).
- **Regressão temporal**: previsão de **temperatura/umidade em t+Δ** usando **janelas passadas**.
- **Classificação binária**: inferência de **luminosidade** (2 estados) em t+Δ.
- **Modelos por camada**: *student* leve no IoT e *teacher* robusto no Cloud.
- **Distilação teacher → student** (opcional) para melhorar inferência sem expor dados.
- **Modelos globais por target** (temperatura, umidade e luminosidade separados).
- **WebSocket** full-duplex entre IoT e Cloud.
- Métricas e análises: *score* de qualidade, **RMSE/MAE/R2/MAPE** (regressão), **Acurácia/Precisão/Recall/F1/BCE** (classificação), distillation error, round time, throughput, participação e qualidade.

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
├── project/              # Código (IoT/Cloud) + análises
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
- O *score* é registrado e usado nas análises (não há ponderação p/q na agregação centralizada).

## Serviços
- **data_prep**: faz download e gera janelas temporais do Intel Lab para cada IoT:
```
sudo docker compose up data_prep
```
- **cloud**: agrega **diretamente** os updates dos IoTs de forma **assíncrona** com janela deslizante e histórico; mantém **modelo global por target**, treina *teacher* (opcional) e envia *feedback*.
- **iot1..iot6**: treinam localmente (TensorFlow), enviam atualização criptografada (Salsa20) + métricas locais, e podem usar **distilação** com *teacher*.
- **analyzer**: consolida logs em CSV e gera gráficos.

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
    sudo docker compose run --rm analyzer python -m project.analysis.compare_async_sync --async-dir outputs/experiments/<RUN_ID>/run_a --sync-dir outputs/experiments/<RUN_ID>/run_b
    sudo docker compose run --rm analyzer python -m project.analysis.compare_runs --base outputs/experiments/<RUN_ID>/run_a --new outputs
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
- Pesos de agregação (Cloud): `ALPHA_EDGE` (histórico) e `BETA_EDGE` (updates)
- Distilação: `DISTILL_ALPHA`
- Teacher (cloud): `TEACHER_ENABLE`, `TEACHER_EPOCHS`, `TEACHER_BATCH`, `TEACHER_LR`, `TEACHER_MAX_SAMPLES`, `TEACHER_REFRESH_EVERY`
- Push do teacher (dados mínimos): `TEACHER_PUSH_INIT`, `TEACHER_PUSH_EVERY` (0 = não enviar pesos do teacher)
- Fine-tuning no Cloud (student): `SERVER_FT_EPOCHS`, `SERVER_FT_BATCH`, `SERVER_FT_EVERY`, `SERVER_FT_MAX_SAMPLES`, `SERVER_FT_ALPHA`
- Modelos por camada: `STUDENT_MODEL_TYPE` (tcn|gru|mlp), `CLOUD_TEACHER_MODEL_TYPE` (tcn|gru|mlp)
- MLP por camada: `STUDENT_MLP_*` (IoT, TinyML)
- TCN por camada: `STUDENT_TCN_*`, `CLOUD_TCN_*` (tamanho do modelo)
- Agregação por amostras no Cloud: `EDGE_WEIGHT_BY_SAMPLES`
- WebSocket tamanho máximo de mensagem: `WS_MAX_SIZE` (ex.: `16mb`); aumente se usar teacher muito grande
- WebSocket keepalive: `WS_PING_INTERVAL` e `WS_PING_TIMEOUT` (em segundos). Use `0` para desativar pings e evitar `keepalive ping timeout` em treinos longos.
- Dataset (regressão temporal): `config/dataset.yml`
  - `window_size`, `delta_steps`, `features`, `targets`
  - `scaling.with_target` habilita normalização do alvo por target
  - `tasks` define regressão vs classificação por alvo
  - `classification.<target>.threshold` define o limiar binário (default: mediana do treino)
- Mapa IoT→Mote: `config/clients.yml`
- Pesos globais (para análise por target): `SAVE_GLOBAL_WEIGHTS=1` e opcional `SAVE_GLOBAL_WEIGHTS_EVERY`
- Função de perda: `LOSS_FN` (mse|mae|huber), `HUBER_DELTA` e `LOSS_FN_CLASSIF` (binary_crossentropy)

> **Nota:** na arquitetura centralizada (IoT → Cloud), variáveis `EDGE_*` e `SYNC_MODE` são legadas e **não** são usadas, exceto `ALPHA_EDGE`, `BETA_EDGE` e `EDGE_WEIGHT_BY_SAMPLES`, que agora controlam a agregação do Cloud. `BETA_CLOUD` não participa da agregação centralizada.

> **Nota importante:** o *student* (modelo agregado) **precisa ter a mesma arquitetura** em IoT/Cloud para que os vetores de pesos sejam compatíveis. Modelos maiores no Cloud devem ser usados como *teacher* e/ou em *fine‑tuning*, sem quebrar a agregação.
>
> Para **tráfego mínimo**, mantenha `TEACHER_PUSH_INIT=0` e `TEACHER_PUSH_EVERY=0`: só os pesos do *student* são transmitidos.

## Reprodutibilidade
- Scripts de **pré-processamento** reproduzem o cenário **não-IID por mote** (Intel Lab).
- **Assíncrono verdadeiro**: IoTs têm *delays* aleatórios (normal) por rodada, emulando heterogeneidade.
- Agregação e métricas seguem o **FL centralizado assíncrono** com janela deslizante.

## Extensões (Contribuições)
- **Modelos temporais (TCN)** em vez de MLP puro, preservando a metodologia FL.
- **Teacher → Student Distillation** para melhorar a inferência local sem expor dados.
- **Modelos globais por target** (temperatura e umidade separados) para evitar interferência negativa.
- **Métricas ampliadas** (R2/MAPE e distillation error) e análises por target.

## Experimentos (Comparação entre Execuções)
Para rodar duas execuções e comparar métrricas:
```
bash scripts/run_experiments.sh
```
Os resultados ficam em `logs/experiments/<RUN_ID>/` e `outputs/experiments/<RUN_ID>/`.

## Licenças
- Código sob MIT.
- Intel Lab Sensor Data: conforme a licença/termos da base.
