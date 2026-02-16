# HFL-HAR Docker Simulation (IoT → Edge → Cloud)

Reprodução **fiel** da metodologia do artigo *“Hierarchical Aggregation for Federated Learning in Heterogeneous IoT Scenarios: Enhancing Privacy and Communication Efficiency”*:
- Três camadas: **IoT** (dispositivos), **Edge** (pré-agregação assíncrona com janela deslizante), **Cloud** (agregação global ponderada por participação e qualidade).
- **Criptografia leve (Salsa20)** sobre *updates* do modelo.
- **HAR dataset** (UCI) distribuído por participante (não-IID).
- **WebSocket** full-duplex entre camadas.
- Métricas e análises: acurácia/perda locais, acurácia global, round time, throughput, participação e qualidade.

> **Importante:** este repositório automatiza **download + preparação** do dataset, **treino federado** por 50 rodadas (configurável), geração de **logs** em JSONL e scripts de **análise** e **plots**.

## Visão Rápida
```
docker-compose up --build
# aguarde a execução das rodadas (ou CTRL+C quando quiser parar)
docker compose run --rm analyzer python -m project.analysis.extract_metrics
docker compose run --rm analyzer python -m project.analysis.plot_curves
```

## Estrutura
```
.
├── docker-compose.yml
├── .env.example            # Copie para .env e ajuste se necessário
├── docker/                 # Dockerfiles por serviço
├── data/                   # Scripts para obter e particionar HAR por cliente
├── project/                # Código (IoT/Edge/Cloud) + análises
├── config/                 # Mapas de clientes e hiperparâmetros
├── logs/                   # Logs JSONL por serviço/rodada
├── outputs/                # Saídas de análises (CSV/PNG)
└── scripts/                # Scripts utilitários
```

## Serviços
- **data_prep**: faz o download e a geração de dados heterogêneos do dataset HAR para os IoT devices: 
```
docker compose up data_prep      # Executar 2x
```
- **cloud**: agrega modelos dos edges com pesos `v_e = β·p_e + (1-β)·q_e`, guarda métricas globais e envia *feedback*.
- **edge1/edge2**: recebem atualizações de IoTs, agregam de forma **assíncrona** usando **janela deslizante** e suavização histórica.
- **iot1..iot4**: treinam localmente (TensorFlow), enviam atualização criptografada (Salsa20) + métricas locais.
- **analyzer**: consolida logs em CSV e gera gráficos.

## Como usar
1. **Dependências**: Docker + Docker Compose.
2. **Configurar env**: `cp .env.example .env` (ajuste se quiser).
3. **Subir**: `docker compose up --build` (prepara dataset automaticamente).
4. **Analisar**:
   ```bash
   docker compose run --rm analyzer python -m project.analysis.extract_metrics
   docker compose run --rm analyzer python -m project.analysis.plot_curves
   docker compose run --rm analyzer python -m project.analysis.summarize
   ```
5. **Limpar** (volumes/artefatos): `bash scripts/clean.sh`

> Dica: Logs em `./logs/*.jsonl`. Use `jq` para inspeção.

## Parâmetros Principais
- Rodadas: `SIM_ROUNDS` (default **50**)
- Épocas locais por rodada: `LOCAL_EPOCHS` (default **1**)
- LR/Batch: `LR`, `BATCH_SIZE`
- Janela deslizante: `SW_INIT`, `PDESIRED`, `ALPHA_SW`
- Pesos de agregação: `ALPHA_EDGE` (histórico), `BETA_EDGE` (updates), `BETA_CLOUD` (p vs q)
- Mapa IoT→Participante: `config/clients.yml`

## Reprodutibilidade
- Scripts de **pré-processamento** reproduzem o cenário **não-IID por participante** (HAR).
- **Assíncrono verdadeiro**: IoTs têm *delays* aleatórios (normal) por rodada, emulando heterogeneidade.
- Agregações e métricas replicam o **método do artigo** (com variáveis/nomes análogos).

## Licenças
- Código sob MIT. HAR dataset (UCI) conforme a licença deles.
