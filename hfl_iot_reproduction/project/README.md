# Projeto

Implementação das camadas **IoT**, **Edge** e **Cloud** com:
- **MQTT (pub/sub)** para comunicação entre camadas
- **Salsa20** (criptografia leve) para updates de modelo
- **Agregação Assíncrona** no Edge com janela deslizante
- **Agregação Global Ponderada** no Cloud
- **Regressão temporal** (RMSE/MAE/R2/MAPE + score) e **classificação binária** (acurácia/precisão/recall/F1/BCE)
- **Modelos por camada** (student no IoT/Edge, teacher no Cloud) com suporte a TCN/GRU/MLP
- **IoT TinyML**: student MLP pequeno; **Edge** pode treinar teacher moderado local
- **Distilação teacher → student** (opcional, controlada por `DISTILL_ALPHA`)
- **Modelos globais por target** (temperatura, umidade e luminosidade separados)
- **Edge/Cloud fine-tuning** opcionais para melhorar inferência
- **Métricas e Logs** em JSONL
- **Scripts de Análise**

Execute cada serviço via `docker-compose.yml` na raiz.
