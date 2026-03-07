# Projeto

Implementação das camadas **IoT** e **Cloud** com:
- **WebSockets** para comunicação
- **Salsa20** (criptografia leve) para updates de modelo
- **Agregação Assíncrona Centralizada** no Cloud com janela deslizante
- **Regressão temporal** (RMSE/MAE/R2/MAPE + score) e **classificação binária** (acurácia/precisão/recall/F1/BCE)
- **Modelos por camada** (student no IoT, teacher no Cloud) com suporte a TCN/GRU/MLP
- **IoT TinyML**: student MLP pequeno
- **Distilação teacher → student** (opcional, controlada por `DISTILL_ALPHA`)
- **Modelos globais por target** (temperatura, umidade e luminosidade separados)
- **Cloud fine-tuning** opcional para melhorar inferência
- **Métricas e Logs** em JSONL
- **Scripts de Análise**

Execute cada serviço via `docker-compose.yml` na raiz.
