# Comparação: Artigo Base (HFL) vs Implementação Centralizada (IoT → Cloud)

Esta branch remove a camada **Edge** e a ponderação **p/q** do HFL, adotando **FL centralizado assíncrono** com agregação direta no Cloud. A tabela abaixo deixa explícito **o que permanece** e **o que mudou**.

| Elemento | Artigo Base (síntese) | Implementação atual | Implicação Metodológica |
| --- | --- | --- | --- |
| Arquitetura | Hierarquia IoT → Edge → Cloud | **IoT → Cloud (sem Edge)** | Deixa de reproduzir o núcleo hierárquico do HFL |
| Agregação no Edge | Assíncrona com janela deslizante | **Inexistente** | Camada removida |
| Agregação no Cloud | Ponderação por participação/qualidade (p, q) | **Agregação centralizada assíncrona** (`w = (α·w + β·Σw_i)/(α+β·|A|)`) | Remove a regra p/q do paper |
| Tarefa principal | Foco em tarefa única/homogênea | **Multitarefas heterogêneas** (regressão + classificação) | Amplia para cenários reais |
| Dados | Não fixa dataset específico | **Intel Lab Sensor Data** | Concretiza em cenário IoT real não‑IID |
| Não‑IID | Heterogeneidade assumida | Não‑IID por **mote** (sensor) | Distribuição realista e rastreável |
| Modelos por camada | Não enfatiza diferença arquitetural | **IoT student** leve + **Cloud teacher** robusto | Aderência a restrições de hardware no IoT |
| Modelo global | Único modelo global | **Modelo global por target** (temp/humidity/light) | Evita interferência negativa entre tarefas |
| Conhecimento global → local | Implícito via agregação | **Distilação teacher → student** (opcional) | Transferência de conhecimento sem expor dados |
| Segurança | Privacidade por agregação | **Criptografia Salsa20** nos updates | Confidencialidade no transporte |
| Comunicação | Não especifica stack | **WebSocket full‑duplex** | Execução próxima do cenário real |
| Métricas | Erro/qualidade globais | Métricas ampliadas (RMSE/MAE/R2/MAPE + BCE/Acc/F1) | Avaliação mais completa |

## Leitura crítica (o que muda em relação ao paper)
- **A reprodução do HFL não é mantida**: a camada Edge foi removida e a agregação p/q do Cloud foi substituída por **agregação centralizada assíncrona**.
- O resultado é um **baseline FL centralizado** com janela deslizante, útil para comparação, mas **não equivalente** ao método hierárquico do artigo.

## O que permanece como contribuição nesta branch
1. **Heterogeneidade de tarefas** (regressão + classificação).
2. **Modelos por capacidade** (student leve no IoT, teacher robusto no Cloud).
3. **Separação de modelos globais por target**.
4. **Distilação teacher → student**.
5. **Criptografia operacional** (Salsa20).

---
Se quiser, posso gerar uma versão acadêmica desta comparação com redação de dissertação.
