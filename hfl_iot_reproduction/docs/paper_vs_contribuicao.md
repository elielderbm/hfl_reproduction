# Comparação: Artigo Base vs Implementação (Contribuições)

A tabela abaixo sintetiza **o que permanece fiel ao artigo** e **o que foi estendido** na sua implementação. A ideia é deixar explícito, de forma acadêmica, **onde termina a reprodução** e **onde começa a contribuição**.

| Elemento | Artigo Base (síntese) | Implementação/Contribuição | Implicação Metodológica |
| --- | --- | --- | --- |
| Arquitetura hierárquica | Hierarquia IoT → Edge → Cloud | Mantida integralmente | Preserva o núcleo do HFL e garante comparabilidade conceitual |
| Agregação no Edge | Assíncrona com janela deslizante | Mantida (com janela adaptativa) | Continua modelando heterogeneidade temporal |
| Agregação no Cloud | Ponderação por participação/qualidade (p, q) | Mantida: `v_e = β·p + (1-β)·q` | Mantém a mesma lógica de peso global do artigo |
| Tarefa principal | Foco em tarefa única/homogênea | **Multitarefas heterogêneas** (regressão + classificação) | Amplia o método para cenários reais com tarefas distintas |
| Dados | Não fixa um dataset específico | Instanciado com **Intel Lab Sensor Data** | Concretiza a metodologia em cenário IoT real não‑IID |
| Não‑IID | Heterogeneidade assumida | Não‑IID por **mote** (sensor) | Controla distribuição de forma realista e rastreável |
| Modelos por camada | Não enfatiza diferença arquitetural | **Modelos distintos por camada** (IoT leve, Edge moderado, Cloud robusto) | Aderência a restrições reais de hardware |
| Modelo global | Único modelo global | **Modelo global por target** (temp, humidity, light) | Evita interferência negativa entre tarefas |
| Conhecimento global → local | Implícito via agregação | **Distilação teacher → student** (opcional) | Transferência de conhecimento sem expor dados |
| Teacher/Student | Não explicitado | Teacher robusto no Cloud e teacher moderado no Edge | Melhora inferência sem quebrar compatibilidade de pesos |
| Fine‑tuning | Não descrito | **Fine‑tuning no Edge/Cloud** em dados proxy | Ajusta o global sem alterar regra de agregação |
| Segurança | Privacidade por agregação | **Criptografia Salsa20** nos updates | Medida prática de confidencialidade no transporte |
| Comunicação | Não especifica stack | **MQTT (pub/sub) com broker** | Aproxima da execução em ambiente real e reduz acoplamento |
| Métricas | Erro/qualidade globais | Métricas ampliadas (RMSE/MAE/R2/MAPE + BCE/Acc/F1) | Avaliação mais completa, multi‑tarefa |
| Avaliação por target | Não requerido | **Relatórios por target** (temp, humidity, light) | Permite análise isolada de cada tarefa |
| Contribuição científica | Método HFL | Extensão para **heterogeneidade de tarefas + modelos** | Expansão da aplicabilidade do método |

## Leitura crítica (onde termina a reprodução)
- **Tudo até a regra de agregação** (janela no Edge, ponderação `p/q` no Cloud) está **alinhado ao artigo**.
- A partir de **multitarefas, modelos por camada, distilação e fine‑tuning**, você entra em **contribuição autoral**.

## Leitura crítica (o que você adicionou)
1. **Heterogeneidade de tarefas**: regressão e classificação coexistindo na mesma arquitetura hierárquica.
2. **Modelos por capacidade**: IoT mínimo (TinyML), Edge moderado, Cloud robusto.
3. **Separação de modelos globais por target**: evita interferência negativa entre tarefas diferentes.
4. **Distilação + fine‑tuning**: melhora inferência local sem alterar o mecanismo de agregação base.
5. **Criptografia operacional**: reforça privacidade além da agregação em si.

## Fronteira metodológica (até onde você vai)
Você **não altera a matemática central** da agregação descrita no paper; em vez disso, **amplia a aplicabilidade** do método para cenários realistas, heterogêneos e com restrições computacionais diferenciadas.

---
Se quiser, posso transformar esta comparação em seção de capítulo de dissertação (com citações formais e redação acadêmica padronizada).
