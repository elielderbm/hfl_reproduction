# Sequência de Mensagens MQTT (IoT ↔ Edge ↔ Cloud)

Este diagrama descreve o **fluxo real de mensagens** implementado no sistema, com os **tópicos MQTT** e o papel de cada camada.

```mermaid
sequenceDiagram
    autonumber
    participant I as IoT
    participant B as MQTT Broker
    participant E as Edge
    participant C as Cloud

    Note over I,E: Topologia lógica: IoT → Edge → Cloud

    I->>B: publish `hfl/edge/<edge_id>/iot/up`
    B->>E: deliver (type=hello)
    E->>B: publish `hfl/edge/<edge_id>/iot/down/<iot_id>` (type=init)
    B->>I: deliver init (student + teacher opcional)

    alt EDGE_WAIT_FOR_* ativo
        E->>B: publish `.../down/<iot_id>` (type=start)
        B->>I: deliver start
    end

    loop rounds
        I->>B: publish `.../iot/up` (type=update, pesos criptografados)
        B->>E: deliver update
        E->>B: publish `.../iot/down/<iot_id>` (type=model, pesos do edge)
        B->>I: deliver model (edge → iot)

        E->>B: publish `hfl/cloud/up/<edge_id>` (type=edge_model)
        B->>C: deliver edge_model
        C->>B: publish `hfl/cloud/down/<edge_id>` (type=model)
        B->>E: deliver model (cloud → edge)
    end

    Note over E,C: Agregação no Edge (janela assíncrona) e no Cloud (p/q)
```

## Tópicos e tipos

Tópicos:
- `hfl/edge/<edge_id>/iot/up` → IoT → Edge (`hello`, `update`)
- `hfl/edge/<edge_id>/iot/down/<iot_id>` → Edge → IoT (`init`, `model`, `start`)
- `hfl/cloud/up/<edge_id>` → Edge → Cloud (`hello`, `edge_model`)
- `hfl/cloud/down/<edge_id>` → Cloud → Edge (`init`, `model`)

Tipos de mensagem:
- `hello`: handshake lógico com identificação e target
- `init`: pesos iniciais do *student* (+ teacher opcional)
- `start`: liberação após barreira de prontidão
- `update`: pesos locais criptografados + métricas
- `edge_model`: agregação do edge para o cloud
- `model`: retorno do modelo agregado da camada superior
