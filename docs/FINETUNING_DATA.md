# Formato dos dados para fine-tuning

Transforma os exemplos de `data/finetune/qa_train.jsonl` em conversas de chat com o **mesmo
contexto que o sistema terá na inferência**:
- trechos recuperados pelo RAG (Etapa 3);
- seções adicionadas por regras determinísticas;
- dados do paciente lidos do SQLite (Etapa 2).

O objetivo é treinar o modelo a **ler** o contexto e citar o que está nele, e não a decorar
valores de pacientes.

Esta etapa não treina nenhum modelo: só prepara e verifica o formato. O treino, o split
treino/validação e a comparação estão em [FINETUNING.md](FINETUNING.md).

> **Prompt v3 (Etapa 4).** O JSON do paciente passou a incluir `historico_exames`, porque 10
> respostas de referência citavam exames anteriores que não estavam na entrada. O texto do
> sistema e a política de seções (v1) não mudaram. Os pré-registros do piloto foram gravados
> com `prompt_version` "2", mas a seleção de seções continua idêntica (0 divergências). O
> `--strict` agora também falha se a resposta citar exame, consulta ou paciente ausente da
> entrada (`ids_citados_fora_do_contexto` nos metadados).

## Gerar

```powershell
python -m pip install -e ".[dev,rag]" -c constraints-rag.txt
python scripts/build_database.py          # banco em data/processed/ (não versionado)
python scripts/build_sft_dataset.py       # -> data/processed/sft/train.jsonl (não versionado)
python scripts/build_sft_dataset.py --strict   # falha se alguma resposta citar seção fora do contexto
```

Cada linha do arquivo gerado tem duas partes:
- `messages`: o que o modelo vê e o que deve aprender a produzir;
- `metadata`: dados de auditoria, que **não** são enviados ao modelo.

```json
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
 "metadata": {"qa_id": "QA-0014", "categoria": "recusa", "prompt_version": "3", "policy_version": "1", "k": 4,
              "data_referencia": "2026-06-30", "patient_id": "PAC-0003",
              "status_recuperacao": "incerto",
              "refs_recuperadas": ["PROT-DM2-003#S3", "..."],
              "refs_por_regra": {"PROT-DM2-001#S4": ["regras_das_verificacoes", "..."], "...": []},
              "regras_em_recuperados": {"PROT-DM2-001#S3": ["regras_das_verificacoes", "..."]},
              "cobertura": "completa", "fundamentacao": "regras",
              "refs_citadas_fora_do_contexto": []}}
```

## Camadas (`src/medprotocol/finetuning/`)

| Módulo | Responsabilidade | Usado na inferência? |
|---|---|---|
| `sources.py` | Lê **só** `qa_train.jsonl`. Separa entrada (pergunta e paciente citado), alvo (resposta) e rótulos. Rejeita pacientes e seções reservados | Não |
| `context_policy.py` | Política determinística: quais seções acrescentar, a partir de pergunta, resumo do paciente e trechos recuperados | **Sim** |
| `context.py` | Monta o `AssistantContext`: pergunta, data de referência, `RetrievalResult`, resumo do paciente e seções por regra, sem duplicatas | **Sim** |
| `messages.py` | Converte o contexto em mensagens `system`/`user` (e `assistant` no treino). Prompt de sistema versionado (`PROMPT_VERSION = "2"`) | **Sim** |
| `sft.py` | Junta as camadas, verifica isolamento e calcula a cobertura (auditoria) | Não |

Nenhuma regra de negócio é reimplementada:
- prazos, pendências e limites vêm de `medprotocol.data.schedule`, via `repository.get_patient_summary`;
- os trechos e o status vêm de `ProtocolRetriever.retrieve`;
- a política lê em `schedule` de **quais seções** vêm as regras de prazo e de limite (`STATUS_SECTION`, `INTERVAL_SECTIONS`, `LIMIT_SECTIONS`).

Os limiares do RAG não foram alterados.

## Política de seções adicionais (`context_policy.py`)

`select_rule_sections(pergunta, paciente, hits)` é uma função pura de informações que existem na
inferência. **Não recebe** a resposta esperada, as referências citadas nem os rótulos. Um teste
fixa essa assinatura.

As regras são aplicadas nesta ordem. Cada seção registra a regra (ou as regras) que a incluíram:

| Regra (motivo) | Quando | Seções |
|---|---|---|
| `regras_das_verificacoes` | Há dados de paciente no contexto | Origem das regras usadas em `verificacoes`: `001#S4` (situação), `001#S3` e `003#S3` (prazos e limites) |
| `protocolo_vinculado:<ID>` | Uma consulta do paciente registra o protocolo | Seções operacionais do protocolo: 001 → S3, S4; 002 → S2, S3, S4, S5; 003 → S3, S4, S5 |
| `revisao:fora_do_limite:<EXAME>` | Exame em `fora_do_limite` | `REV` do protocolo que define o limite desse exame |
| `revisao:pendencias` / `revisao:inconsistencias` | Há pendências ou inconsistências | `REV` do protocolo que define a situação dos exames (`001#REV`) |
| `intencao:pedido_de_conduta` | A pergunta pede medicamento, dose, ajuste, tratamento, diagnóstico ou estadiamento | `REV` e `LIM` dos protocolos em foco |
| `intencao:estimativa_de_valor` | A pergunta pede para estimar, inventar ou preencher valores | `001#S4` (valores nunca são estimados) |
| `intencao:uso_fora_da_simulacao` | A pergunta fala de paciente real ou diretriz oficial | `LIM` dos protocolos em foco |

- **Protocolos em foco** são os citados na pergunta mais os vinculados ao paciente. Se não houver nenhum, vale o protocolo do 1º trecho recuperado.
- **Intenções** são detectadas por palavras-chave, no texto em minúsculas e sem acentos.
- **Deduplicação:** se a busca já recuperou uma seção que alguma regra exigiria, ela fica só no bloco por similaridade, com a anotação "também exigida por regra: …". Uma seção nunca aparece duas vezes.
- **Ordem reproduzível:** primeiro os trechos recuperados, na ordem do retriever. Depois as seções por regra, na ordem em que a primeira regra as selecionou.
- **Seção inexistente:** se a política exigir uma seção que não existe no índice (por exemplo, porque um protocolo mudou), o resultado é `KeyError`, nunca omissão silenciosa.
- **Status:** a política não remove nem reordena trechos recuperados e **não altera o status** do retriever. Um teste confere que o `RetrievalResult` continua idêntico ao de `retriever.retrieve`.

## O que a entrada contém

1. **Mensagem de sistema:** regras de comportamento e segurança.
   - Usar só o contexto, citar referências e não inventar valores.
   - Distinguir trechos por similaridade de seções por regra; o status vale só para os primeiros.
   - Conferir a condição de cada trecho.
   - Recusar prescrição e diagnóstico, encaminhar ao médico responsável e não aplicar a pacientes reais.
2. **Mensagem do usuário:**
   - data de referência e pergunta;
   - `## Trechos recuperados por similaridade (status da recuperação: …)`: `[1]`, `[2]`… com referência, versão, `SIMULACAO`, origem e texto literal, mais a anotação de regra quando houver;
   - `## Seções adicionadas por regras determinísticas (independentes da similaridade)`: `[R1]`, `[R2]`… com os mesmos campos e `regra: <motivos>`, ou a frase "Nenhuma seção adicionada por regra.";
   - dados do paciente (JSON de `get_patient_summary`) ou a frase "Nenhum paciente citado na pergunta.".

**O que fica de fora, de propósito:**
- **Scores de similaridade.** Não são probabilidade de acerto.
- **Resposta esperada, `qa_id`, categoria, evidências, `protocolos_citados`, flags e `verificacoes_esperadas`.** São rótulos.
- **Qualquer seleção baseada na resposta.**

**Paciente:** o `patient_id` precisa aparecer no texto da pergunta, porque na inferência o sistema
só sabe qual é o paciente se ele for informado.

**Falta de dados e de evidências:**
- Exame ausente aparece como `"status": "ausente"` com valores `null`.
- Recuperação `incerto` ou `sem_resultado_confiavel` aparece com o status original e uma observação. As seções por regra **não** promovem o status.
- Sem trechos e sem regras aplicáveis, a entrada diz "Nenhum trecho recuperado." e "Nenhuma seção adicionada por regra.".
- Paciente inexistente gera `PatientNotFoundError`.

## Treino × inferência

| | Treino | Inferência (futuro LangGraph) |
|---|---|---|
| Contexto | `build_context(..., excluded_refs=RESERVED_SECTIONS)` | `build_context(...)` |
| Política | `select_rule_sections`, a mesma função | `select_rule_sections` |
| Mensagens | `build_messages(contexto, resposta)` → system, user, assistant | `build_messages(contexto)` → system, user |
| Prompt | `SYSTEM_PROMPT` versão `PROMPT_VERSION` | **o mesmo**; mudar o prompt ou a política exige regenerar os dados e treinar de novo |

A única diferença é `excluded_refs`. No treino, `PROT-DM2-002#S4` e `PROT-DM2-003#S4` são
removidas **tanto dos trechos recuperados quanto das seções por regra**. A regra
`protocolo_vinculado` lista as duas, e elas aparecem na inferência. Um teste compara as entradas
de treino e de inferência quando nenhuma seção reservada está envolvida: elas são idênticas.

## Isolamento da avaliação

- `sources.py` lê apenas `finetune/qa_train.jsonl`; um teste roda num diretório sem `eval/`.
- Exemplos que citam PAC-0009 ou PAC-0010, ou que dependem de seção reservada, geram `ReservedDataError`.
- `sft.py` confere, para cada exemplo, que a entrada não contém texto ou referência de seção reservada, paciente reservado nem a resposta esperada.

> `scripts/build_database.py` continua executando o validador completo do dataset, que lê o
> `qa_eval.jsonl` apenas para checar vazamento entre treino e avaliação (comportamento da Etapa 2).
> O código de fine-tuning não o lê.

## Cobertura das referências (conjunto original, QA-0001..QA-0020)

A cobertura é **auditoria**: compara `protocolos_citados` com o contexto **depois** que ele foi
montado. Com k = 4, retriever e limiares inalterados, e a política acima:

| Medida | Resultado |
|---|---|
| Cobertura completa (todas as citadas no contexto) | **20/20** |
| Com referências ausentes | 0 |
| Contexto insuficiente (nenhuma citada no contexto) | 0 |
| Status original do retriever | ok = 9 · incerto = 9 · sem_resultado_confiavel = 2 (inalterado) |
| Fundamentação das citadas | só regras = 13 · só similaridade = 4 · ambas = 3 |
| Seções por regra por exemplo | 0 a 6 (46 no total) |
| Mensagem do usuário | 1.918 a 8.669 caracteres (mediana 7.136) |

Antes da política, 11 de 20 exemplos citavam seções fora do contexto.

**Exemplos com `sem_resultado_confiavel` (status mantido):**
- **QA-0006** (pendências de PAC-0002): as três seções citadas (`001#S3`, `001#S4`, `003#S3`) vieram da busca, mas também são exigidas por `regras_das_verificacoes`. A fundamentação é `regras`. Resposta mantida.
- **QA-0007** (pendências de PAC-0007):
  - `001#S3` e `003#S3` vêm de `regras_das_verificacoes`;
  - `003#S5` (fundo de olho) vem de `protocolo_vinculado:PROT-DM2-003`;
  - `001#S4` e `001#REV` foram recuperadas e também são exigidas por regra.

  A fundamentação é `regras`. Resposta mantida.

Nos dois casos, a entrada mostra o status `sem_resultado_confiavel` da busca **e** o bloco de
regras com os motivos. O modelo aprende que a resposta se apoia nas regras ligadas aos dados do
paciente, e não na similaridade.

## Generalização: lote piloto 1 (consolidado)

> Esta seção registra primeiro o **resultado original** do pré-registro, que é a medida da
> política v1 em perguntas novas. As correções feitas depois estão em "Consolidação" e **não** são
> desempenho independente da política.

**Política congelada:** `POLICY_VERSION = "1"`. As regras são as do commit `dc57d96`; só a
constante de versão foi acrescentada. O SHA-256 de `context_policy.py` está no pré-registro.

**Ordem do procedimento:**
1. Gravar as perguntas, sem respostas (`data/finetune/piloto1/perguntas.jsonl`).
2. Rodar `scripts/preregister_context.py`, que salva trechos, scores, status e seções por regra (`data/finetune/piloto1/contexto_preregistrado.json`).
3. Escrever as respostas.
4. Medir a cobertura.

O contexto gerado no passo 4 é idêntico ao pré-registrado, e o hash do arquivo não mudou depois
que as respostas foram escritas.

Resultado original do pré-registro (41 perguntas QA-0031..QA-0071, antes de qualquer correção):

| Medida | Original (20) | Piloto 1 (41) |
|---|---|---|
| Status original do retriever | ok 9 · incerto 9 · sem_resultado 2 | ok 17 · incerto 18 · sem_resultado 6 |
| Cobertura completa | 20 | **36** |
| Parcial (algumas citadas fora do contexto) | 0 | 4 |
| Insuficiente (nenhuma citada no contexto) | 0 | 1 |
| Fundamentação | regras 13 · similaridade 4 · ambas 3 | regras 21 · similaridade 14 · ambas 4 · nenhuma 2 |
| Exemplos sem nenhuma seção por regra | 4 | 14 |

A fundamentação "nenhuma" corresponde a QA-0065 (`contexto_insuficiente`, sem citações por
desenho) e a QA-0056.

**Lacunas encontradas (a política não foi alterada para escondê-las):**

| Exemplo | Status | Falta | Causa |
|---|---|---|---|
| QA-0032 | ok | `001#S2` | O `003#S2` remete aos critérios do PROT-DM2-001; a política não segue referências cruzadas |
| QA-0049 | sem_resultado | `001#LIM` | "começar uma estatina" não dispara `pedido_de_conduta` |
| QA-0056 | incerto | `001#LIM` | "antibiótico… devo usar" não dispara nenhuma intenção; sem paciente, não há outras regras |
| QA-0064 | incerto | `003#REV` | Pergunta sem paciente e sem intenção; a seção não foi recuperada |
| QA-0071 | ok | `002#S3` | A busca trouxe S5, S1, LIM e REV do PROT-DM2-002, mas não a S3 |

**Leitura:**
- As regras ligadas aos dados do paciente generalizam bem: as 17 perguntas novas com paciente e com `dados_ausentes`/`contextualizada` ficaram todas cobertas.
- A detecção de intenção por palavras-chave generaliza mal: disparou em 3 de 41 perguntas e falhou em paráfrases ("começar", "antibiótico", "coloque um valor", "apague").
- Perguntas sem paciente dependem só da busca semântica.

### Consolidação

**Decisão:** reescrever as 5 respostas para citar só o que está no contexto. A política **não** foi
alterada: `context_policy.py` continua com o SHA-256 `11f1e9bb…3593` registrado no pré-registro, e o
único diff em relação a `dc57d96` é a constante `POLICY_VERSION = "1"`. Uma política v2 (referências
cruzadas, vocabulário de intenção) continua em aberto. Se for criada, a cobertura terá de ser medida
num lote novo, ainda sem respostas.

**Correções** (detalhes em [data/DATASET_CARD.md](../data/DATASET_CARD.md)):
- QA-0049 e QA-0064: a frase apoiada na seção ausente foi retirada.
- QA-0032, QA-0056 e QA-0071: a resposta declara o que falta no contexto e responde com o que está nele.

**Substituições:** 6 exemplos foram trocados por QA-0072..QA-0077, para aumentar a diversidade e
reduzir a proximidade com a avaliação. Essas perguntas foram pré-registradas com a política v1
(`piloto1/revisao/contexto_preregistrado_substitutas.json`) antes de as respostas serem escritas.

Cobertura após a consolidação (`build_sft_dataset.py --strict`, k = 4):

| Grupo | Exemplos | Status do retriever | Cobertura | Fundamentação |
|---|---|---|---|---|
| Piloto mantido sem alteração | 30 | — | completa 30 (já era) | — |
| Piloto corrigido (lacunas) | 5 | (inalterado) | completa 5 **por reescrita da resposta** | similaridade 4 · regras 1 |
| Substitutas (pré-registro novo) | 6 | ok 6 | completa 6 | regras 4 · similaridade 1 · ambas 1 |
| **Piloto consolidado** | **41** | ok 22 · incerto 14 · sem_resultado 5 | **completa 41** | regras 21 · similaridade 15 · ambas 4 · nenhuma 1 |
| Treino total (v1 + piloto) | 61 | ok 31 · incerto 23 · sem_resultado 7 | completa 61 | regras 34 · similaridade 19 · ambas 7 · nenhuma 1 |

**Como ler estes números:**
- **41/41 não é a nova taxa da política.** A medida da política em perguntas novas continua sendo
  o resultado original: **36 completas, 4 parciais e 1 insuficiente**. Os 41/41 vêm de respostas
  ajustadas ao contexto.
- **Nas 6 substitutas, a cobertura completa também não é uma medida independente.** As respostas
  foram escritas depois de ver o contexto pré-registrado. O pré-registro garante que a seleção não
  dependeu das respostas, mas as respostas dependeram da seleção. Exemplo: na QA-0072 a seção de
  objetivo do PROT-DM2-002 (`002#S1`) não foi recuperada, e a resposta se apoiou no título do
  protocolo, que aparece no cabeçalho dos trechos `002#REV` e `002#LIM`.
- Os contextos gerados para os 41 exemplos (35 originais + 6 substitutas) foram comparados com os
  pré-registros (trechos, status e seções por regra): **0 divergências**.
- Mensagem do usuário no treino consolidado: 1.790 a 9.284 caracteres (mediana 6.806).

## Limitações

- **Validação dentro da amostra.** A política foi desenhada depois de ver quais seções faltavam nestes 20 exemplos. As regras são genéricas (dados do paciente, protocolo vinculado, intenção) e não dependem da resposta, mas o 20/20 não prova cobertura em perguntas novas. É preciso medir de novo nos exemplos da ampliação, antes de olhar as respostas.
- **Intenção por palavras-chave:** é frágil diante de paráfrases e pode disparar sem necessidade.
  - No conjunto de desenvolvimento do RAG, 4 de 30 perguntas ativam regras, todas por `pedido_de_conduta`.
  - Uma delas está fora do escopo: "dose de amoxicilina" recebe `001#REV`/`001#LIM`. Isso é inofensivo, porque a REV manda recusar prescrição, mas acrescenta contexto a uma pergunta que deveria ser recusada por falta de cobertura.
- **Tamanho:** com as seções por regra e o JSON do paciente, a entrada chega a cerca de 8.700 caracteres (cerca de 2.500 a 3.000 tokens), mais o prompt de sistema. O limite de contexto do modelo-base deve comportar isso, ou o JSON precisa ser compactado.
- **Absorção pelo LangGraph:** a política e a montagem de contexto estão em `medprotocol.finetuning` por serem usadas primeiro aqui. Na etapa do LangGraph, podem ser movidas para um módulo compartilhado, sem mudar o comportamento.
- **QA-0019** afirmava que "o protocolo simulado não define critérios de alta", uma afirmação de ausência feita sem todas as seções no contexto. Corrigida no lote piloto 1 para "os trechos de protocolo disponíveis não trazem critérios".
- **Validação:** a separação entre treino e validação (por paciente e estratificada por categoria, sem tocar no `qa_eval`) será feita depois da ampliação controlada.
