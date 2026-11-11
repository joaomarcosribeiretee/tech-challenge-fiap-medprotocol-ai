# Formato dos dados para fine-tuning

Transforma os exemplos de `data/finetune/qa_train.jsonl` em conversas de chat com o **mesmo
contexto que o sistema terá na inferência**:
- trechos recuperados pelo RAG (Etapa 3);
- seções adicionadas por regras determinísticas;
- dados do paciente lidos do SQLite (Etapa 2).

O objetivo é treinar o modelo a **ler** o contexto e citar o que está nele, e não a decorar
valores de pacientes.

Esta etapa não treina nenhum modelo: só prepara e verifica o formato.

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
 "metadata": {"qa_id": "QA-0014", "categoria": "recusa", "prompt_version": "2", "k": 4,
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

## Cobertura das referências (resultado atual)

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

## Limitações

- **Validação dentro da amostra.** A política foi desenhada depois de ver quais seções faltavam nestes 20 exemplos. As regras são genéricas (dados do paciente, protocolo vinculado, intenção) e não dependem da resposta, mas o 20/20 não prova cobertura em perguntas novas. É preciso medir de novo nos exemplos da ampliação, antes de olhar as respostas.
- **Intenção por palavras-chave:** é frágil diante de paráfrases e pode disparar sem necessidade.
  - No conjunto de desenvolvimento do RAG, 4 de 30 perguntas ativam regras, todas por `pedido_de_conduta`.
  - Uma delas está fora do escopo: "dose de amoxicilina" recebe `001#REV`/`001#LIM`. Isso é inofensivo, porque a REV manda recusar prescrição, mas acrescenta contexto a uma pergunta que deveria ser recusada por falta de cobertura.
- **Tamanho:** com as seções por regra e o JSON do paciente, a entrada chega a cerca de 8.700 caracteres (cerca de 2.500 a 3.000 tokens), mais o prompt de sistema. O limite de contexto do modelo-base deve comportar isso, ou o JSON precisa ser compactado.
- **Absorção pelo LangGraph:** a política e a montagem de contexto estão em `medprotocol.finetuning` por serem usadas primeiro aqui. Na etapa do LangGraph, podem ser movidas para um módulo compartilhado, sem mudar o comportamento.
- **QA-0019** afirma que "o protocolo simulado não define critérios de alta". É uma afirmação de ausência: o contexto traz quase todo o PROT-DM2-001, mas não todas as seções. O risco é baixo; vale revisar na ampliação.
- **Validação:** a separação entre treino e validação (por paciente e estratificada por categoria, sem tocar no `qa_eval`) será feita depois da ampliação controlada.
