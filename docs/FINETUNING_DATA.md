# Formato dos dados para fine-tuning

Transforma os exemplos de `data/finetune/qa_train.jsonl` em conversas de chat com o **mesmo
contexto que o sistema terá na inferência**: trechos recuperados pelo RAG (Etapa 3) e dados do
paciente lidos do SQLite (Etapa 2). O objetivo é treinar o modelo a **ler** o contexto e citar o
que está nele, e não a decorar valores de pacientes.

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
- `metadata`: dados para conferência, que **não** são enviados ao modelo.

```json
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
 "metadata": {"qa_id": "QA-0014", "categoria": "recusa", "prompt_version": "1", "k": 4,
              "data_referencia": "2026-06-30", "patient_id": "PAC-0003",
              "status_recuperacao": "incerto", "refs_recuperadas": ["PROT-DM2-003#S3", "..."],
              "refs_citadas_fora_do_contexto": ["PROT-DM2-001#S4"]}}
```

## Camadas (`src/medprotocol/finetuning/`)

| Módulo | Responsabilidade | Usado na inferência? |
|---|---|---|
| `sources.py` | Lê **só** `qa_train.jsonl`. Separa entrada (pergunta, paciente citado), alvo (resposta) e rótulos. Rejeita pacientes e seções reservados | Não |
| `context.py` | Monta o `AssistantContext`: pergunta, data de referência, `RetrievalResult` do retriever e `get_patient_summary` do repositório | **Sim** |
| `messages.py` | Converte o contexto em mensagens `system`/`user` (e `assistant` no treino). Contém o prompt de sistema versionado (`PROMPT_VERSION`) | **Sim** |
| `sft.py` | Junta as camadas, faz as verificações de isolamento e calcula a cobertura | Não |

Nenhuma regra de negócio é reimplementada: prazos, pendências e limites vêm de
`medprotocol.data.schedule`, via `repository.get_patient_summary`; os trechos e o status vêm de
`ProtocolRetriever.retrieve`. Os limiares do RAG não foram alterados.

## O que a entrada contém

1. **Mensagem de sistema:** regras de comportamento e segurança (usar só o contexto, citar
   referências, não inventar valores, tratar o status da recuperação, conferir a condição de cada
   trecho, recusar prescrição e diagnóstico, encaminhar ao médico responsável, não aplicar a
   pacientes reais).
2. **Mensagem do usuário:**
   - data de referência;
   - pergunta;
   - trechos recuperados (k = 4 por padrão), cada um com referência `PROT#SEÇÃO`, versão, status `SIMULACAO`, origem e texto literal, mais o **status da recuperação** e uma observação sobre o que ele significa;
   - dados do paciente (JSON de `get_patient_summary`) quando a pergunta cita um paciente; caso contrário, a frase "Nenhum paciente citado na pergunta.".

**O que fica de fora, de propósito:**
- **Scores de similaridade.** Não são probabilidade de acerto (veja [RAG.md](RAG.md)), e o modelo poderia tratá-los como confiança.
- **Resposta esperada, `qa_id`, categoria, evidências, `protocolos_citados`, flags e `verificacoes_esperadas`.** São rótulos: o sistema real não os teria.
- **Qualquer seleção de trechos baseada na resposta.** A consulta ao retriever é só a pergunta.

**Paciente:** o `patient_id` precisa aparecer no texto da pergunta. Na inferência, o sistema só sabe
qual é o paciente se ele for informado.

**Falta de dados:**
- Exame ausente aparece como `"status": "ausente"` com valores `null`, e o exame entra em `pendencias`.
- Recuperação `incerto` ou `sem_resultado_confiavel` aparece com o status e uma observação explícita.
- Sem trechos, a entrada diz "Nenhum trecho recuperado.".
- Paciente inexistente gera `PatientNotFoundError`; nunca vira contexto vazio.

## Treino × inferência

| | Treino | Inferência (futuro LangGraph) |
|---|---|---|
| Contexto | `build_context(..., excluded_refs=RESERVED_SECTIONS)` | `build_context(...)` |
| Mensagens | `build_messages(contexto, resposta)` → system, user, assistant | `build_messages(contexto)` → system, user; o modelo gera a resposta |
| Prompt de sistema | `SYSTEM_PROMPT` versão `PROMPT_VERSION` | **o mesmo**; mudar o prompt exige regenerar os dados e treinar de novo |

A única diferença é `excluded_refs`. No treino, as seções reservadas para a avaliação final
(`PROT-DM2-002#S4` e `PROT-DM2-003#S4`) são removidas do resultado **como se não estivessem no
índice**: o retriever busca k + 2 trechos, descarta os reservados, mantém os k primeiros e
recalcula o status com `classify` sobre o melhor trecho restante. Assim elas não entram na entrada
nem como distratores. Na inferência, o índice é completo.

## Isolamento da avaliação

- `sources.py` lê apenas `finetune/qa_train.jsonl`; um teste roda num diretório sem `eval/`.
- Exemplos que citam PAC-0009 ou PAC-0010, em qualquer campo, ou que dependem de seção reservada geram `ReservedDataError`.
- `sft.py` confere, para cada exemplo, que a entrada não contém:
  - o texto ou a referência de seção reservada;
  - paciente reservado;
  - a resposta esperada.
- Nenhum exemplo atual depende de seção reservada. A seção `002#S4` aparecia entre os 4 primeiros trechos de QA-0002 e QA-0010 e é removida pela exclusão.

> `scripts/build_database.py` continua executando o validador completo do dataset, que lê o
> `qa_eval.jsonl` apenas para checar vazamento entre treino e avaliação (comportamento da Etapa 2).
> O código de fine-tuning não o lê.

## Cobertura das referências (resultado atual)

Com k = 4 e o retriever da Etapa 3, sem nenhum ajuste:

| Status da recuperação | Exemplos |
|---|---|
| `ok` | 9 |
| `incerto` | 9 |
| `sem_resultado_confiavel` | 2 |

**11 de 20 respostas citam pelo menos uma seção que não está entre os trechos recuperados**
(lista em `metadata.refs_citadas_fora_do_contexto`). Os casos mais comuns:
- perguntas sobre paciente, como "pendências de PAC-0007", que recuperam mal as seções de regra (`001#S3`, `001#S4`, `001#REV`);
- `LIM` e `REV` citadas como fundamento de recusa ou encaminhamento.

Um caso relevante para a segurança: no QA-0010 (hipoglicemia de PAC-0006), a seção de conduta
`002#S3` **não** foi recuperada.

Treinar com esses exemplos como estão ensinaria o modelo a **citar seções que ele não viu**, o
oposto do objetivo. Por isso a cobertura é medida e reportada, e não corrigida: escolher k ou
trechos olhando a resposta esperada é proibido. As opções estão em "Decisões pendentes".

## Decisões pendentes

1. **Como tratar a falta de cobertura** (antes de qualquer treino):
   - **(a)** Anexar de forma determinística as seções que são a fonte das regras calculadas (`001#S3`, `001#S4`, `003#S3`) sempre que houver dados de paciente no contexto, porque as `verificacoes` vêm delas. Não depende da resposta, mas é uma regra de orquestração que o LangGraph também teria que aplicar.
   - **(b)** Anexar `LIM` e `REV` de cada protocolo recuperado: são seções de segurança curtas.
   - **(c)** Aumentar k para todas as perguntas. O custo é mais contexto e mais distratores.
   - **(d)** Reescrever as respostas para citar apenas o que for recuperado.
   - **(e)** Descartar os exemplos sem cobertura completa. Com 20 exemplos, isso é inviável.
2. **Tamanho da entrada:** o JSON do paciente com `indent=2` leva a mensagem do usuário a até cerca de 5.600 caracteres. Um formato compacto reduziria tokens; isso deve ser decidido junto com o modelo-base e o limite de contexto.
3. **Status no treino:** 2 exemplos têm `sem_resultado_confiavel` e, ainda assim, a resposta usa os trechos. Isso pode ensinar o modelo a ignorar o status. Avaliar junto com a decisão 1.
4. **Validação:** a separação entre treino e validação (por paciente e estratificada por categoria, sem tocar no `qa_eval`) só será feita depois da ampliação controlada do dataset.
