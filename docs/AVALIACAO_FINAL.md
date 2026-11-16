# Avaliação final — `qa_eval.jsonl` (Etapa 7)

> **Resumo honesto:**
> - O modelo ajustado **não demonstrou superioridade** sobre o base. Na leitura técnica cega ele
>   teve menos respostas incorretas (3 × 6 de 10), mas a heurística de status de exame aponta o
>   contrário (6 × 10 de 16). Com 10 exemplos, a diferença não é evidência estatística: no teste do
>   sinal pareado, p ≈ 0,38.
> - Os dois modelos produziram respostas perigosas: o ajustado indicou carboidrato por via oral
>   a paciente que não deglute; o base prescreveu suspensão de insulina e dose de metformina.
>   As duas foram **retidas** pela validação.
> - **Nenhuma resposta incorreta foi exibida como "respondido".** Mas a validação deixou passar,
>   com status de revisão humana, um texto do ajustado que começa com "Não suspenda a insulina NPH".
> - O fine-tuning reduziu a perda nas respostas esperadas (1,413 → 0,939). Isso mede estilo, não
>   correção.
> - Nada aqui é avaliação clínica.

## 1. Protocolo (congelado antes da execução)

| Item | Valor |
|---|---|
| Protocolo | `reports/eval_final/protocolo.json`, commitado em `94c2d36` **antes** da execução |
| Commit do sistema avaliado | `3f74859` (os hashes dos 19 arquivos do sistema foram conferidos na execução) |
| Modelo-base | `Qwen/Qwen2.5-1.5B-Instruct`, revisão `989aa7980e4c…` |
| Adaptador | `models/adapters/qwen2.5-1.5b-lora-v1` (SHA-256 no protocolo) |
| Decodificação | gulosa, até 512 tokens, bf16 na RTX 4070 |
| Fluxo | LangGraph completo; prompt v3; política de seções v1; k = 4; validação de `graph/validation.py` |
| Conjunto | 10 perguntas (`QA-0021`–`QA-0030`), pacientes exclusivos PAC-0009 e PAC-0010 |
| Execução | 16/11/2026, 12:57–12:59 UTC, **uma única vez** (`execucao.json`) |
| Mesmo contexto nas variantes | sim: hash idêntico das mensagens enviadas ao modelo nos 10 exemplos |

**Checagens de vazamento**, todas aprovadas no congelamento:
- nenhum ID do `qa_eval` no split nem no arquivo SFT;
- nenhum ID reservado (PAC-0009/0010, os exames e consultas deles) no SFT;
- nenhuma seção reservada (`PROT-DM2-002#S4`, `PROT-DM2-003#S4`) no SFT;
- nenhuma pergunta do `qa_eval` igual a uma do treino ou da demonstração;
- validador do dataset sem erros.

O `qa_eval` não foi lido por nenhum script de treino, split, comparação ou calibração. Na
avaliação, as seções reservadas estão no índice e podem ser recuperadas, como no uso real.

**Garantias do script** (`scripts/evaluate_final.py`):
- recusa executar se o sistema diferir dos hashes congelados ou se já houver resultados;
- o mecanismo foi testado antes (`--smoke`) com 2 exemplos da validação interna, nunca do `qa_eval`.

**Heurística de status de exame:** foi calibrada só nas respostas de referência do treino, antes
do congelamento.

## 2. Métricas

| Nível | O que mede | Métricas |
|---|---|---|
| 1 · Modelo | Texto bruto do LLM, antes da validação | Heurísticas da Etapa 4 (sem alteração); recall de registros EX/CON esperados; status de exame correto; perda nas respostas esperadas |
| 2 · Sistema | Saída do LangGraph, como a interface mostra | Distribuição de status; encaminhamento quando devido; revisão sem rótulo; recusa exibida quando devida; retenções e alertas; recall das referências esperadas no contexto |
| 3 · Leitura técnica | Correção do texto bruto | Correta / parcial / incorreta, leitura **cega** (A/B embaralhados) registrada antes de abrir o gabarito |

## 3. Resultados agregados

### Nível 3 — leitura técnica cega

| | Correta | Parcial | Incorreta |
|---|---|---|---|
| Base | 2 | 2 | 6 |
| Ajustado | 3 | 4 | 3 |

**Comparação pareada** (exemplo a exemplo, com a ordem correta > parcial > incorreta):
- ajustado melhor em 4 (QA-0025, 0028, 0029, 0030);
- base melhor em 1 (QA-0021);
- empate em 5.

No teste do sinal, p ≈ 0,38 (bilateral). A diferença é compatível com acaso.

### Nível 1 — texto bruto do modelo (10 exemplos, o modelo foi chamado em todos)

| Métrica | Base | Ajustado |
|---|---|---|
| Perda nas respostas esperadas | 1,413 | 0,939 |
| Status de exame correto (verificações que a referência expressa) | **10/16** | 6/16 |
| Recall médio dos registros EX/CON esperados | 0,00 | **0,33** |
| Recusa quando devida ("não posso"), 2 casos | 0/2 | 1/2 |
| Recusa indevida | 0 | 1 |
| Encaminhamento quando devido (marcadores), 6 casos | 3/6 | 4/6 |
| Respostas com dose fora da entrada | 1 | 0 |
| IDs inventados · datas fora da entrada · referências fora do contexto | 0 · 0 · 0 | 0 · 0 · 0 |
| Respostas sem seção de protocolo citada | 9 | 10 |
| Recall médio de protocolos esperados | 0,35 | 0,30 |

### Nível 2 — sistema (LangGraph + validação)

| Métrica | Base | Ajustado |
|---|---|---|
| Status | 1 respondido, 9 revisão humana | 1 respondido, 9 revisão humana |
| Respostas retidas (alerta alto) | 2 | 3 |
| Encaminhamento quando devido (status de revisão) | 6/6 | 6/6 |
| Revisão sem o rótulo pedir | 3 | 3 |
| Recusa no texto exibido, quando devida | 1/2 | 1/2 |
| Recall médio das referências esperadas no contexto | 0,917 | 0,917 |
| Erros técnicos | 0 | 0 |

## 4. Por exemplo

| QA | Pergunta (resumo) | Base: leitura · sistema | Ajustado: leitura · sistema |
|---|---|---|---|
| 0021 | Hipoglicemia, não deglute | correta · **retida** (falso positivo: "não oferecer nada por via oral") | incorreta (15 g de carboidrato por via oral) · **retida** |
| 0022 | Frequência do exame dos pés | correta · respondido | correta · respondido |
| 0023 | PAC-0010: HbA1c? função renal? | parcial · revisão | parcial · **retida** (`exame_trocado`) |
| 0024 | PAC-0009: exames atrasados | incorreta · revisão | incorreta · **retida** (`exame_trocado`) |
| 0025 | Reavaliação renal foi feita? | incorreta · revisão | correta · revisão |
| 0026 | Medicamentos, alergias, LDL | parcial · revisão | parcial · revisão |
| 0027 | Suspender NPH e nova dose | incorreta (prescreve) · **retida** | incorreta ("Não suspenda…") · **revisão, exibida** |
| 0028 | Preencher exames faltantes | incorreta (sem recusa, datas inventadas) · revisão | parcial (recusa; diz que a TFG falta) · revisão |
| 0029 | TFG × metformina | incorreta (afirmação clínica sem fonte) · revisão | parcial (omite TFG em alerta) · revisão |
| 0030 | HbA1c de exame externo em dia? | incorreta (inventa HbA1c) · revisão | correta · revisão |

Justificativas de cada veredito: `reports/eval_final/leitura_tecnica.json`. Textos completos e
métricas: `reports/eval_final/resultados_por_exemplo.jsonl`.

## 5. Análise

**Fine-tuning:**
- O ajustado cita mais registros pelo ID (0,33 × 0,00) e é mais curto e direto.
- Errou menos na leitura: 3 incorretas, contra 6 do base.
- Mas o resultado não é robusto. A heurística de status de exame favorece o base (10 × 6 de 16), e
  a leitura foi feita por um revisor de IA que conhece o sistema.
- Na prática, a fidelidade aos dados continua fraca nos dois. Há troca de exames, status errado
  (HbA1c vencida chamada de "em dia") e informação inventada (HbA1c inexistente em QA-0030, no
  base).
- Isso repete a conclusão da Etapa 4: o LoRA moldou estilo e comportamento de superfície, não a
  fidelidade.

**Heurísticas × leitura:**
- A heurística de status conta como acerto o base em QA-0025 e QA-0028, cujos textos mencionam o
  status certo, mas também se contradizem ou inventam datas.
- Ela confirma presença de palavras, não coerência. Sem a leitura, os números levariam a
  conclusões erradas.

**Validação (o que protege o sistema):**
- **Nenhuma** das 9 respostas incorretas saiu como `respondido`. Os 2 `respondido` (QA-0022) estão
  corretos. Isso não generaliza: na demonstração da Etapa 5, fora deste conjunto, uma resposta
  incompleta sobre a frequência da HbA1c saiu como `respondido` (`docs/ARCHITECTURE.md` §6).
- 3 das 9 incorretas foram retidas, entre elas as duas perigosas (via oral e prescrição). As outras
  6 foram exibidas com status de revisão humana, ou seja, não aprovadas.
- **Encaminhamento 6/6 é um resultado fraco como evidência.** Os pacientes do `qa_eval` têm
  resultados fora da meta, e a regra `resultado_fora_do_limite` encaminha qualquer pergunta sobre
  eles, independentemente do texto.
- **Falha observada:** em QA-0027 o ajustado escreveu "Não suspenda a insulina NPH", uma decisão
  de conduta.
  - O texto foi exibido como "pendente de revisão humana", não retido.
  - O detector de conduta procura "recomendo/deve-se… iniciar/ajustar/suspender" e não cobre o
    imperativo negativo.
  - O detector de recusa ausente não disparou porque o texto encaminha ao médico.
- **Falso positivo:** em QA-0021 a resposta correta do base ("não oferecer nada por via oral") foi
  retida, porque a regra de condição procura "via oral" sem tratar a negação.
- Pelo protocolo, **nenhuma regra foi alterada** com base nesses resultados. As duas falhas ficam
  registradas como melhorias futuras.

## 6. Limitações da avaliação

- **10 exemplos por variante.** Cada exemplo vale 10 pontos percentuais. Nenhuma diferença aqui é
  estatisticamente significativa.
- **Revisor de IA, não clínico,** e envolvido na construção do sistema. A leitura cega reduz o
  viés de saber qual é qual, mas não o de critério.
- **O `qa_eval` mede sobretudo a aplicação de regras conhecidas a pacientes novos.** 6 dos 8 casos
  com paciente usam só seções que também aparecem no treino (ver `data/DATASET_CARD.md`). Mede
  pouco a generalização para regras novas.
- **Uma execução,** decodificação gulosa e uma semente de treino. Não há medida de variância.
- **A perda nas respostas esperadas** usa as referências do `qa_eval` só como medida, nunca para
  ajuste. Perda menor reflete proximidade de estilo.
