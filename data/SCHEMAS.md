# Schemas do dataset sintético (v1)

Todos os schemas são verificados por `scripts/validate_dataset.py`
(lógica em `src/medprotocol/data/validation.py`). Data de referência fixa: **2026-06-30**.

## 1. Protocolos — `protocols/PROT-DM2-NNN.md`

Markdown com cabeçalho `chave: valor` entre linhas `---`.

| Campo | Valores | Observação |
|---|---|---|
| `protocol_id` | `PROT-DM2-NNN` | igual ao nome do arquivo |
| `titulo` | texto | |
| `versao` | texto (ex.: `1.0`) | regras de `schedule.py` são amarradas à versão |
| `vigencia` | `AAAA-MM-DD` | |
| `status` | `SIMULACAO` | único valor aceito |
| `natureza_regras` | `simulada` \| `baseada_em_referencia_verificada` | origem das regras |
| `referencias_externas_verificadas` | `nenhuma` ou a referência | `simulada` exige `nenhuma`; o outro valor exige referência |
| `validacao_clinica` | `nao_realizada` | único valor aceito nesta versão |
| `escopo` | texto | |

Corpo: aviso contendo `SIMULAÇÃO` e seções `## [ID] Título`. Seções obrigatórias:
`[S1]` objetivo, `[LIM]` limites de utilização, `[REV]` encaminhamento para revisão humana.
Referência a seção: `PROT-DM2-001#S3`.

## 2. Pacientes, consultas e exames — `synthetic/*.csv`

UTF-8, cabeçalho exato (coluna extra ou faltante é erro, impedindo por exemplo uma coluna `nome`).

**`patients.csv`**: `patient_id` (`PAC-NNNN`), `sexo` (`F`/`M`), `ano_nascimento` (adulto),
`ano_diagnostico_dm2`, `comorbidades`, `medicamentos_em_uso`, `alergias`
(listas separadas por `;`, `nenhuma` quando vazio).

**`consultations.csv`**: `consultation_id` (`CON-NNNN`), `patient_id`, `data`,
`tipo` (`ambulatorial`/`internacao`), `protocol_id`, `resumo`.

**`exams.csv`**: `exam_id` (`EX-NNNN`), `patient_id`, `tipo`, `valor`, `unidade`, `data_coleta`.

| tipo | unidade | faixa plausível |
|---|---|---|
| HBA1C | % | 4–15 |
| GLICEMIA_JEJUM | mg/dL | 40–600 |
| GLICEMIA_CAPILAR | mg/dL | 20–600 |
| CREATININA | mg/dL | 0,3–10 |
| TFG | mL/min/1.73m2 | 5–150 |
| RAC | mg/g | 0–3000 |
| LDL | mg/dL | 20–400 |

Datas de consultas e exames não podem ser posteriores à data de referência.

## 3. Perguntas e respostas — `finetune/qa_train.jsonl` e `eval/qa_eval.jsonl`

Um objeto JSON por linha.

| Campo | Tipo | Regra |
|---|---|---|
| `qa_id` | `QA-NNNN` | único entre os dois arquivos |
| `split` | `train` \| `eval` | deve corresponder ao arquivo |
| `categoria` | ver abaixo | |
| `pergunta`, `resposta_esperada` | texto | não vazios |
| `patient_id` | `PAC-NNNN` \| `null` | |
| `protocolos_citados` | lista de `PROT-…#Sx` | igual ao conjunto das evidências de protocolo |
| `evidencias` | lista de `{tipo, ref, trecho}` | ver abaixo; ao menos uma de protocolo |
| `requer_revisao_humana` | bool | a resposta envolve decisão do profissional |
| `deve_recusar` | bool | o assistente recusa a ação pedida |
| `verificacoes_esperadas` | lista (opcional) | recalculada pelo validador |

### Evidências

| tipo | ref | trecho deve ser |
|---|---|---|
| `protocolo` | `PROT-DM2-NNN#ID` | trecho literal da seção (espaços normalizados) |
| `exame` | `EX-NNNN` do mesmo paciente | exatamente `TIPO valor unidade (data)`, ex.: `HBA1C 8.4 % (2026-05-12)` |
| `consulta` | `CON-NNNN` do mesmo paciente | trecho literal do `resumo` |
| `paciente` | o próprio `patient_id` | uma linha `campo: valor` do cadastro |

### Categorias e regras

| categoria | regras verificadas |
|---|---|
| `consulta_protocolo` | sem `patient_id` |
| `dados_ausentes` | `patient_id` + evidência do paciente + ao menos uma verificação `vencido`/`ausente` |
| `contextualizada` | `patient_id` + evidência do paciente |
| `recusa` | `deve_recusar = true`; `requer_revisao_humana` pode ser `true` (recusa e encaminha) ou `false` |
| `revisao_humana` | `requer_revisao_humana = true` e `deve_recusar = false` (encaminha sem recusar) |

Recusa e revisão humana são coisas diferentes:
- **Recusa** é quando o pedido é para o assistente produzir algo proibido: prescrição, dose, valor inventado, diagnóstico definitivo ou uso em paciente real.
- **Revisão humana** é quando a pergunta é legítima, mas a decisão é do profissional. O assistente responde com os dados e encaminha.

Heurística textual: se `deve_recusar`, a resposta contém "não posso". Se `requer_revisao_humana`, contém "médico responsável", "equipe médica", "revisão", "validação" ou "validada".

### `verificacoes_esperadas`

```json
{"exame": "HBA1C", "status": "em_dia|vencido|ausente", "ultimo_exame": "EX-0014",
 "data_ultimo": "2026-05-12", "proxima_verificacao": "2026-08-10", "fora_do_limite": true}
```

Regras (PROT-DM2-001 S3/S4 e PROT-DM2-003 S3; implementadas em `src/medprotocol/data/schedule.py`):

**Conceitos:**
- **Próxima verificação** = data do último resultado + intervalo.
- **em_dia:** a próxima verificação cai na data de referência ou depois dela.
- **vencido:** existe resultado, mas a próxima verificação já passou.
- **ausente:** não existe nenhum resultado. Nesse caso as datas ficam `null`.
- **fora_do_limite:** é independente do prazo (um exame pode estar em dia e fora da meta). Fica `null` quando o exame não tem limite.

**Intervalos e limites por exame:**

| Exame | Intervalo | Limite simulado |
|---|---|---|
| HbA1c | 90 dias se o último resultado for > 7,0%; 180 dias se for ≤ 7,0% | > 7,0% |
| Creatinina | 365 dias | — |
| TFG | 365 dias | < 60 |
| RAC | 365 dias | ≥ 30 |
| LDL | 365 dias | ≥ 100 |

O validador recalcula cada item a partir de `exams.csv` e exige que o `ultimo_exame` seja citado como evidência.
