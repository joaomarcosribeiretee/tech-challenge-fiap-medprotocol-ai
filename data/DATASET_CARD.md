# Dataset card — MedProtocol AI, dataset sintético v1

> **Status: v1 aprovado como semente após revisão humana assistida** (3 exemplos corrigidos;
> veja "Revisão do treino"). Ainda não serve para demonstrar melhoria do fine-tuning. Formato
> contextualizado, cobertura das referências e limitações em
> [docs/FINETUNING_DATA.md](../docs/FINETUNING_DATA.md).
>
> **Lote piloto 1: consolidado após revisão técnica** (41 exemplos: QA-0031..QA-0077, sem
> QA-0033/0038/0039/0044/0052/0053, que foram substituídos). A revisão foi técnica e de
> consistência com os protocolos simulados; **não houve revisão por profissional de saúde**.
> Veja "Lote piloto 1".

## Conteúdo

| Conjunto | Arquivo(s) | Registros |
|---|---|---|
| Protocolos simulados | `protocols/PROT-DM2-001..003.md` | 3 |
| Pacientes | `synthetic/patients.csv` | 16 (10 do v1 + 6 do piloto) |
| Consultas | `synthetic/consultations.csv` | 16 (10 + 6) |
| Exames | `synthetic/exams.csv` | 88 (56 + 32) |
| Perguntas/respostas de treino | `finetune/qa_train.jsonl` | 61 (20 do v1 + 41 do piloto) |
| Perguntas/respostas de avaliação | `eval/qa_eval.jsonl` | 10 (inalterado) |

No v1 (QA-0001..QA-0020), cada categoria tem 4 casos de treino e 2 de avaliação: `consulta_protocolo`,
`dados_ausentes`, `contextualizada`, `recusa` e `revisao_humana`. O lote piloto acrescenta as
categorias opcionais `contexto_insuficiente` e `selecao_de_trecho`, que existem só no treino.
Data de referência: **2026-06-30**. Schemas em [SCHEMAS.md](SCHEMAS.md).

## Como os dados foram criados

- **Tudo foi escrito do zero**, com assistência de IA (Claude) e direcionamento do desenvolvedor.
  Não houve fonte de dados reais: nenhum prontuário, planilha, base pública de pacientes ou
  documento institucional real foi usado como ponto de partida.
- **Pacientes:** são integralmente sintéticos e foram desenhados para cobrir situações de teste:
  - exame em dia;
  - exame vencido;
  - exame ausente;
  - exame fora da meta;
  - casos-limite, como HbA1c exatamente em 7,0% e próxima verificação a 1 dia da data de referência;
  - histórico com resultado anterior;
  - hipoglicemia em internação;
  - divergência entre o que foi solicitado em consulta e o que consta registrado.
- **Protocolos:** as regras são simuladas (`natureza_regras: simulada`), sem referência externa
  verificada (`referencias_externas_verificadas: nenhuma`) e sem validação clínica
  (`validacao_clinica: nao_realizada`). Alguns valores (ex.: 70 mg/dL para hipoglicemia, oferta
  de 15 g de carboidrato) foram escolhidos por serem plausíveis e comuns em material educativo,
  **mas não foram conferidos contra nenhuma fonte e não devem ser lidos como referência clínica**.
- **Perguntas/respostas:**
  - Foram redigidas manualmente.
  - Um script auxiliar temporário, fora do repositório, serviu apenas para serializar em JSONL e copiar o texto canônico dos exames.
  - Os arquivos JSONL versionados são a fonte de verdade.
  - Todas as datas e situações citadas foram conferidas pelo validador contra `exams.csv`.

## Anonimização e dados pessoais

Os pacientes não são dados reais anonimizados. São registros **fabricados do zero**, sem
correspondência com pessoas reais. Mesmo assim, o schema segue minimização de dados:
- sem nome, CPF, cartão SUS, endereço, telefone ou e-mail;
- só o ano de nascimento, nunca a data completa;
- identificadores artificiais e sequenciais (`PAC-0001`).

Verificações automáticas (no validador):
- **Cabeçalho exato dos CSVs:** uma coluna extra, como `nome`, é erro.
- **Busca por padrões** em todos os `.md`, `.csv` e `.jsonl`: CPF (com e sem máscara), cartão SUS (15 dígitos), e-mail, telefone e CEP.

**Limitação:** a busca por padrões é uma **verificação adicional, não uma prova de anonimização**.
Ela não detecta nomes próprios, combinações raras de atributos nem dados pessoais em formato
inesperado. Neste dataset a garantia vem da origem (nada foi derivado de pessoas reais), e não
da busca.

## Separação treino × avaliação (vazamento)

Verificações automáticas:
- `qa_id` únicos entre os conjuntos; o `split` de cada registro corresponde ao arquivo;
- **PAC-0009 e PAC-0010 são exclusivos da avaliação.** O treino não pode citá-los, nem os seus exames
  (`EX-0047`–`EX-0056`) ou consultas (`CON-0009`, `CON-0010`), em nenhum campo;
- nenhuma pergunta repetida, comparando texto normalizado;
- similaridade por sequência de palavras, com IDs mascarados, entre todos os pares treino × avaliação
  (pergunta e resposta): ≥ 0,90 é erro e ≥ 0,75 é aviso.

Resultado na v1: **nenhum erro ou aviso**. Maior similaridade: 0,538 (perguntas QA-0020 × QA-0027).

### Revisão manual de paráfrases e respostas equivalentes

A similaridade lexical não detecta paráfrases. Por isso, cada caso de avaliação foi comparado
manualmente com os casos de treino mais próximos:

| Avaliação | Treino mais próximo | Julgamento |
|---|---|---|
| QA-0021 (hipoglicemia sem deglutição, S4) | QA-0002 (paciente consciente, S3) | Mesmo tema, conduta diferente. A conduta da S4 não aparece no treino. A definição de hipoglicemia grave aparece como evidência secundária em QA-0010. **Sobreposição parcial, aceita.** |
| QA-0022 (exame dos pés, S4) | QA-0001 (mesmo molde "com que frequência") | Só o molde da pergunta coincide; o conteúdo é inédito. |
| QA-0023 (PAC-0010: HbA1c ausente, função renal vencida) | QA-0005, QA-0006 | Mesmas regras, paciente e dados inéditos. |
| QA-0024 (PAC-0009: pendências) | QA-0006, QA-0007 | Mesmas regras, dados inéditos. Usa uma regra que não aparece no treino (S4: em dia ≠ dentro do limite). |
| QA-0025 (PAC-0009: reavaliação solicitada × registro) | QA-0009 (alerta renal de PAC-0004) | Tema renal comum. A divergência entre consulta e registro é inédita. |
| QA-0026 (PAC-0010: cadastro + LDL) | QA-0012, QA-0019 (LDL fora do limite) | Mesma regra de LDL, paciente diferente. **Resposta estruturalmente parecida (0,439)**: aceita, pois testa a aplicação da regra. |
| QA-0027 (recusa: suspender NPH/dose de PAC-0009) | QA-0013, QA-0020 | Mesmo padrão de recusa, medicamentos e dados diferentes. |
| QA-0028 (recusa: preencher exames de PAC-0010) | QA-0014 (estimar creatinina de PAC-0003) | Mesmo padrão de recusa, paciente e exames diferentes. |
| QA-0029 (TFG × metformina, PAC-0009) | QA-0020 (redução de metformina, PAC-0001) | Mesmo padrão de encaminhamento, justificativa clínica diferente. |
| QA-0030 (exames externos não digitalizados) | — | Situação inédita. |

**Conclusão:** nenhuma resposta da avaliação é reproduzível a partir de uma resposta de treino.
Há, porém, sobreposição intencional de **regras**: dos 8 casos de avaliação com paciente, 6
usam apenas trechos de protocolo que também aparecem no treino. A avaliação mede, portanto,
sobretudo a **aplicação de regras conhecidas a pacientes inéditos** e o comportamento de
recusa/encaminhamento. Ela mede pouco a recuperação de regras novas. Isso deve constar no
relatório de avaliação.

## Revisão do treino (v1)

Os 20 exemplos de `qa_train.jsonl` foram revisados individualmente contra os protocolos e os
CSVs, sem abrir o `qa_eval.jsonl`. Resultado: 17 aprovados, 3 corrigidos e nenhum bloqueante.

| Exemplo | Problema | Correção |
|---|---|---|
| QA-0010 | Concluía que a conduta da S3 foi seguida, mas a S3 vale para paciente "consciente **e capaz de deglutir**", e o registro CON-0006 só informa "consciente" | A resposta diz que as etapas registradas são compatíveis, que a capacidade de deglutir não está registrada e não deve ser presumida, e que a confirmação cabe à equipe médica. Incluída a evidência "Repetir a glicemia capilar após 15 minutos." |
| QA-0013, QA-0017 | "Vencida desde 16/05/2026": pela regra S4, o exame ainda está em dia **na** data prevista e só vence no dia seguinte | Passou a "próxima verificação prevista para 16/05/2026; vencida na data de referência de 30/06/2026" |

O formato contextualizado para o fine-tuning e a cobertura das referências estão em
[docs/FINETUNING_DATA.md](../docs/FINETUNING_DATA.md).

## Lote piloto 1 (consolidado)

**Revisão pontual antes do lote:** QA-0019 dizia "o protocolo simulado não define critérios de alta",
mas o contexto dele não contém todas as seções. A resposta passou a dizer que "os trechos de
protocolo disponíveis não trazem critérios", sem transformar ausência de evidência em evidência de ausência.

**41 exemplos**, sem paráfrase direta do v1. Distribuição final (igual à original, porque cada
substituta mantém a categoria do exemplo que substitui):

| Categoria | Piloto | Treino total (v1 + piloto) |
|---|---|---|
| `consulta_protocolo` | 6 | 10 |
| `dados_ausentes` | 5 | 9 |
| `contextualizada` | 6 | 10 |
| `revisao_humana` | 6 | 10 |
| `recusa` (inclui fora do escopo) | 6 | 10 |
| `contexto_insuficiente` | 6 | 6 |
| `selecao_de_trecho` | 6 | 6 |
| **Total** | **41** | **61** |

No piloto, 6 exemplos têm `deve_recusar=true` e 17 têm `requer_revisao_humana=true`; 15 não citam paciente.

**Pacientes novos (PAC-0011..PAC-0016)**, desenhados para:
- vencimento exatamente na data de referência (PAC-0011);
- valores no limite: LDL 99 e 100, RAC 29, TFG 64;
- duas HbA1c no mesmo dia (PAC-0012, inconsistência sinalizada pelo repositório);
- nenhum exame monitorado registrado (PAC-0013);
- hipoglicemia de 54 mg/dL com repetição da oferta, em paciente registrado como capaz de deglutir (PAC-0014);
- RAC e LDL fora do limite com HbA1c vencida (PAC-0015);
- três HbA1c em queda (PAC-0016).

**Como foi feito:**
- As perguntas foram gravadas primeiro em `finetune/piloto1/perguntas.jsonl`.
- O contexto foi pré-registrado com a política congelada (`finetune/piloto1/contexto_preregistrado.json`) **antes** de as respostas serem escritas.
- As respostas e `verificacoes_esperadas` foram geradas por um script auxiliar fora do repositório, que lê só os CSVs e usa `schedule.verification`. O script audita cada data DD/MM/AAAA da resposta: ela precisa ser a data de referência, uma data de exame ou consulta do paciente, ou uma próxima verificação calculada.
- Na consolidação, as respostas corrigidas e as substitutas foram redigidas manualmente. Um script auxiliar fora do repositório só serializou o JSONL e copiou `verificacoes_esperadas` de `get_patient_summary`.
- O `qa_eval.jsonl` não foi aberto.
- Os temas que este card descreve como inéditos na avaliação foram evitados: divergência entre consulta e registro, exames não digitalizados, a regra "em dia ≠ dentro do limite" como trecho citado, e TFG × metformina.
- Nenhum exemplo usa PAC-0009/PAC-0010 nem as seções reservadas.

### Revisão técnica do lote (consolidação)

Revisão técnica e de consistência com os protocolos simulados, feita com assistência de IA e
direcionamento do desenvolvedor. **Não é revisão clínica**: nenhum profissional de saúde revisou o lote.
O `qa_eval.jsonl` não foi aberto; só os validadores preexistentes o leram, programaticamente.

**1. Lacunas de cobertura: respostas reescritas, perguntas e contexto inalterados.** Em cada caso a
resposta passou a se apoiar só no contexto pré-registrado. A política não foi alterada.

| Exemplo | Citava fora do contexto | Correção |
|---|---|---|
| QA-0032 | `001#S2` | Diz que o PROT-DM2-003 remete aos critérios do PROT-DM2-001, que **o texto desses critérios não está nos trechos fornecidos** e que é preciso consultá-los. Usa `003#S1` (adultos com DM2) |
| QA-0049 | `001#LIM` | Retirada a frase apoiada na LIM; a `001#REV`, que está no contexto, já sustenta o encaminhamento |
| QA-0056 | `001#LIM` | Recusa mantida. A resposta descreve o escopo dos três protocolos pelas seções S1 recuperadas e cita a `002#LIM` ("não define… doses de medicamentos") |
| QA-0064 | `003#REV` | Retirada a frase sobre diagnóstico de retinopatia. Começa por "os trechos fornecidos não trazem esse critério", e não por "não" |
| QA-0071 | `002#S3` | Declara que **a regra de notificar o médico não está nos trechos**, compara só S5 (registro/auditoria) e REV (revisão do esquema) e indica onde confirmar |

Nenhum exemplo foi excluído: em todos foi possível responder com o que estava no contexto, declarando
a insuficiência quando necessário.

**2. Diversidade e proximidade com a avaliação: 6 exemplos substituídos.** As perguntas substitutas
receberam **IDs novos** (QA-0072..QA-0077), para que o ID de cada pergunta continue ligado a um único
pré-registro. Elas foram gravadas em `finetune/piloto1/revisao/perguntas_substitutas.jsonl` e o
contexto foi pré-registrado com a mesma política v1 (`revisao/contexto_preregistrado_substitutas.json`)
**antes** de as respostas serem escritas. Os arquivos de pré-registro originais não foram alterados.

| Removido | Motivo | Substituto | Situação nova |
|---|---|---|---|
| QA-0033 | Similaridade 0,455 com a avaliação (fundo de olho) | QA-0072 | Escopo: o protocolo de hipoglicemia vale para ambulatório? |
| QA-0038 | Similaridade 0,477 (situação renal de PAC-0012, 4º exemplo dele) | QA-0073 | PAC-0013: intervalo da HbA1c indeterminável sem resultado anterior |
| QA-0039 | Similaridade 0,530 (exames vencidos de PAC-0015) | QA-0074 | PAC-0003: limite de alerta inaplicável a RAC ausente |
| QA-0044 | Mesmo molde de QA-0009 (alerta renal) | QA-0075 | PAC-0014: HbA1c na meta × hipoglicemia, sem relação nos trechos |
| QA-0052 | 3º exemplo sobre os dados renais de PAC-0004 | QA-0076 | PAC-0011: HbA1c 7,1% sem tolerância/arredondamento da meta |
| QA-0053 | Repetia as pendências de PAC-0007 (QA-0007) | QA-0077 | PAC-0015: um resultado sinalizado não dispensa sinalizar os outros |

Depois da substituição, os pares treino × avaliação mais parecidos do validador são todos ≤ 0,538;
o maior par do piloto é QA-0059 × QA-0027 (0,417, pergunta).

**3. QA-0031, QA-0051 e QA-0071 (hipoglicemia) foram mantidos**, porque ensinam comportamentos
diferentes: QA-0031 é consulta direta à S5; QA-0051 aplica S5, S3 e REV aos dados de PAC-0014; e
QA-0071, depois da correção, ensina a separar o que está no contexto (S5 e REV) do que falta (S3).

**4. Verificações:** validador sem erros nem avisos; `build_sft_dataset.py --strict` sem citações
fora do contexto; datas e IDs das respostas alteradas auditados contra os CSVs (só data de referência,
datas de exame/consulta do próprio paciente e próximas verificações calculadas); nenhuma seção
reservada e nenhum paciente reservado.

**Pendências que continuam abertas:**
- Revisão clínica por profissional de saúde (não realizada).
- A QA-0072 se apoia no título do PROT-DM2-002 que aparece no cabeçalho do trecho, porque a seção de
  objetivo (`002#S1`) não foi recuperada.
- A QA-0073 trata de HbA1c ausente, regra também usada na avaliação; o paciente e o raciocínio
  (intervalo indeterminável) são diferentes.
- Concentração de pacientes no piloto: PAC-0013 e PAC-0014 aparecem em 4 exemplos cada.

## Limitações

- **Volume:** 61 casos de treino e 10 de avaliação ainda são pouco para o fine-tuning final; servem para validar formato, qualidade e cobertura.
- **Autoria única:** os casos têm estilo e vocabulário homogêneos, o que pode inflar o desempenho do modelo ajustado nesse estilo.
- **Coerência clínica:** a coerência foi revisada quanto às regras simuladas, não por profissional de saúde. A spec (§5) pede revisão de consistência dos documentos clínicos, e ela está **pendente**.
- **Heurísticas textuais:** recusa e encaminhamento são verificados por palavras-chave, o que é necessário mas não suficiente.
- **Protocolo × consulta:** só há uma divergência modelada, em QA-0025.
- **Escopo das regras:** exame dos pés e fundo de olho existem apenas como texto de consulta, sem regra temporal calculada.
- **Idioma:** apenas português.

## Como validar

```powershell
python scripts/validate_dataset.py     # relatório + pares mais parecidos; código de saída 1 se houver erro
python -m pytest -v                     # inclui testes negativos: cada defeito inserido deve ser detectado
```
