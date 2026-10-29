# Dataset card — MedProtocol AI, dataset sintético v1

> **Status: amostra inicial aguardando revisão humana.** Não ampliar antes da revisão de qualidade.

## Conteúdo

| Conjunto | Arquivo(s) | Registros |
|---|---|---|
| Protocolos simulados | `protocols/PROT-DM2-001..003.md` | 3 |
| Pacientes | `synthetic/patients.csv` | 10 |
| Consultas | `synthetic/consultations.csv` | 10 |
| Exames | `synthetic/exams.csv` | 56 |
| Perguntas/respostas de treino | `finetune/qa_train.jsonl` | 20 |
| Perguntas/respostas de avaliação | `eval/qa_eval.jsonl` | 10 |

Cada categoria tem 4 casos de treino e 2 de avaliação: `consulta_protocolo`, `dados_ausentes`,
`contextualizada`, `recusa` e `revisao_humana`. Data de referência: **2026-06-30**. Schemas em [SCHEMAS.md](SCHEMAS.md).

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

## Limitações

- **Volume:** 30 casos servem para validar formato e qualidade, não para o fine-tuning final.
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
