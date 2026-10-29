# Dados

> **Todos os dados deste diretório são sintéticos e fictícios.** Não representam pacientes reais
> nem diretrizes médicas oficiais. Não devem ser usados como orientação clínica.

- [SCHEMAS.md](SCHEMAS.md): schemas dos três conjuntos e regras verificadas.
- [DATASET_CARD.md](DATASET_CARD.md): como os dados foram criados, limitações, anonimização e vazamento.

## Estrutura

| Pasta | Conteúdo | Formato | Versionado? |
|---|---|---|---|
| `protocols/` | Protocolos institucionais simulados (DM2) | Markdown | Sim |
| `synthetic/` | Pacientes, consultas e exames sintéticos | CSV | Sim |
| `finetune/` | Perguntas/respostas de **treino** | JSONL | Sim |
| `eval/` | Perguntas/respostas de **avaliação**, separadas do treino | JSONL | Sim |
| `processed/` | Saídas derivadas (ex.: SQLite) | — | Não (gerado por scripts) |
| `vectorstore/` | Índices vetoriais (Chroma/FAISS) | — | Não (gerado por scripts) |

Regra: versionar a **fonte** pequena e revisável; ignorar o que pode ser **regenerado**.

## Convenções

- Todo protocolo tem `status: SIMULACAO`, `natureza_regras`, `validacao_clinica: nao_realizada`
  e seções identificadas (`PROT-DM2-001#S3`).
- Nenhum caso de `eval/` pode aparecer em `finetune/`. PAC-0009 e PAC-0010 são exclusivos da avaliação.
- Identificadores são artificiais (`PAC-0001`, `CON-0001`, `EX-0001`, `QA-0001`).
- Data de referência fixa: 2026-06-30.

Validação: `python scripts/validate_dataset.py`.
