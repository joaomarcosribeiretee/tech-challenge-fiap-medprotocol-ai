# Avaliação final — artefatos

Gerados por `scripts/evaluate_final.py`. Análise em [`docs/AVALIACAO_FINAL.md`](../../docs/AVALIACAO_FINAL.md).

| Arquivo | Conteúdo | Quando |
|---|---|---|
| `protocolo.json` | Sistema congelado (commit, modelo, hashes), checagens de vazamento e definição das métricas | `--freeze`, commitado em `94c2d36` **antes** da execução |
| `execucao.json` | Horário, commit, confirmação de mesmo contexto e perda nas respostas esperadas | `--run` (única) |
| `resultados_por_exemplo.jsonl` | 20 linhas (10 perguntas × base/ajustado): texto bruto do modelo, saída do sistema, métricas | `--run` |
| `audit.jsonl` | 20 eventos de auditoria do fluxo (hashes, sem texto) | `--run` |
| `leitura_cega.jsonl` | As duas respostas de cada pergunta como A/B, em ordem aleatória (semente 20261116) | `--run` |
| `gabarito_cego.json` | Correspondência A/B → variante, aberta só depois da leitura | `--run` |
| `leitura_tecnica.json` | Vereditos e justificativas da leitura cega (revisão técnica por IA, não clínica) | manual, depois do `--run` |
| `resumo.json` | Agregados dos três níveis por variante | `--report` |
