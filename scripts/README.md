# Scripts de preparação

| Script | Função |
|---|---|
| `validate_dataset.py` | Valida o dataset sintético (estrutura, evidências, regras temporais, segurança, dados pessoais, vazamento treino/avaliação). Código de saída 1 se houver erro. |
| `build_database.py` | Valida o dataset e gera `data/processed/medprotocol.db` de forma atômica e reproduzível. |
| `query_patient.py PAC-NNNN` | Imprime o resumo estruturado do paciente em JSON (aceita apenas o ID). |
| `build_index.py` | Reconstrói o índice vetorial dos protocolos em `data/vectorstore/` (extra `rag`). |
| `search_protocols.py "pergunta"` | Mostra os trechos recuperados, referências, scores e status (sem LLM). |
| `evaluate_retrieval.py` | Hit@1, Hit@3, MRR e calibração de limiares no conjunto de desenvolvimento (nunca no `qa_eval`). |

Previsto: montar datasets de fine-tuning a partir de `data/finetune/`.

## Convenções

- Execução: `python scripts/<nome>.py` a partir da raiz do repositório.
- Lógica reutilizável fica em `src/medprotocol/`; o script apenas orquestra.
- Idempotentes: rodar duas vezes produz o mesmo resultado.
- Configuração via variáveis de ambiente (`medprotocol.config`), nunca segredos no código.
