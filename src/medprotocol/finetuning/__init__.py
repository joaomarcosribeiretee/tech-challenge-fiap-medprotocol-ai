"""Preparação de dados para fine-tuning supervisionado (formato de mensagens com contexto).

Camadas:
- sources:  fonte dos exemplos de treino (somente qa_train.jsonl) e proteção dos dados reservados;
- context:  montagem do contexto que o sistema teria na inferência (RAG + SQLite);
- messages: formatação das mensagens (system/user/assistant) para o modelo;
- sft:      composição das três camadas em registros de treino, com relatório de cobertura.
"""
