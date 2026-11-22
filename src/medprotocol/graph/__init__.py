"""Fluxo LangGraph do assistente (Etapa 5): estado explícito, validação determinística e auditoria.

- workflow:   grafo, saída estruturada (`AssistantResponse`) e fachada `MedProtocolAssistant`;
- validation: verificações determinísticas da pergunta e da resposta;
- llm:        Qwen2.5-1.5B local com/sem adaptador LoRA (um único modelo carregado);
- audit:      log JSONL sem texto de pergunta/resposta.
"""
