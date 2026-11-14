"""Preparação de dados para fine-tuning supervisionado (formato de mensagens com contexto).

Camadas:
- sources:  fonte dos exemplos de treino (somente qa_train.jsonl) e proteção dos dados reservados;
- context:  montagem do contexto que o sistema teria na inferência (RAG + SQLite);
- messages: formatação das mensagens (system/user/assistant) para o modelo;
- sft:      composição das três camadas em registros de treino, com relatório de cobertura.

Etapa 4 (treino; os módulos com torch/peft importam essas bibliotecas só quando usados):
- split:          separação treino/validação sem vazamento por paciente;
- chat_format:    template de chat do modelo e máscara da perda na resposta;
- train_config:   hiperparâmetros versionados (JSON);
- lora_trainer:   laço de treino LoRA;
- generation:     geração gulosa com e sem o adaptador;
- answer_metrics: métricas heurísticas de comportamento (base × ajustado).
"""
