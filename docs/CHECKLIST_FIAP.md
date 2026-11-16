# Checklist dos requisitos — Tech Challenge Fase 3 (FIAP)

**Fonte:** enunciado oficial "8IADT – Fase 3 – Tech challenge" (PDF de 5 páginas). Ele **não está
no repositório**: foi lido de uma cópia local do desenvolvedor em 16/11/2026. Cada item abaixo
transcreve o requisito do PDF, e não a especificação interna (`docs/PROJECT_SPEC.md`).

**Legenda:** ✅ atendido · 🟡 parcial · ⬜ pendente (depende do desenvolvedor)

## Entregas técnicas

### 1. Fine-tuning de LLM com dados médicos internos

| Requisito (PDF) | Status | Implementação | Evidência verificável |
|---|---|---|---|
| Realizar o fine-tuning de um modelo LLM | ✅ | LoRA real em Qwen2.5-1.5B-Instruct (RTX 4070, 214 s, 3 épocas) | `scripts/train_lora.py`, `src/medprotocol/finetuning/lora_trainer.py`, `reports/finetune/qwen2.5-1.5b-lora-v1/train_report.json`, `docs/FINETUNING.md` |
| … utilizando protocolos médicos do hospital | ✅ | 3 protocolos simulados entram no contexto de cada exemplo (RAG + política de seções) | `data/protocols/`, `data/processed/sft/train.jsonl` (gerado), `docs/FINETUNING_DATA.md` |
| … exemplos de perguntas frequentes feitas por médicos | ✅ | 61 perguntas/respostas em 7 categorias | `data/finetune/qa_train.jsonl`, `data/DATASET_CARD.md` |
| … modelos de laudos, receitas e procedimentos internos | 🟡 | **Procedimentos:** sim, como as condutas de hipoglicemia (PROT-DM2-002) e a periodicidade de exames. **Laudos e receitas: não há modelos no dataset.** Receitas foram tratadas só como pedidos a recusar (prescrição fica fora do escopo por segurança) | `data/protocols/PROT-DM2-002.md`; ausência confirmada por busca no repositório |
| Preparar os dados com preprocessing, anonimização e curadoria | ✅ | **Preprocessing:** chunking por seção, mensagens SFT com o mesmo contexto da inferência e máscara de perda. **Anonimização:** dados fabricados do zero, minimização (sem nome/CPF; só ano de nascimento) e varredura de padrões de PII. **Curadoria:** validador, revisão de 61 exemplos, pré-registro de contexto e split sem paciente nos dois lados | `scripts/validate_dataset.py`, `src/medprotocol/data/validation.py`, `data/DATASET_CARD.md`, `data/finetune/piloto1/`, `data/finetune/split_v1.json` |

### 2. Criação de assistente médico com LangChain

| Requisito (PDF) | Status | Implementação | Evidência verificável |
|---|---|---|---|
| Usar LangChain para construir um pipeline que integre a LLM customizada | 🟡 | O pipeline é um **LangGraph** (biblioteca do projeto LangChain) que chama o Qwen2.5 + LoRA no nó `gerar_resposta`. **Ressalva:** o modelo é chamado via Transformers/PEFT (`LocalQwenGenerator`), sem estar encapsulado como `ChatModel`/`Runnable` do LangChain | `src/medprotocol/graph/workflow.py`, `src/medprotocol/graph/llm.py` |
| … realizar consultas em base de dados estruturada | 🟡 | O nó `consultar_paciente` do LangGraph consulta o SQLite (pacientes, consultas, exames) em modo somente leitura. **Ressalva:** a consulta usa `sqlite3`, não um componente SQL do LangChain | `src/medprotocol/db/repository.py`, `docs/DATABASE.md`, `tests/test_db.py` |
| … contextualizar as respostas com informações atualizadas do paciente | ✅ | Resumo do paciente na data de referência (último exame, histórico, verificações pendentes, fora da meta) entra no prompt | `src/medprotocol/finetuning/context.py`, `messages.py`; demonstração em `reports/ui/e2e_streamlit.jsonl` |
| (Contexto do desafio) recuperação de protocolos internos | ✅ | LangChain `InMemoryVectorStore` + `HuggingFaceEmbeddings` (e5-small), com referência PROTOCOLO#SEÇÃO | `src/medprotocol/rag/`, `docs/RAG.md` (Hit@1 0,864; Hit@3 0,955 no conjunto de desenvolvimento) |

### 3. Segurança e validação

| Requisito (PDF) | Status | Implementação | Evidência verificável |
|---|---|---|---|
| Limites de atuação: nunca prescrever diretamente, sem validação humana | ✅ com falha conhecida | Prompt de sistema; detector de pedido de conduta; regras que retêm dose, sugestão de conduta e recusa ausente; status `revisao_humana`. **Falha observada na avaliação final:** um "Não suspenda a insulina NPH" não foi retido; foi exibido como pendente de revisão | `src/medprotocol/graph/validation.py`, `docs/ARCHITECTURE.md` §4, `docs/AVALIACAO_FINAL.md` §5 |
| Logging detalhado para rastreamento e auditoria | ✅ | Evento JSONL por pergunta: ID, modelo e versão, paciente, fontes, alertas, status, tempos por nó; texto só como hash (minimização) | `src/medprotocol/graph/audit.py`, `reports/ui/audit_e2e.jsonl`, `reports/eval_final/audit.jsonl` |
| Explainability: indicar a fonte usada na resposta | ✅ | Cada resposta lista os trechos do contexto, com versão, origem (similaridade ou regra) e marca de citação; a interface mostra a tabela de fontes e os alertas | `AssistantResponse.fontes` em `workflow.py`, `app/streamlit_app.py` |

### 4. Organização do código

| Requisito (PDF) | Status | Implementação | Evidência verificável |
|---|---|---|---|
| Projeto modularizado em Python | ✅ | Pacote `medprotocol` com `data`, `db`, `rag`, `finetuning`, `graph`, `ui` e `evaluation`; scripts só orquestram | `src/medprotocol/`, `scripts/README.md` |
| Instruções completas no README | ✅ | Instalação dos dois ambientes, dados, índice, treino, fluxo, interface, avaliação e testes | `README.md` |
| (Adicional) testes automatizados | ✅ | 244 testes pytest | `tests/`, comando no README |

## Entregáveis da Fase 3

| Entregável (PDF) | Status | Onde |
|---|---|---|
| Código-fonte: pipeline de fine-tuning | ✅ | `scripts/build_sft_dataset.py`, `split_sft_dataset.py`, `train_lora.py`, `compare_models.py` |
| Código-fonte: integração com LangChain | 🟡 | RAG com LangChain ✅; a LLM e o SQLite entram pelo LangGraph, não como componentes LangChain (ver item 2) |
| Código-fonte: fluxos do LangGraph | ✅ | `src/medprotocol/graph/workflow.py` (7 nós, arestas condicionais) |
| Dataset anonimizado ou exemplo de dados sintéticos | ✅ | `data/` + `data/DATASET_CARD.md` |
| Relatório: explicação do processo de fine-tuning | ✅ | `docs/RELATORIO_TECNICO.md` §5 e `docs/FINETUNING.md` |
| Relatório: descrição do assistente médico | ✅ | `docs/RELATORIO_TECNICO.md` §2 e §6–7 |
| Relatório: diagrama do fluxo LangChain | ✅ | `docs/arquitetura.svg` / `docs/arquitetura.png` |
| Relatório: avaliação do modelo e análise dos resultados | ✅ | `docs/AVALIACAO_FINAL.md`, `docs/RELATORIO_TECNICO.md` §8 |
| Vídeo de até 15 min: treinamento e funcionamento da LLM | ⬜ | Roteiro pronto em `docs/ROTEIRO_VIDEO.md`; gravação pelo desenvolvedor |
| Vídeo: execução de um fluxo automatizado | ⬜ | idem |
| Vídeo: resposta a perguntas clínicas contextualizadas | ⬜ | idem |
| Vídeo: logs e validação das respostas | ⬜ | idem |

## Outras observações do PDF

- **Trabalho "a princípio em grupo":** o repositório tem um único autor. Confirmar com a turma se
  isso é aceito.
- **Datasets sugeridos (PubMedQA, MedQuAD):** são sugestão, não obrigação. O projeto usa dados
  sintéticos próprios em português, alinhados aos protocolos simulados. Não houve comparação com
  essas bases.
- **"Sugerir tratamentos" (texto do desafio):** foi deliberadamente **não** implementado. O
  assistente recusa prescrição e encaminha ao médico, em linha com o requisito 3 ("nunca
  prescrever diretamente, sem validação humana"). Os "alertas para a equipe médica" existem como
  alertas e gatilhos de revisão (`resultado_fora_do_limite`, `dados_inconsistentes` etc.).

## Resumo

- **Atendidos:** fine-tuning real, dados sintéticos com curadoria, contexto do paciente, RAG com
  fontes, LangGraph, limites de atuação, auditoria, explainability, modularização, README,
  relatório, diagrama e avaliação.
- **Parciais (3):**
  - modelos de laudos e receitas ausentes no dataset;
  - a LLM customizada não está encapsulada como componente LangChain;
  - a consulta SQL é feita sem componente LangChain.
- **Pendente (1):** o vídeo.

**Conformidade não é de 100%.** Os itens parciais são decisões do desenvolvedor: aceitar e
documentar, ou implementar antes da entrega. Ver "Pendências" no README.
