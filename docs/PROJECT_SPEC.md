# MedProtocol AI — Especificação do Projeto

## 1. Objetivo

Desenvolver um assistente de apoio à consulta de protocolos hospitalares, utilizando fine-tuning de LLM, LangChain, LangGraph e dados estruturados.

O sistema é um protótipo educacional para o Tech Challenge da Fase 3 da FIAP PosTech IA para Devs.

## 2. Escopo clínico

Acompanhamento de pacientes com diabetes mellitus tipo 2 em um hospital fictício.

O assistente deverá:
- Responder perguntas sobre protocolos institucionais simulados.
- Consultar dados sintéticos de pacientes e exames.
- Identificar informações ausentes e verificações pendentes.
- Apresentar referências dos protocolos utilizados.
- Encaminhar recomendações para validação humana.
- Registrar eventos relevantes para auditoria.

O assistente não poderá:
- Realizar diagnósticos definitivos.
- Prescrever medicamentos ou tratamentos de forma autônoma.
- Substituir avaliação profissional.
- Inventar resultados de exames ou dados do paciente.
- Apresentar protocolos fictícios como diretrizes médicas oficiais.

## 3. Requisitos obrigatórios da FIAP

### Fine-tuning
- Selecionar um modelo de linguagem adequado aos recursos disponíveis.
- Preparar dados de protocolos, FAQs e exemplos de documentos internos.
- Implementar preprocessing, curadoria e anonimização.
- Executar fine-tuning real e documentar o processo.
- Comparar o modelo-base com o modelo ajustado.

### LangChain
- Integrar o modelo ajustado.
- Implementar recuperação de protocolos com fontes.
- Consultar banco de dados estruturado.
- Incluir informações atualizadas do paciente no contexto.

### LangGraph
- Implementar um fluxo explícito de execução.
- Separar consulta documental, consulta ao banco, geração e validação.
- Tratar erros e ausência de dados.

### Segurança
- Impedir prescrições autônomas.
- Exigir revisão humana para sugestões clínicas.
- Implementar logs de execução.
- Indicar fontes utilizadas.
- Evitar exposição desnecessária de dados.

### Engenharia
- Código Python modularizado.
- Testes automatizados.
- README com instruções de execução.
- Configurações por variáveis de ambiente.

## 4. Arquitetura inicial

Interface → LangGraph → LangChain/RAG + SQLite → LLM ajustada → Validação → Resposta com fontes e logs.

Tecnologias inicialmente propostas:
- Python
- LangChain
- LangGraph
- SQLite
- Chroma ou FAISS
- Transformers, PEFT e LoRA/QLoRA
- Streamlit
- Pytest

As bibliotecas e versões serão confirmadas durante a implementação.

## 5. Estratégia de dados

Criar dados exclusivamente sintéticos:
- Protocolos hospitalares fictícios, com identificação e versão.
- FAQs de profissionais de saúde.
- Exemplos fictícios de laudos, receitas e procedimentos.
- Pacientes e exames simulados.
- Exemplos de perguntas e respostas para fine-tuning.
- Casos de avaliação separados dos dados de treinamento.

Documentos de caráter clínico devem ser revisados quanto à consistência e não devem ser tratados como orientação médica validada.

## 6. Avaliação

Avaliar:
- Qualidade das respostas.
- Fidelidade às fontes.
- Correção da recuperação de documentos.
- Uso correto dos dados estruturados.
- Recusa ou encaminhamento de solicitações inadequadas.
- Comportamento quando faltam informações.
- Diferenças entre modelo-base e modelo ajustado.

## 7. Entregáveis

- Repositório Git com código modular.
- Pipeline executável de fine-tuning.
- Integração LangChain.
- Fluxos LangGraph.
- Dataset sintético e documentação.
- Relatório técnico com avaliação.
- Diagrama da arquitetura.
- README completo.
- Vídeo demonstrativo de até 15 minutos.

## 8. Critério de conclusão

O projeto estará concluído quando os requisitos obrigatórios estiverem implementados, executados, testados, documentados e demonstráveis, com evidências dos resultados.

## 9. Fora do escopo

- Integração com hospitais reais.
- Uso de prontuários verdadeiros.
- Implantação para uso clínico.
- Prescrição automatizada.
- Aplicativo mobile.
- Infraestrutura complexa de produção.