# MedProtocol AI

Assistente educacional de apoio à consulta de protocolos hospitalares, desenvolvido para o
**Tech Challenge Fase 3 — FIAP PosTech IA para Devs**.

> ⚠️ Protótipo educacional. Todos os protocolos, pacientes e exames são **fictícios**.
> O sistema não realiza diagnósticos, não prescreve e não substitui avaliação profissional.

## Objetivo

Apoiar profissionais na consulta de protocolos institucionais simulados de acompanhamento de
pacientes com **diabetes mellitus tipo 2** em um hospital fictício, combinando um LLM ajustado
(fine-tuning), recuperação de documentos com fontes (LangChain/RAG), dados estruturados (SQLite)
e um fluxo explícito com validação (LangGraph). Especificação completa: [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md).

## Escopo

O assistente deverá: responder sobre protocolos simulados, consultar dados sintéticos de
pacientes e exames, apontar informações ausentes, citar fontes, encaminhar recomendações para
validação humana e registrar eventos de auditoria.

Fora do escopo: hospitais ou prontuários reais, uso clínico, prescrição automatizada, app mobile,
infraestrutura complexa de produção.

## Arquitetura prevista

```
Interface → LangGraph → LangChain/RAG + SQLite → LLM ajustada → Validação → Resposta com fontes e logs
```

## Tecnologias propostas

| Área | Tecnologia | Etapa em que entra |
|---|---|---|
| Linguagem | Python ≥ 3.11 | 0 |
| Testes | pytest | 0 |
| Dados estruturados | SQLite | Dados sintéticos |
| Recuperação | LangChain (`InMemoryVectorStore`) + `multilingual-e5-small` local | 3 |
| Fine-tuning | Transformers, PEFT, LoRA/QLoRA | Fine-tuning |
| Orquestração | LangGraph | Fluxos |
| Interface | Streamlit | Interface |

As bibliotecas são adicionadas ao `pyproject.toml` somente na etapa que as utiliza, com versão documentada.

## Preparar o ambiente

Pré-requisito: Python 3.11 ou superior.

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
copy .env.example .env
python -m pytest
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
python -m pytest
```

> Nota: o fine-tuning (PyTorch, bitsandbytes) pode exigir uma versão específica de Python
> (ex.: 3.11/3.12) ou execução em GPU (ex.: Google Colab). Isso será definido na etapa de fine-tuning.

## Configuração

Variáveis de ambiente documentadas em [`.env.example`](.env.example) e lidas por
`medprotocol.config.load_settings()`. O arquivo `.env` nunca é versionado.

## Estrutura

```
AGENTS.md            instruções para agentes de desenvolvimento
docs/                especificação e documentação técnica
src/medprotocol/     código do pacote (config, data, db, rag; futuros módulos: finetuning, graph, ui)
tests/               testes pytest
data/                dados sintéticos (ver data/README.md, SCHEMAS.md e DATASET_CARD.md)
scripts/             scripts de preparação de dados (ver scripts/README.md)
```

## Validar o dataset e gerar o banco

```powershell
python scripts/validate_dataset.py
python scripts/build_database.py          # gera data/processed/medprotocol.db (não versionado)
python scripts/query_patient.py PAC-0001  # resumo estruturado do paciente em JSON
```

Detalhes do schema, das funções de consulta e do tratamento de erros: [docs/DATABASE.md](docs/DATABASE.md).

## Buscar trechos dos protocolos (RAG, sem LLM)

Requer o extra opcional `rag` (embeddings locais; ~1 GB de dependências + modelo de 470 MB no primeiro uso):

```powershell
python -m pip install -e ".[dev,rag]"
python scripts/build_index.py                                     # índice em data/vectorstore/ (não versionado)
python scripts/search_protocols.py "Qual é a periodicidade da HbA1c?" -k 3
python scripts/evaluate_retrieval.py                              # métricas no conjunto de desenvolvimento
```

Modelo, chunking, significado dos scores, métricas e limitações: [docs/RAG.md](docs/RAG.md).

## Roadmap

- [x] **Etapa 0** — Fundação do projeto (estrutura, configuração, testes)
- [x] **Etapa 1** — Dataset sintético v1: 3 protocolos, 10 pacientes, 30 perguntas/respostas
  (ver [data/DATASET_CARD.md](data/DATASET_CARD.md))
- [x] **Etapa 2** — Persistência em SQLite e funções de consulta (ver [docs/DATABASE.md](docs/DATABASE.md))
- [ ] **Etapa 3** — Recuperação de protocolos com LangChain, sem LLM (implementada; **aguardando revisão**,
  ver [docs/RAG.md](docs/RAG.md))
- [ ] Fine-tuning (LoRA/QLoRA) e comparação base × ajustado
- [ ] Fluxos LangGraph com validação e auditoria
- [ ] Interface Streamlit
- [ ] Avaliação, relatório técnico e vídeo
