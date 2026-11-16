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

## Arquitetura

Detalhes do grafo, validação e status: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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
| Fine-tuning | Transformers, PEFT (LoRA bf16), PyTorch CUDA; Qwen2.5-1.5B-Instruct | 4 |
| Orquestração | LangGraph (estado explícito, validação determinística, auditoria) | 5 |
| Interface | Streamlit | 6 |

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

> Nota: o fine-tuning usa um segundo ambiente (`.venv-train`) com PyTorch CUDA; ver a seção
> "Fine-tuning LoRA" abaixo. Não foi preciso Python 3.11/3.12: há rodas CUDA para 3.14.

## Configuração

Variáveis de ambiente documentadas em [`.env.example`](.env.example) e lidas por
`medprotocol.config.load_settings()`. O arquivo `.env` nunca é versionado.

## Estrutura

```
AGENTS.md            instruções para agentes de desenvolvimento
docs/                especificação e documentação técnica
src/medprotocol/     código do pacote (config, data, db, rag, finetuning, graph, ui)
app/                 interface Streamlit (streamlit_app.py)
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
python -m pip install -e ".[dev,rag]"                            # ou com -c constraints-rag.txt (versões testadas)
python scripts/build_index.py                                     # índice em data/vectorstore/ (não versionado)
python scripts/search_protocols.py "Qual é a periodicidade da HbA1c?" -k 3
python scripts/evaluate_retrieval.py                              # métricas no conjunto de desenvolvimento
```

Modelo, chunking, significado dos scores, métricas e limitações: [docs/RAG.md](docs/RAG.md).

## Preparar dados para fine-tuning (sem treinar)

```powershell
python scripts/build_sft_dataset.py       # mensagens com contexto RAG + regras + SQLite -> data/processed/sft/train.jsonl
```

Formato, uso em treino e inferência, isolamento da avaliação e cobertura das referências:
[docs/FINETUNING_DATA.md](docs/FINETUNING_DATA.md).

## Fine-tuning LoRA e comparação base × ajustado

Usa um ambiente **separado** (`.venv-train`), com torch CUDA, para não alterar o ambiente do RAG.
Requer GPU NVIDIA (testado numa RTX 4070 de 12 GB) e baixa ~3,1 GB de pesos do Qwen2.5-1.5B-Instruct
(Apache-2.0) no primeiro uso.

```powershell
python scripts/split_sft_dataset.py               # (.venv) manifesto treino/validação, determinístico
py -3.14 -m venv .venv-train
.venv-train\Scripts\python -m pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv-train\Scripts\python -m pip install -e ".[dev,train]" -c constraints-train.txt
.venv-train\Scripts\python scripts/train_lora.py --config configs/finetune/qwen2.5-1.5b-lora-v1.json --smoke
.venv-train\Scripts\python scripts/train_lora.py --config configs/finetune/qwen2.5-1.5b-lora-v1.json
.venv-train\Scripts\python scripts/compare_models.py --config configs/finetune/qwen2.5-1.5b-lora-v1.json
```

## Perguntar ao assistente (fluxo LangGraph)

Executa no `.venv-train`, que tem RAG, LangGraph, torch CUDA e PEFT. A recuperação nesse
ambiente é idêntica à do `.venv`; ver [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```powershell
.venv-train\Scripts\python -m pip install -e ".[dev,train,rag,graph]" -c constraints-train.txt
.venv-train\Scripts\python scripts/ask.py "Quando está prevista a próxima HbA1c de PAC-0001?"
.venv-train\Scripts\python scripts/ask.py --variant base "..."            # modelo-base (adaptador desligado)
.venv-train\Scripts\python scripts/ask.py --file scripts/demo/perguntas.txt --out reports/graph/demo_ajustado.jsonl
```

A saída é JSON com `status` (`respondido`, `evidencia_insuficiente`, `revisao_humana` ou
`erro`), `resposta`, `fontes`, `alertas` e `audit_id`. A auditoria vai para `logs/audit.jsonl`,
sem o texto da pergunta nem o da resposta. **Passar nas verificações não significa segurança clínica.**

## Interface Streamlit

Também roda no `.venv-train`, a partir da raiz do repositório. Pré-requisitos: banco
(`build_database.py`), índice (`build_index.py`), adaptador (`train_lora.py`) e GPU CUDA.

```powershell
.venv-train\Scripts\python -m pip install -e ".[dev,train,rag,graph,ui]" -c constraints-train.txt
.venv-train\Scripts\python -m streamlit run app/streamlit_app.py      # abre em http://localhost:8501
```

A primeira carga leva cerca de 12 s: monta o fluxo e carrega o modelo uma única vez. Cada
pergunta leva de 0 a 15 s. A tela tem:
- campo de pergunta e seletor ajustado/base;
- status colorido;
- resposta ou, se retida, a mensagem do sistema, com o texto original numa área separada
  "NÃO VALIDADO";
- alertas, fontes e ID de auditoria;
- aviso educacional fixo.

Sem GPU ou sem adaptador, a tela avisa e só as perguntas que dispensam o modelo funcionam.
Detalhes e limitações: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §8.

```powershell
.venv-train\Scripts\python scripts/ui_e2e.py --variants ajustado base --out reports/ui/e2e_streamlit.jsonl   # teste real da tela
```

O adaptador fica em `models/adapters/` (não versionado). Os relatórios de treino e comparação
ficam em `reports/finetune/`. Escolha do modelo, hiperparâmetros, resultados e limitações:
[docs/FINETUNING.md](docs/FINETUNING.md).

## Roadmap

- [x] **Etapa 0** — Fundação do projeto (estrutura, configuração, testes)
- [x] **Etapa 1** — Dataset sintético v1: 3 protocolos, 10 pacientes, 30 perguntas/respostas
  (ver [data/DATASET_CARD.md](data/DATASET_CARD.md))
- [x] **Etapa 2** — Persistência em SQLite e funções de consulta (ver [docs/DATABASE.md](docs/DATABASE.md))
- [x] **Etapa 3** — Recuperação de protocolos com LangChain, sem LLM (revisada; métricas, limitações
  e riscos de segurança em [docs/RAG.md](docs/RAG.md))
- [ ] Fine-tuning (LoRA/QLoRA) e comparação base × ajustado
  - [x] Revisão do treino v1 (3 correções) e formato contextualizado das mensagens
    (ver [docs/FINETUNING_DATA.md](docs/FINETUNING_DATA.md))
  - [x] Cobertura das referências: política determinística de seções adicionais (20/20 cobertos)
  - [x] Lote piloto 1 consolidado: 41 exemplos novos (61 no treino; 7 categorias), 6 pacientes
    sintéticos novos, contexto pré-registrado com a política v1 congelada. Cobertura original
    no pré-registro: 36 completas, 4 parciais, 1 insuficiente; 41/41 após correção das respostas
    e substituição de 6 exemplos (não é medida independente da política). Revisão técnica,
    **sem revisão clínica** (ver [data/DATASET_CARD.md](data/DATASET_CARD.md))
  - [x] Separação treino/validação: 48/13, sem paciente nos dois lados (`data/finetune/split_v1.json`)
  - [x] Treino LoRA real (Qwen2.5-1.5B-Instruct, RTX 4070, 3,6 min) e comparação base × ajustado
    na validação. A perda caiu de 1,30 para 0,88; o ajustado aprendeu formato, recusa e
    encaminhamento, mas **não ficou mais correto** na leitura das respostas (ver [docs/FINETUNING.md](docs/FINETUNING.md))
  - [ ] Iteração de dados/treino com base nos erros observados e avaliação final no `qa_eval`
- [x] **Etapa 5** — Fluxo LangGraph: paciente (SQLite) → protocolos (RAG) → política v1 → Qwen2.5 + LoRA →
  validação determinística → auditoria, com saída estruturada (status, fontes, alertas, ID de auditoria).
  Testado com LLM simulado (25 testes) e com o modelo real em 8 perguntas. As regras retêm ou
  encaminham a maioria dos erros, mas **não todos** (ver [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))
- [x] **Etapa 6** — Interface Streamlit sobre o fluxo da Etapa 5. Tem seletor ajustado/base, status,
  fontes, alertas, ID de auditoria e área separada para o texto retido. Validada com 19 testes
  (assistente simulado) e de ponta a ponta com o modelo real: 16 perguntas, resultados idênticos aos
  de `ask.py` (ver [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §8)
- [ ] Avaliação, relatório técnico e vídeo
