# Recuperação de protocolos (RAG, Etapa 3)

Busca semântica de trechos dos protocolos **simulados** em `data/protocols/`, com referência
verificável (`PROT-DM2-001#S3`). Esta etapa **não gera respostas**: só recupera trechos literais.
Dados de pacientes continuam exclusivamente no SQLite; o índice não contém nenhum registro de paciente.

## Instalação

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,rag]"
```

Versões instaladas e testadas (Python 3.14.4, Windows, só CPU):
langchain-core 1.7.1 · langchain-huggingface 1.2.3 · sentence-transformers 6.2.0 ·
transformers 5.21.1 · torch 2.15.0 · numpy 2.5.4.

**Recursos:**
- Cerca de 1 GB a mais no ambiente virtual, quase tudo torch.
- 470 MB do modelo, baixado no primeiro uso para `~/.cache/huggingface/hub`.
- Sem GPU e sem chave de API.

**Tempo medido nesta máquina:**
- Primeira carga do modelo: cerca de 24 s, incluindo o download.
- Indexação completa: cerca de 9 s.

## Decisões

| Item | Escolha | Motivo |
|---|---|---|
| Embeddings | `intfloat/multilingual-e5-small`, revisão `614241f622f53c4eeff9890bdc4f31cfecc418b3` (MIT) | Treinado para busca, multilíngue (inclui português), 118 M de parâmetros, 384 dimensões, roda em CPU |
| Prefixos E5 | `passage: ` na indexação e `query: ` nas perguntas, via `prompt` do sentence-transformers | O modelo foi treinado assim. A revisão fixada não define prompts padrão, então não há prefixo duplicado |
| Índice | `InMemoryVectorStore` (langchain-core), similaridade de cosseno com vetores normalizados | São 20 chunks: busca exata e sem dependência nativa. Migrar para FAISS ou Chroma é trocar a classe |
| Chunking | 1 chunk por seção: título do protocolo, `[ID]`, título da seção e corpo literal | Nunca mistura seções. O título ajuda seções curtas, como `[S4] Pés` |
| Limite de tamanho | 512 tokens, medidos com o tokenizer do modelo e com o prefixo | Seção maior é dividida entre itens, mantendo o `section_id`. Item que sozinho excede o limite gera erro, nunca truncamento silencioso. Hoje nenhuma seção chega perto do limite |

**Metadados de cada chunk:** `ref`, `protocol_id`, `section_id`, `section_title`, `titulo`,
`versao`, `status`, `natureza_regras`, `validacao_clinica`, `source`, `chunk_index`, `chunk_count`.

## Construir ou reconstruir o índice

```powershell
python scripts/build_index.py            # sempre reconstrói a partir dos .md versionados
```

O índice vai para `VECTORSTORE_DIR` (padrão `data/vectorstore/`, **não versionado**) como
`index.json` e `manifest.json`. O cache é só uma otimização. Ele só é reutilizado quando:
- o manifesto é idêntico ao atual: hash SHA-256 de cada protocolo, modelo, revisão, prefixos, limite e versão do chunking; **e**
- o conteúdo e os metadados dos chunks em cache são iguais aos do chunking atual.

Cache corrompido, adulterado ou desatualizado é reconstruído automaticamente. Também dá para
apagar `data/vectorstore/` a qualquer momento.

## Consultar (sem LLM)

```powershell
python scripts/search_protocols.py "Qual é a periodicidade da HbA1c?" -k 3
```

Em Python:

```python
from medprotocol.rag.pipeline import load_protocol_retriever

retriever, _ = load_protocol_retriever("data", "data/vectorstore")
result = retriever.retrieve("Qual é a periodicidade da HbA1c?", k=4)
result.status   # "ok" | "incerto" | "sem_resultado_confiavel"
for hit in result.hits:
    hit.ref, hit.score, hit.trecho
```

### O que significa o score
É a **similaridade de cosseno** entre o vetor da pergunta e o do trecho, uma medida de
proximidade vetorial. **Não é probabilidade de acerto.** Com este modelo, os scores ficam
comprimidos entre cerca de 0,75 e 0,92, e mesmo perguntas sem relação com os protocolos passam de 0,75.

### O que significa o status
O status compara o melhor score com o que foi observado no conjunto de desenvolvimento:

| Status | Condição | Leitura |
|---|---|---|
| `ok` | score ≥ 0,8709 | Acima de todas as perguntas fora do escopo observadas. **Não garante que o 1º trecho esteja certo** (veja DEV-010 abaixo) |
| `incerto` | 0,8433 ≤ score < 0,8709 | Faixa em que perguntas dentro e fora do escopo se sobrepuseram |
| `sem_resultado_confiavel` | score < 0,8433 | Abaixo de todas as perguntas do escopo observadas |

Os hits são sempre devolvidos, inclusive nos status `incerto` e `sem_resultado_confiavel`, para
depuração. Quem consumir o resultado (a futura etapa com LangGraph) deve decidir o que fazer com o
status, e não ignorá-lo.

## Avaliação (somente conjunto de desenvolvimento)

```powershell
python scripts/evaluate_retrieval.py
```

O conjunto `data/rag_dev/retrieval_dev.jsonl` tem 22 perguntas dentro do escopo, cobrindo 18 das 20
seções, e 8 fora do escopo. Foi escrito para esta etapa e é **independente** dos conjuntos de
fine-tuning e avaliação:

- **`qa_eval.jsonl` não foi usado** para avaliar nem para ajustar nada. Ele fica **preservado para a avaliação integrada final** do sistema completo.
  - Um teste compara apenas o **texto** das perguntas de desenvolvimento com as perguntas reservadas: a maior similaridade com as de avaliação é 0,316, e o teste exige menos de 0,6. Esse teste não faz nenhuma busca com perguntas de avaliação.
- **`PROT-DM2-002#S4` e `PROT-DM2-003#S4` ficaram de fora** do conjunto de desenvolvimento, porque qualquer pergunta sobre elas seria próxima, por tema, dos casos reservados QA-0021 e QA-0022.
- As três demonstrações da etapa tratam desses temas. Por isso são só demonstrações e testes de aceitação, e não orientaram nenhum ajuste.

### Resultados reais

| Métrica | Valor | Contagem |
|---|---|---|
| Hit@1 | 0,864 | 19/22 |
| Hit@3 | 0,955 | 21/22 |
| MRR | 0,914 | — |
| Perguntas fora do escopo com status diferente de `ok` | 1,000 | 8/8 (**no próprio conjunto de calibração**, por isso otimista) |
| Perguntas do escopo com status `ok` | 0,818 | 18/22 |

**Calibração.** Os limiares saíram do mesmo conjunto:
- menor melhor-score dentro do escopo: 0,8434 (DEV-004);
- maior melhor-score fora do escopo: 0,8709 (DEV-105).

**Não há separação**: as faixas se sobrepõem, por isso existe o status `incerto`. A taxa de rejeição
é medida nos mesmos dados usados para calibrar, então é uma estimativa otimista. Não há conjunto
separado para estimá-la sem viés.

### Erros observados

| Pergunta | O que aconteceu | Provável causa |
|---|---|---|
| DEV-004 "alvo de hemoglobina glicada" | Seção certa (`001#S3`) só em 9º; 1º foi `002#S1` | O protocolo usa a sigla "HbA1c" e o modelo não liga a sigla ao nome por extenso. É a pergunta do escopo de menor score e define o limiar inferior |
| DEV-001 "finalidade do protocolo ambulatorial" | `001#LIM` em 1º, `S1` em 2º | Perguntas sobre "para que serve" ficam próximas de "limites de utilização" |
| DEV-010 "para que serve o protocolo de hipoglicemia" | `002#LIM` em 1º com status **ok** | Mesmo padrão; mostra que `ok` não garante acerto |
| DEV-105 "vacinas para adultos com diabetes" (fora do escopo) | Score 0,8709, acima de 18 perguntas do escopo | Vocabulário do domínio ("diabetes") aproxima perguntas não cobertas pelos protocolos |
| Demo 3 "não consegue deglutir" | Certo (`002#S4`, 0,8835), mas `002#S3` ("**capaz** de deglutir") vem logo atrás com 0,8724 | Embeddings lidam mal com negação: a seção oposta fica quase empatada |

## Testes

```powershell
python -m pytest tests/test_rag_chunking.py tests/test_rag_index.py   # rápidos, sem o modelo
python -m pytest tests/test_rag_retrieval.py                          # modelo real (pulado sem o extra [rag])
```

**O que os testes cobrem:**
- **Chunking:** 1 chunk por seção real; metadados completos e coerentes; trecho igual ao texto literal; nenhuma mistura de seções; nenhum ID de paciente; divisão sem perda nem truncamento; erro quando um item sozinho excede o limite.
- **Cache:** reutilização só com manifesto e conteúdo idênticos; reconstrução quando há mudança, corrupção ou adulteração.
- **Prefixos E5:** o vetor de cada pergunta e de cada trecho indexado é igual ao do texto com **um** prefixo, e diferente do texto sem prefixo, com prefixo duplicado ou com o prefixo trocado.
- **Recuperação:** referências existentes, scores ordenados, as três demonstrações entre os 3 primeiros, perguntas sem relação sem status `ok`, entradas inválidas rejeitadas.
- **Calibração:** relata sobreposição em vez de inventar separação, e o arredondamento mantém cada pergunta do lado certo do limiar.

## Limitações

- **Os protocolos recuperados são simulados e não têm validação clínica.** Todos estão com `status: SIMULACAO`, `natureza_regras: simulada` e `validacao_clinica: nao_realizada`, e esses metadados acompanham cada trecho recuperado. Recuperar um trecho não o torna orientação médica.
- **`ok` não garante acerto.** No DEV-010, o status foi `ok` com a seção errada em 1º lugar. O status só indica a faixa do score.
- **Sinônimos e siglas:** "hemoglobina glicada" não encontrou a seção que usa a sigla "HbA1c" (DEV-004, 9º lugar).
- **Negação:** seções com sentido oposto ficam com scores quase iguais. "Incapaz de deglutir" (`002#S4`, 0,8835) e "capaz de deglutir" (`002#S3`, 0,8724) ficaram a 0,011 de distância.
- **Perguntas fora do escopo:** quando usam o vocabulário do domínio, recebem scores altos (DEV-105, vacinas: 0,8709). O score sozinho não separa escopo de fora do escopo; os limiares só indicam faixas.
- O corpus é minúsculo (20 chunks). As métricas de 22 e 8 perguntas têm alta variância, e um erro muda o Hit@1 em cerca de 4,5 pontos.
- Os limiares valem só para este modelo, revisão, prefixos e chunking. Mudar qualquer um exige rodar `evaluate_retrieval.py` de novo e recalibrar.
- A busca é só semântica: siglas e termos exatos ("HbA1c" × "hemoglobina glicada") podem falhar. Uma busca híbrida, combinando BM25 com a semântica, ou a expansão de siglas são melhorias possíveis, ainda não implementadas.
- Não há reranking: o 1º resultado pode estar errado, e a etapa seguinte deve considerar os k trechos e o status.
- O modelo trunca acima de 512 tokens. O chunking impede que isso aconteça com os protocolos, mas uma pergunta muito longa seria truncada (perguntas ficam limitadas a 1000 caracteres).
