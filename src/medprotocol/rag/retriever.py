"""Recuperação de trechos de protocolos para uma pergunta (sem gerar respostas).

O score é a similaridade de cosseno entre o vetor da pergunta e o do chunk: uma medida de
proximidade vetorial, NÃO uma probabilidade de o trecho estar correto. O status indica
apenas em que faixa o melhor score caiu em relação ao que foi observado no conjunto de
desenvolvimento (ver docs/RAG.md):

- "ok": melhor score >= upper (acima de todas as perguntas fora do escopo observadas);
- "sem_resultado_confiavel": melhor score < lower (abaixo de todas as perguntas do escopo);
- "incerto": entre os dois (faixa em que escopo e fora do escopo se sobrepõem).

Mesmo com status "ok", o primeiro resultado pode estar errado; os hits seguintes e as
referências devem ser verificados pela etapa seguinte.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.vectorstores import InMemoryVectorStore

DEFAULT_K = 4
MAX_K = 20
MAX_QUERY_CHARS = 1000

# Calibrados exclusivamente no conjunto de desenvolvimento (data/rag_dev/retrieval_dev.jsonl)
# com scripts/evaluate_retrieval.py, para o modelo/revisão e chunking atuais. Valores e
# análise em docs/RAG.md. Houve sobreposição entre escopo e fora do escopo:
#   LOWER = menor melhor-score dentro do escopo (DEV-004)
#   UPPER = maior melhor-score fora do escopo (DEV-105)
# Mudar modelo, revisão, prefixos ou chunking exige recalibrar.
LOWER_THRESHOLD = 0.8433
UPPER_THRESHOLD = 0.8709

STATUS_OK = "ok"
STATUS_UNCERTAIN = "incerto"
STATUS_NO_RELIABLE = "sem_resultado_confiavel"


@dataclass(frozen=True)
class RetrievalHit:
    ref: str
    protocol_id: str
    section_id: str
    section_title: str
    versao: str
    status_protocolo: str
    source: str
    chunk_index: int
    trecho: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    pergunta: str
    status: str
    hits: list[RetrievalHit]
    lower_threshold: float
    upper_threshold: float


def classify(top_score: float | None, lower: float, upper: float) -> str:
    if top_score is None or top_score < lower:
        return STATUS_NO_RELIABLE
    if top_score >= upper:
        return STATUS_OK
    return STATUS_UNCERTAIN


class ProtocolRetriever:
    def __init__(
        self,
        store: InMemoryVectorStore,
        lower_threshold: float = LOWER_THRESHOLD,
        upper_threshold: float = UPPER_THRESHOLD,
    ) -> None:
        if lower_threshold > upper_threshold:
            raise ValueError("lower_threshold não pode ser maior que upper_threshold")
        self.store = store
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def retrieve(self, pergunta: str, k: int = DEFAULT_K) -> RetrievalResult:
        if not isinstance(pergunta, str) or not pergunta.strip():
            raise ValueError("pergunta vazia")
        if len(pergunta) > MAX_QUERY_CHARS:
            raise ValueError(f"pergunta excede {MAX_QUERY_CHARS} caracteres")
        if not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise ValueError(f"k deve ser inteiro entre 1 e {MAX_K}")

        results = self.store.similarity_search_with_score(pergunta.strip(), k=k)
        hits = [
            RetrievalHit(
                ref=doc.metadata["ref"],
                protocol_id=doc.metadata["protocol_id"],
                section_id=doc.metadata["section_id"],
                section_title=doc.metadata["section_title"],
                versao=doc.metadata["versao"],
                status_protocolo=doc.metadata["status"],
                source=doc.metadata["source"],
                chunk_index=doc.metadata["chunk_index"],
                trecho=doc.page_content,
                score=float(score),
            )
            for doc, score in results
        ]
        top = hits[0].score if hits else None
        return RetrievalResult(
            pergunta=pergunta.strip(),
            status=classify(top, self.lower_threshold, self.upper_threshold),
            hits=hits,
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
        )
