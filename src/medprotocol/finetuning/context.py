"""Montagem do contexto do assistente: o mesmo código serve ao treino e à inferência.

O contexto contém apenas o que o sistema real teria no momento da pergunta:
- a pergunta e a data de referência;
- os trechos recuperados pelo retriever da Etapa 3 (com referência e status), usando só a
  pergunta como consulta;
- o resumo do paciente devolvido por `repository.get_patient_summary` (Etapa 2), quando a
  pergunta cita um paciente. Prazos e pendências vêm de `schedule`, sem regras duplicadas aqui.

`excluded_refs` remove seções do resultado como se elas não estivessem no índice: o status é
recalculado com `classify` sobre o melhor trecho restante. No treino, é usado para manter as
seções reservadas para a avaliação fora da entrada, inclusive como distratores; na inferência,
fica vazio.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from datetime import date

from medprotocol.data import schedule
from medprotocol.db.repository import get_patient_summary
from medprotocol.rag.retriever import DEFAULT_K, MAX_K, ProtocolRetriever, RetrievalResult, classify


@dataclass(frozen=True)
class AssistantContext:
    pergunta: str
    data_referencia: date
    recuperacao: RetrievalResult
    patient_id: str | None
    paciente: dict | None  # resumo de get_patient_summary; None quando a pergunta não cita paciente


def retrieve_excluding(
    retriever: ProtocolRetriever, pergunta: str, k: int = DEFAULT_K, excluded_refs: frozenset[str] = frozenset()
) -> RetrievalResult:
    if not excluded_refs:
        return retriever.retrieve(pergunta, k=k)
    result = retriever.retrieve(pergunta, k=min(MAX_K, k + len(excluded_refs)))
    hits = [hit for hit in result.hits if hit.ref not in excluded_refs][:k]
    top = hits[0].score if hits else None
    return replace(result, hits=hits, status=classify(top, result.lower_threshold, result.upper_threshold))


def build_context(
    pergunta: str,
    retriever: ProtocolRetriever,
    conn: sqlite3.Connection | None,
    patient_id: str | None = None,
    *,
    reference_date: date = schedule.REFERENCE_DATE,
    k: int = DEFAULT_K,
    excluded_refs: frozenset[str] = frozenset(),
) -> AssistantContext:
    """Erros do repositório (paciente inexistente, banco ausente) são propagados, nunca ocultados."""
    recuperacao = retrieve_excluding(retriever, pergunta, k, excluded_refs)
    paciente = None
    if patient_id is not None:
        if conn is None:
            raise ValueError("pergunta cita paciente, mas nenhuma conexão com o banco foi fornecida")
        paciente = get_patient_summary(conn, patient_id, reference_date)
    return AssistantContext(pergunta.strip(), reference_date, recuperacao, patient_id, paciente)
