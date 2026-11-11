"""Montagem do contexto do assistente: o mesmo código serve ao treino e à inferência.

O contexto contém apenas o que o sistema real teria no momento da pergunta:
- a pergunta e a data de referência;
- os trechos recuperados pelo retriever da Etapa 3 (com referência e status), usando só a
  pergunta como consulta;
- o resumo do paciente devolvido por `repository.get_patient_summary` (Etapa 2), quando a
  pergunta cita um paciente. Prazos e pendências vêm de `schedule`, sem regras duplicadas aqui;
- seções adicionadas pela política determinística de `context_policy`, separadas dos trechos
  recuperados e sem duplicá-los. O status da recuperação continua sendo o do retriever: as
  seções por regra não o alteram.

`excluded_refs` remove seções do contexto (recuperadas ou por regra) como se não estivessem no
índice: o status é recalculado com `classify` sobre o melhor trecho restante. No treino, é usado
para manter as seções reservadas para a avaliação fora da entrada, inclusive como distratores;
na inferência, fica vazio.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, replace
from datetime import date

from medprotocol.data import schedule
from medprotocol.db.repository import get_patient_summary
from medprotocol.finetuning.context_policy import select_rule_sections
from medprotocol.rag.retriever import DEFAULT_K, MAX_K, ProtocolRetriever, RetrievalResult, classify


@dataclass(frozen=True)
class RuleSection:
    """Seção incluída por regra determinística (sem score: não veio da busca semântica)."""

    ref: str
    protocol_id: str
    section_id: str
    section_title: str
    versao: str
    status_protocolo: str
    source: str
    chunk_index: int
    trecho: str
    motivos: tuple[str, ...]


@dataclass(frozen=True)
class AssistantContext:
    pergunta: str
    data_referencia: date
    recuperacao: RetrievalResult
    patient_id: str | None
    paciente: dict | None  # resumo de get_patient_summary; None quando a pergunta não cita paciente
    secoes_por_regra: tuple[RuleSection, ...] = ()
    # ref recuperada por similaridade que uma regra também exigiria -> motivos (não é duplicada)
    regras_em_recuperados: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def refs(self) -> list[str]:
        """Todas as referências no contexto: recuperadas, depois por regra."""
        return [hit.ref for hit in self.recuperacao.hits] + [s.ref for s in self.secoes_por_regra]


def retrieve_excluding(
    retriever: ProtocolRetriever, pergunta: str, k: int = DEFAULT_K, excluded_refs: frozenset[str] = frozenset()
) -> RetrievalResult:
    if not excluded_refs:
        return retriever.retrieve(pergunta, k=k)
    result = retriever.retrieve(pergunta, k=min(MAX_K, k + len(excluded_refs)))
    hits = [hit for hit in result.hits if hit.ref not in excluded_refs][:k]
    top = hits[0].score if hits else None
    return replace(result, hits=hits, status=classify(top, result.lower_threshold, result.upper_threshold))


def lookup_sections(retriever: ProtocolRetriever, ref: str, motivos: tuple[str, ...]) -> list[RuleSection]:
    """Chunks de uma seção no índice, na ordem de chunk_index. Seção inexistente é erro."""
    entries = sorted(
        (entry for entry in retriever.store.store.values() if entry["metadata"]["ref"] == ref),
        key=lambda entry: entry["metadata"]["chunk_index"],
    )
    if not entries:
        raise KeyError(f"seção {ref} exigida pela política de contexto não existe no índice")
    return [
        RuleSection(
            ref=ref,
            protocol_id=e["metadata"]["protocol_id"],
            section_id=e["metadata"]["section_id"],
            section_title=e["metadata"]["section_title"],
            versao=e["metadata"]["versao"],
            status_protocolo=e["metadata"]["status"],
            source=e["metadata"]["source"],
            chunk_index=e["metadata"]["chunk_index"],
            trecho=e["text"],
            motivos=motivos,
        )
        for e in entries
    ]


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

    retrieved = {hit.ref for hit in recuperacao.hits}
    sections: list[RuleSection] = []
    already_retrieved: dict[str, tuple[str, ...]] = {}
    for selection in select_rule_sections(pergunta, paciente, recuperacao.hits):
        if selection.ref in excluded_refs:
            continue
        if selection.ref in retrieved:
            already_retrieved[selection.ref] = selection.motivos
            continue
        sections += lookup_sections(retriever, selection.ref, selection.motivos)
    return AssistantContext(
        pergunta.strip(), reference_date, recuperacao, patient_id, paciente, tuple(sections), already_retrieved
    )
