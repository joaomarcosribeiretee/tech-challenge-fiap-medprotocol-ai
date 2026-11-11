"""Política determinística de seções adicionais ao contexto (complementa a busca semântica).

`select_rule_sections` é uma função pura de informações disponíveis na inferência:
- a pergunta (protocolos citados nela e intenção identificável por palavras-chave);
- o resumo do paciente devolvido pelo repositório (verificações, pendências, limites, consultas);
- os trechos recuperados pelo retriever (protocolo do 1º trecho, quando não há outro foco).

Nunca recebe a resposta esperada nem rótulos de treino. Cada seção selecionada carrega os
motivos (regras) que a incluíram. Regras, na ordem em que são aplicadas:

1. regras_das_verificacoes: há dados de paciente -> seções de origem das regras de prazo e de
   limite usadas por `schedule` (STATUS_SECTION, INTERVAL_SECTIONS, LIMIT_SECTIONS).
2. protocolo_vinculado:<ID>: consultas do paciente registram o protocolo -> seções operacionais
   desse protocolo (LINKED_SECTIONS).
3. revisao:<motivo>: exame fora do limite -> REV do protocolo que define o limite; pendências ou
   inconsistências -> REV do protocolo que define a situação dos exames.
4. intencao:<nome>: a pergunta pede conduta (medicamento, dose, diagnóstico...), estimativa de
   valor ou uso fora da simulação -> REV/LIM dos protocolos em foco, ou a regra de valores
   ausentes.

Protocolos em foco = citados na pergunta + vinculados ao paciente; se vazio, o protocolo do 1º
trecho recuperado. A política não remove nem reordena os trechos recuperados.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from medprotocol.data import schedule
from medprotocol.rag.retriever import RetrievalHit

PROTOCOL_ID_RE = re.compile(r"\bPROT-DM2-\d{3}\b")

# Seções operacionais de cada protocolo, incluídas quando uma consulta do paciente o vincula.
LINKED_SECTIONS = {
    "PROT-DM2-001": ("S3", "S4"),
    "PROT-DM2-002": ("S2", "S3", "S4", "S5"),
    "PROT-DM2-003": ("S3", "S4", "S5"),
}

# Intenções identificáveis na pergunta (texto minúsculo e sem acentos).
INTENT_PATTERNS = {
    "pedido_de_conduta": re.compile(
        r"prescrev|prescric|\bdose|medicament|insulina|metformina|suspend|reduzir|ajust|intensific|"
        r"tratamento|diagnostic|estagio|classifi"
    ),
    "estimativa_de_valor": re.compile(r"\bestim|invent|preench|\binferi"),
    "uso_fora_da_simulacao": re.compile(r"paciente real|pacientes reais|diretriz oficial|\boficial\b"),
}


@dataclass(frozen=True)
class RuleSelection:
    ref: str
    motivos: tuple[str, ...]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _protocol(ref: str) -> str:
    return ref.split("#", 1)[0]


def detect_intents(pergunta: str) -> list[str]:
    text = _normalize(pergunta)
    return [name for name, pattern in INTENT_PATTERNS.items() if pattern.search(text)]


def linked_protocols(paciente: dict | None) -> list[str]:
    if not paciente:
        return []
    return sorted({c["protocol_id"] for c in paciente.get("consultas", []) if c.get("protocol_id") in LINKED_SECTIONS})


def focus_protocols(pergunta: str, paciente: dict | None, hits: list[RetrievalHit]) -> list[str]:
    mentioned = set(PROTOCOL_ID_RE.findall(pergunta)) & set(LINKED_SECTIONS)
    focus = sorted(mentioned | set(linked_protocols(paciente)))
    if not focus and hits:
        focus = [hits[0].protocol_id]
    return focus


def select_rule_sections(pergunta: str, paciente: dict | None, hits: list[RetrievalHit]) -> list[RuleSelection]:
    """Seções adicionais, sem duplicatas, na ordem em que a primeira regra as selecionou."""
    selected: dict[str, list[str]] = {}

    def add(ref: str, motivo: str) -> None:
        motivos = selected.setdefault(ref, [])
        if motivo not in motivos:
            motivos.append(motivo)

    if paciente is not None:
        for ref in [schedule.STATUS_SECTION, *schedule.INTERVAL_SECTIONS.values(), *schedule.LIMIT_SECTIONS.values()]:
            add(ref, "regras_das_verificacoes")
        for protocol_id in linked_protocols(paciente):
            for section_id in LINKED_SECTIONS[protocol_id]:
                add(f"{protocol_id}#{section_id}", f"protocolo_vinculado:{protocol_id}")
        for exam in paciente.get("fora_do_limite", []):
            if exam in schedule.LIMIT_SECTIONS:
                add(f"{_protocol(schedule.LIMIT_SECTIONS[exam])}#REV", f"revisao:fora_do_limite:{exam}")
        if paciente.get("pendencias"):
            add(f"{_protocol(schedule.STATUS_SECTION)}#REV", "revisao:pendencias")
        if paciente.get("inconsistencias"):
            add(f"{_protocol(schedule.STATUS_SECTION)}#REV", "revisao:inconsistencias")

    intents = detect_intents(pergunta)
    focus = focus_protocols(pergunta, paciente, hits)
    if "pedido_de_conduta" in intents:
        for protocol_id in focus:
            add(f"{protocol_id}#REV", "intencao:pedido_de_conduta")
            add(f"{protocol_id}#LIM", "intencao:pedido_de_conduta")
    if "estimativa_de_valor" in intents:
        add(schedule.STATUS_SECTION, "intencao:estimativa_de_valor")
    if "uso_fora_da_simulacao" in intents:
        for protocol_id in focus:
            add(f"{protocol_id}#LIM", "intencao:uso_fora_da_simulacao")

    return [RuleSelection(ref, tuple(motivos)) for ref, motivos in selected.items()]
