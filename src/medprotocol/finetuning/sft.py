"""Composição: exemplo de treino -> contexto -> mensagens, com verificações e cobertura.

A resposta esperada é usada só como mensagem do assistente e para o relatório de cobertura,
nunca para escolher trechos. A cobertura compara as referências citadas no rótulo
(`protocolos_citados`) com as recuperadas: uma referência citada que não está no contexto
ensinaria o modelo a citar o que não viu, por isso é reportada (`refs_citadas_fora_do_contexto`).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from medprotocol.data.validation import EVAL_ONLY_PATIENTS
from medprotocol.finetuning.context import build_context
from medprotocol.finetuning.messages import PROMPT_VERSION, build_messages
from medprotocol.finetuning.sources import RESERVED_SECTIONS, ReservedDataError, TrainingExample
from medprotocol.rag.retriever import DEFAULT_K, ProtocolRetriever


@dataclass(frozen=True)
class SftRecord:
    messages: list[dict[str, str]]
    metadata: dict

    def to_json(self) -> dict:
        return {"messages": self.messages, "metadata": self.metadata}


def _reserved_texts(retriever: ProtocolRetriever) -> list[str]:
    return [entry["text"] for entry in retriever.store.store.values() if entry["metadata"]["ref"] in RESERVED_SECTIONS]


def _check_input(example: TrainingExample, prompt_text: str, reserved_texts: list[str]) -> None:
    if any(text in prompt_text for text in reserved_texts) or any(ref in prompt_text for ref in RESERVED_SECTIONS):
        raise ReservedDataError(f"{example.qa_id}: seção reservada na entrada")
    if any(patient in prompt_text for patient in EVAL_ONLY_PATIENTS):
        raise ReservedDataError(f"{example.qa_id}: paciente reservado na entrada")
    if example.resposta_esperada in prompt_text:
        raise ValueError(f"{example.qa_id}: resposta esperada vazou para a entrada")


def build_sft_record(
    example: TrainingExample,
    retriever: ProtocolRetriever,
    conn: sqlite3.Connection | None,
    k: int = DEFAULT_K,
    reserved_texts: list[str] | None = None,
) -> SftRecord:
    context = build_context(example.pergunta, retriever, conn, example.patient_id, k=k, excluded_refs=RESERVED_SECTIONS)
    messages = build_messages(context, example.resposta_esperada)
    prompt_text = "\n".join(m["content"] for m in messages if m["role"] != "assistant")
    _check_input(example, prompt_text, _reserved_texts(retriever) if reserved_texts is None else reserved_texts)

    retrieved = [hit.ref for hit in context.recuperacao.hits]
    cited = example.rotulos.get("protocolos_citados", [])
    return SftRecord(
        messages=messages,
        metadata={
            "qa_id": example.qa_id,
            "categoria": example.rotulos.get("categoria"),
            "prompt_version": PROMPT_VERSION,
            "k": k,
            "data_referencia": context.data_referencia.isoformat(),
            "patient_id": example.patient_id,
            "status_recuperacao": context.recuperacao.status,
            "refs_recuperadas": retrieved,
            "refs_citadas_fora_do_contexto": [ref for ref in cited if ref not in retrieved],
        },
    )


def build_sft_records(
    examples: list[TrainingExample], retriever: ProtocolRetriever, conn: sqlite3.Connection | None, k: int = DEFAULT_K
) -> list[SftRecord]:
    reserved_texts = _reserved_texts(retriever)
    return [build_sft_record(example, retriever, conn, k, reserved_texts) for example in examples]
