"""Fonte dos exemplos de treino para o fine-tuning.

Lê exclusivamente `finetune/qa_train.jsonl`: o conjunto de avaliação (`eval/qa_eval.jsonl`)
nunca é aberto aqui. Cada exemplo é separado em:
- entrada: o que o sistema real teria na inferência (pergunta e paciente citado nela);
- alvo: a resposta esperada;
- rótulos: campos usados só para conferência e relatórios, nunca para montar a entrada.

Dados reservados para a avaliação final são rejeitados com erro explícito:
- pacientes exclusivos da avaliação (PAC-0009, PAC-0010), em qualquer campo;
- seções reservadas dos protocolos (PROT-DM2-002#S4, PROT-DM2-003#S4).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from medprotocol.data.loaders import QA_FILES
from medprotocol.data.validation import EVAL_ONLY_PATIENTS

TRAIN_FILE = QA_FILES["train"]
RESERVED_SECTIONS = frozenset({"PROT-DM2-002#S4", "PROT-DM2-003#S4"})
PATIENT_ID_RE = re.compile(r"\bPAC-\d{4}\b")


class ReservedDataError(ValueError):
    """Um exemplo de treino usa dados reservados para a avaliação final."""


@dataclass(frozen=True)
class TrainingExample:
    qa_id: str
    pergunta: str
    patient_id: str | None
    resposta_esperada: str
    rotulos: dict  # categoria, protocolos_citados, flags... (nunca entram na entrada do modelo)


def _check_reserved(record: dict) -> None:
    qa_id = record.get("qa_id", "?")
    reserved_patients = sorted(EVAL_ONLY_PATIENTS & set(PATIENT_ID_RE.findall(json.dumps(record, ensure_ascii=False))))
    if reserved_patients:
        raise ReservedDataError(f"{qa_id}: usa pacientes reservados para a avaliação {reserved_patients}")
    cited = set(record.get("protocolos_citados", [])) | {
        ev["ref"] for ev in record.get("evidencias", []) if ev.get("tipo") == "protocolo"
    }
    if cited & RESERVED_SECTIONS:
        raise ReservedDataError(f"{qa_id}: depende de seções reservadas {sorted(cited & RESERVED_SECTIONS)}")


def to_example(record: dict) -> TrainingExample:
    _check_reserved(record)
    patient_id = record.get("patient_id")
    if patient_id is not None and patient_id not in PATIENT_ID_RE.findall(record["pergunta"]):
        # Na inferência, o sistema só sabe qual é o paciente se ele estiver na pergunta.
        raise ValueError(f"{record['qa_id']}: patient_id {patient_id} não aparece na pergunta")
    labels = {k: v for k, v in record.items() if k not in ("qa_id", "pergunta", "patient_id", "resposta_esperada")}
    return TrainingExample(record["qa_id"], record["pergunta"], patient_id, record["resposta_esperada"], labels)


def load_training_examples(data_dir: Path) -> list[TrainingExample]:
    path = Path(data_dir) / TRAIN_FILE
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [to_example(record) for record in records]
