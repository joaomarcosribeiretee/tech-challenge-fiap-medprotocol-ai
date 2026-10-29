"""Leitura dos arquivos do dataset sintético.

Formatos:
- Protocolos: Markdown com cabeçalho `chave: valor` entre linhas `---` e seções `## [ID] Título`.
- Pacientes, consultas e exames: CSV UTF-8 com cabeçalho fixo.
- Perguntas/respostas: JSONL (um objeto JSON por linha).

Erros de formato geram `DatasetFormatError` com arquivo e linha, para que o validador
possa reportá-los sem exibir um traceback.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SECTION_RE = re.compile(r"^## \[(?P<id>[A-Z0-9]+)\]\s+(?P<title>.+)$")

PATIENT_FIELDS = (
    "patient_id",
    "sexo",
    "ano_nascimento",
    "ano_diagnostico_dm2",
    "comorbidades",
    "medicamentos_em_uso",
    "alergias",
)
CONSULTATION_FIELDS = ("consultation_id", "patient_id", "data", "tipo", "protocol_id", "resumo")
EXAM_FIELDS = ("exam_id", "patient_id", "tipo", "valor", "unidade", "data_coleta")

PROTOCOLS_DIR = "protocols"
PATIENTS_FILE = "synthetic/patients.csv"
CONSULTATIONS_FILE = "synthetic/consultations.csv"
EXAMS_FILE = "synthetic/exams.csv"
QA_FILES = {"train": "finetune/qa_train.jsonl", "eval": "eval/qa_eval.jsonl"}


class DatasetFormatError(ValueError):
    """Arquivo do dataset com formato inválido."""


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    metadata: dict[str, str]
    sections: dict[str, str]
    body: str
    path: Path
    section_titles: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Dataset:
    protocols: dict[str, Protocol]
    patients: dict[str, dict[str, str]]
    consultations: dict[str, dict[str, str]]
    exams: dict[str, dict[str, str]]
    qa: dict[str, list[dict]]  # split ("train"/"eval") -> registros


def parse_protocol(path: Path) -> Protocol:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise DatasetFormatError(f"{path.name}: o arquivo deve começar com o cabeçalho '---'")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise DatasetFormatError(f"{path.name}: cabeçalho '---' não foi fechado") from None

    metadata: dict[str, str] = {}
    for number, line in enumerate(lines[1:end], start=2):
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise DatasetFormatError(f"{path.name}:{number}: linha de cabeçalho inválida: {line!r}")
        metadata[key.strip()] = value.strip()

    sections: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    current: str | None = None
    for number, line in enumerate(lines[end + 1 :], start=end + 2):
        match = SECTION_RE.match(line)
        if match:
            current = match["id"]
            if current in sections:
                raise DatasetFormatError(f"{path.name}:{number}: seção [{current}] duplicada")
            sections[current] = []
            titles[current] = match["title"].strip()
        elif line.startswith("## "):
            raise DatasetFormatError(f"{path.name}:{number}: seção sem ID no formato '## [ID] Título'")
        elif current is not None:
            sections[current].append(line)

    return Protocol(
        protocol_id=metadata.get("protocol_id", ""),
        metadata=metadata,
        sections={key: "\n".join(value).strip() for key, value in sections.items()},
        body="\n".join(lines[end + 1 :]),
        path=path,
        section_titles=titles,
    )


def load_csv(path: Path, fields: tuple[str, ...], id_field: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise DatasetFormatError(
                f"{path.name}: cabeçalho {reader.fieldnames} difere do schema {list(fields)}"
            )
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise DatasetFormatError(f"{path.name}:{reader.line_num}: número de colunas incorreto")
            key = row[id_field]
            if key in rows:
                raise DatasetFormatError(f"{path.name}:{reader.line_num}: ID duplicado {key}")
            rows[key] = row
    return rows


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetFormatError(f"{path.name}:{number}: JSON inválido ({exc.msg})") from None
        if not isinstance(record, dict):
            raise DatasetFormatError(f"{path.name}:{number}: cada linha deve ser um objeto JSON")
        records.append(record)
    return records


def load_dataset(data_dir: Path) -> Dataset:
    data_dir = Path(data_dir)
    required = [data_dir / PATIENTS_FILE, data_dir / CONSULTATIONS_FILE, data_dir / EXAMS_FILE]
    required += [data_dir / name for name in QA_FILES.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise DatasetFormatError(f"arquivos ausentes: {missing}")

    protocols: dict[str, Protocol] = {}
    for path in sorted((data_dir / PROTOCOLS_DIR).glob("*.md")):
        protocol = parse_protocol(path)
        if protocol.protocol_id in protocols:
            raise DatasetFormatError(f"{path.name}: protocol_id duplicado {protocol.protocol_id}")
        protocols[protocol.protocol_id] = protocol

    return Dataset(
        protocols=protocols,
        patients=load_csv(data_dir / PATIENTS_FILE, PATIENT_FIELDS, "patient_id"),
        consultations=load_csv(data_dir / CONSULTATIONS_FILE, CONSULTATION_FIELDS, "consultation_id"),
        exams=load_csv(data_dir / EXAMS_FILE, EXAM_FIELDS, "exam_id"),
        qa={split: load_jsonl(data_dir / name) for split, name in QA_FILES.items()},
    )


def exam_text(exam: dict[str, str]) -> str:
    """Representação canônica de um exame, usada como trecho de evidência."""
    return f"{exam['tipo']} {exam['valor']} {exam['unidade']} ({exam['data_coleta']})"


def patient_lines(patient: dict[str, str]) -> list[str]:
    """Linhas canônicas `campo: valor` de um paciente, usadas como trecho de evidência."""
    return [f"{key}: {value}" for key, value in patient.items()]
