"""Consultas de leitura sobre o banco SQLite (sem escrita, sem SQL vindo de fora).

Garantias:
- Conexão aberta em modo somente leitura (`mode=ro`): escrita gera erro e um caminho
  inexistente não cria um banco vazio.
- Todas as consultas são constantes deste módulo, com parâmetros vinculados (`?`).
- `patient_id` e `tipo` são validados antes de chegar ao SQL.
- Regras de prazo vêm exclusivamente de `medprotocol.data.schedule`.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from medprotocol.data import schedule
from medprotocol.data.validation import EXAM_TYPES
from medprotocol.db.schema import SCHEMA_VERSION

PATIENT_ID_RE = re.compile(r"^PAC-\d{4}$")
EMPTY_LIST_MARKER = "nenhuma"


class RepositoryError(RuntimeError):
    """Erro base da camada de leitura."""


class DatabaseNotFoundError(RepositoryError):
    """O arquivo do banco não existe; gere-o com scripts/build_database.py."""


class SchemaVersionError(RepositoryError):
    """O banco foi gerado com outra versão do schema."""


class PatientNotFoundError(RepositoryError, LookupError):
    """Nenhum paciente com o ID informado."""


@dataclass(frozen=True)
class Patient:
    patient_id: str
    sexo: str
    ano_nascimento: int
    ano_diagnostico_dm2: int
    comorbidades: tuple[str, ...]
    medicamentos_em_uso: tuple[str, ...]
    alergias: tuple[str, ...]


@dataclass(frozen=True)
class Consultation:
    consultation_id: str
    patient_id: str
    data: str
    tipo: str
    protocol_id: str
    resumo: str


@dataclass(frozen=True)
class Exam:
    exam_id: str
    patient_id: str
    tipo: str
    valor: float
    unidade: str
    data_coleta: str


@contextmanager
def connect_readonly(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = Path(db_path)
    if not db_path.is_file():
        raise DatabaseNotFoundError(
            f"banco não encontrado em {db_path}; gere-o com: python scripts/build_database.py"
        )
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        try:
            row = conn.execute("SELECT value FROM build_metadata WHERE key = 'schema_version'").fetchone()
        except sqlite3.DatabaseError as exc:
            raise SchemaVersionError(f"{db_path} não é um banco gerado pelo MedProtocol AI ({exc})") from exc
        if row is None or row["value"] != SCHEMA_VERSION:
            found = None if row is None else row["value"]
            raise SchemaVersionError(
                f"banco com schema {found!r}, esperado {SCHEMA_VERSION!r}; reconstrua o banco"
            )
        yield conn
    finally:
        conn.close()


# --- Validação de entrada ---------------------------------------------------------------------


def _check_patient_id(patient_id: str) -> str:
    if not isinstance(patient_id, str) or not PATIENT_ID_RE.match(patient_id):
        raise ValueError(f"patient_id inválido: {patient_id!r} (formato esperado: PAC-NNNN)")
    return patient_id


def _check_exam_type(tipo: str) -> str:
    if tipo not in EXAM_TYPES:
        raise ValueError(f"tipo de exame inválido: {tipo!r}; aceitos: {sorted(EXAM_TYPES)}")
    return tipo


def _split_list(value: str) -> tuple[str, ...]:
    """'a;b' -> ('a', 'b'); 'nenhuma' -> () (nenhum item registrado)."""
    if value == EMPTY_LIST_MARKER:
        return ()
    return tuple(item for item in value.split(";") if item)


def _require_patient(conn: sqlite3.Connection, patient_id: str) -> None:
    exists = conn.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    if exists is None:
        raise PatientNotFoundError(f"paciente {patient_id} não encontrado")


def _exam(row: sqlite3.Row) -> Exam:
    return Exam(**dict(row))


# --- Consultas --------------------------------------------------------------------------------


def get_patient(conn: sqlite3.Connection, patient_id: str) -> Patient:
    _check_patient_id(patient_id)
    row = conn.execute(
        "SELECT patient_id, sexo, ano_nascimento, ano_diagnostico_dm2, comorbidades, "
        "medicamentos_em_uso, alergias FROM patients WHERE patient_id = ?",
        (patient_id,),
    ).fetchone()
    if row is None:
        raise PatientNotFoundError(f"paciente {patient_id} não encontrado")
    return Patient(
        patient_id=row["patient_id"],
        sexo=row["sexo"],
        ano_nascimento=row["ano_nascimento"],
        ano_diagnostico_dm2=row["ano_diagnostico_dm2"],
        comorbidades=_split_list(row["comorbidades"]),
        medicamentos_em_uso=_split_list(row["medicamentos_em_uso"]),
        alergias=_split_list(row["alergias"]),
    )


def list_consultations(conn: sqlite3.Connection, patient_id: str) -> list[Consultation]:
    """Consultas do paciente, da mais antiga para a mais recente."""
    _check_patient_id(patient_id)
    _require_patient(conn, patient_id)
    rows = conn.execute(
        "SELECT consultation_id, patient_id, data, tipo, protocol_id, resumo FROM consultations "
        "WHERE patient_id = ? ORDER BY data, consultation_id",
        (patient_id,),
    ).fetchall()
    return [Consultation(**dict(row)) for row in rows]


def list_exams(conn: sqlite3.Connection, patient_id: str, tipo: str | None = None) -> list[Exam]:
    """Exames do paciente (opcionalmente de um tipo), do mais antigo para o mais recente."""
    _check_patient_id(patient_id)
    _require_patient(conn, patient_id)
    if tipo is None:
        rows = conn.execute(
            "SELECT exam_id, patient_id, tipo, valor, unidade, data_coleta FROM exams "
            "WHERE patient_id = ? ORDER BY data_coleta, exam_id",
            (patient_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT exam_id, patient_id, tipo, valor, unidade, data_coleta FROM exams "
            "WHERE patient_id = ? AND tipo = ? ORDER BY data_coleta, exam_id",
            (patient_id, _check_exam_type(tipo)),
        ).fetchall()
    return [_exam(row) for row in rows]


def get_latest_exam(
    conn: sqlite3.Connection,
    patient_id: str,
    tipo: str,
    until: date | None = schedule.REFERENCE_DATE,
) -> Exam | None:
    """Exame mais recente do tipo até `until` (inclusive); None se não houver.

    Desempate igual ao de `schedule.verification`: maior data_coleta, depois maior exam_id.
    """
    _check_patient_id(patient_id)
    _check_exam_type(tipo)
    _require_patient(conn, patient_id)
    limit = "9999-12-31" if until is None else until.isoformat()
    row = conn.execute(
        "SELECT exam_id, patient_id, tipo, valor, unidade, data_coleta FROM exams "
        "WHERE patient_id = ? AND tipo = ? AND data_coleta <= ? "
        "ORDER BY data_coleta DESC, exam_id DESC LIMIT 1",
        (patient_id, tipo, limit),
    ).fetchone()
    return None if row is None else _exam(row)


def get_patient_summary(
    conn: sqlite3.Connection,
    patient_id: str,
    reference_date: date = schedule.REFERENCE_DATE,
) -> dict:
    """Resumo estruturado (serializável em JSON) para uso como contexto do assistente.

    `verificacoes` segue o formato de `verificacoes_esperadas` do dataset e é calculado por
    `schedule.verification`. `inconsistencias` sinaliza registros que exigem atenção, sem
    corrigi-los nem descartá-los silenciosamente.
    """
    patient = get_patient(conn, patient_id)
    consultations = list_consultations(conn, patient_id)
    exams = list_exams(conn, patient_id)
    exam_dicts = [asdict(exam) for exam in exams]

    verifications = [schedule.verification(tipo, exam_dicts, reference_date) for tipo in schedule.MONITORED_EXAMS]

    latest: dict[str, dict] = {}
    for exam in exams:  # ordenados por data; o último de cada tipo prevalece
        if date.fromisoformat(exam.data_coleta) <= reference_date:
            latest[exam.tipo] = asdict(exam)

    issues = [
        f"{exam.exam_id}: data_coleta {exam.data_coleta} posterior à data de referência; ignorado nas verificações"
        for exam in exams
        if date.fromisoformat(exam.data_coleta) > reference_date
    ]
    same_day = Counter((e.tipo, e.data_coleta) for e in exams if e.tipo in schedule.MONITORED_EXAMS)
    issues += [
        f"{tipo}: mais de um resultado em {day}; considerado o de maior exam_id"
        for (tipo, day), count in sorted(same_day.items())
        if count > 1
    ]

    return {
        "paciente": {
            **asdict(patient),
            "comorbidades": list(patient.comorbidades),
            "medicamentos_em_uso": list(patient.medicamentos_em_uso),
            "alergias": list(patient.alergias),
        },
        "data_referencia": reference_date.isoformat(),
        "consultas": [asdict(c) for c in consultations],
        "ultimos_exames": latest,
        "verificacoes": verifications,
        "pendencias": [v["exame"] for v in verifications if v["status"] in ("vencido", "ausente")],
        "fora_do_limite": [v["exame"] for v in verifications if v["fora_do_limite"]],
        "inconsistencias": issues,
    }
