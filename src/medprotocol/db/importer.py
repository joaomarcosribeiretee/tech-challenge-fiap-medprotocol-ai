"""Construção do banco SQLite a partir dos CSVs sintéticos (único módulo que escreve no banco).

Fluxo:
1. Valida o dataset completo com `validate_dataset`; qualquer erro interrompe a importação.
2. Cria o banco em um arquivo temporário no mesmo diretório, numa única transação.
3. Substitui o banco de destino de forma atômica (`os.replace`).

Assim, uma importação com falha nunca deixa um banco parcial nem apaga o anterior.
O resultado é determinístico: os mesmos CSVs geram o mesmo conteúdo (sem timestamps).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from medprotocol.data import schedule
from medprotocol.data.loaders import (
    CONSULTATION_FIELDS,
    CONSULTATIONS_FILE,
    EXAM_FIELDS,
    EXAMS_FILE,
    PATIENT_FIELDS,
    PATIENTS_FILE,
    load_dataset,
)
from medprotocol.data.validation import validate_dataset
from medprotocol.db.schema import SCHEMA_SQL, SCHEMA_VERSION


class DatasetImportError(RuntimeError):
    """O dataset não pôde ser importado; `errors` lista os motivos."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


@dataclass(frozen=True)
class BuildResult:
    db_path: Path
    counts: dict[str, int]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _insert(conn: sqlite3.Connection, table: str, fields: tuple[str, ...], rows) -> None:
    # Nomes de tabela/colunas vêm de constantes do código, nunca de entrada externa.
    placeholders = ", ".join("?" for _ in fields)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({placeholders})",
        ([row[field] for field in fields] for row in rows),
    )


def build_database(
    data_dir: Path,
    db_path: Path,
    reference_date: date = schedule.REFERENCE_DATE,
) -> BuildResult:
    data_dir, db_path = Path(data_dir), Path(db_path)

    report = validate_dataset(data_dir, reference_date)
    if not report.ok:
        raise DatasetImportError(
            f"dataset inválido ({len(report.errors)} erro(s)); banco não foi gerado", report.errors
        )
    dataset = load_dataset(data_dir)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)

    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_SQL)
        with conn:
            _insert(conn, "patients", PATIENT_FIELDS, dataset.patients.values())
            _insert(conn, "consultations", CONSULTATION_FIELDS, dataset.consultations.values())
            exams = [{**row, "valor": float(row["valor"])} for row in dataset.exams.values()]
            _insert(conn, "exams", EXAM_FIELDS, exams)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "data_referencia": reference_date.isoformat(),
            }
            for name in (PATIENTS_FILE, CONSULTATIONS_FILE, EXAMS_FILE):
                metadata[f"sha256:{name}"] = _sha256(data_dir / name)
            conn.executemany("INSERT INTO build_metadata (key, value) VALUES (?, ?)", sorted(metadata.items()))
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise DatasetImportError("violação de chave estrangeira", [str(v) for v in violations])
    except sqlite3.Error as exc:
        conn.close()
        tmp_path.unlink(missing_ok=True)
        raise DatasetImportError(f"falha ao gravar o banco: {exc}") from exc
    except BaseException:
        conn.close()
        tmp_path.unlink(missing_ok=True)
        raise
    conn.close()
    os.replace(tmp_path, db_path)

    return BuildResult(
        db_path=db_path,
        counts={
            "patients": len(dataset.patients),
            "consultations": len(dataset.consultations),
            "exams": len(dataset.exams),
        },
    )
