"""Importação e consultas SQLite, sempre sobre bancos temporários (tmp_path)."""

import shutil
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import pytest

from medprotocol.data import schedule
from medprotocol.data.loaders import load_dataset
from medprotocol.db.importer import DatasetImportError, build_database
from medprotocol.db.repository import (
    DatabaseNotFoundError,
    PatientNotFoundError,
    SchemaVersionError,
    connect_readonly,
    get_latest_exam,
    get_patient,
    get_patient_summary,
    list_consultations,
    list_exams,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "medprotocol.db"
    build_database(DATA_DIR, path)
    return path


@pytest.fixture
def conn(db_path):
    with connect_readonly(db_path) as connection:
        yield connection


@pytest.fixture
def data_copy(tmp_path):
    target = tmp_path / "data"
    shutil.copytree(DATA_DIR, target)
    return target


def replace_in(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# --- Importação ---------------------------------------------------------------------------------


def test_build_imports_all_rows(tmp_path):
    result = build_database(DATA_DIR, tmp_path / "out" / "db.sqlite")
    assert result.counts == {"patients": 10, "consultations": 10, "exams": 56}
    assert not (tmp_path / "out" / "db.sqlite.tmp").exists()


def test_build_is_reproducible(tmp_path):
    first, second = tmp_path / "a.db", tmp_path / "b.db"
    build_database(DATA_DIR, first)
    build_database(DATA_DIR, second)
    def dump(path):
        with closing(sqlite3.connect(path)) as raw:
            return list(raw.iterdump())

    assert dump(first) == dump(second)


def test_build_records_source_hashes_and_reference_date(conn):
    metadata = dict(conn.execute("SELECT key, value FROM build_metadata").fetchall())
    assert metadata["data_referencia"] == "2026-06-30"
    assert metadata["schema_version"] == "1"
    assert {k for k in metadata if k.startswith("sha256:")} == {
        "sha256:synthetic/patients.csv",
        "sha256:synthetic/consultations.csv",
        "sha256:synthetic/exams.csv",
    }


def test_inconsistent_dataset_is_rejected_and_previous_db_kept(data_copy, tmp_path):
    target = tmp_path / "db.sqlite"
    build_database(DATA_DIR, target)
    before = target.read_bytes()

    replace_in(data_copy / "synthetic/exams.csv", "EX-0001,PAC-0001", "EX-0001,PAC-9999")
    with pytest.raises(DatasetImportError) as excinfo:
        build_database(data_copy, target)

    assert any("PAC-9999" in error for error in excinfo.value.errors)
    assert target.read_bytes() == before
    assert not (tmp_path / "db.sqlite.tmp").exists()


def test_schema_enforces_foreign_keys_and_types(db_path, tmp_path):
    copy = tmp_path / "copy.db"
    shutil.copy(db_path, copy)
    with closing(sqlite3.connect(copy)) as raw:
        raw.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute("INSERT INTO exams VALUES ('EX-9999', 'PAC-9999', 'LDL', 90.0, 'mg/dL', '2026-01-01')")
        with pytest.raises(sqlite3.IntegrityError):  # STRICT: texto não numérico em coluna REAL
            raw.execute("INSERT INTO exams VALUES ('EX-9998', 'PAC-0001', 'LDL', 'alto', 'mg/dL', '2026-01-01')")
        with pytest.raises(sqlite3.IntegrityError):  # data inexistente
            raw.execute("INSERT INTO exams VALUES ('EX-9997', 'PAC-0001', 'LDL', 90.0, 'mg/dL', '2026-02-30')")


# --- Conexão ------------------------------------------------------------------------------------


def test_missing_database_raises_and_is_not_created(tmp_path):
    missing = tmp_path / "nao_existe.db"
    with pytest.raises(DatabaseNotFoundError):
        with connect_readonly(missing):
            pass
    assert not missing.exists()


def test_connection_is_read_only(conn):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM patients")


def test_foreign_file_is_rejected(tmp_path):
    other = tmp_path / "outro.db"
    with closing(sqlite3.connect(other)) as raw:
        raw.execute("CREATE TABLE x (y)")
    with pytest.raises(SchemaVersionError):
        with connect_readonly(other):
            pass


def test_schema_version_mismatch_is_rejected(db_path, tmp_path):
    copy = tmp_path / "old.db"
    shutil.copy(db_path, copy)
    with closing(sqlite3.connect(copy)) as raw, raw:
        raw.execute("UPDATE build_metadata SET value = '0' WHERE key = 'schema_version'")
    with pytest.raises(SchemaVersionError):
        with connect_readonly(copy):
            pass


# --- Consultas ----------------------------------------------------------------------------------


def test_get_patient_parses_lists(conn):
    patient = get_patient(conn, "PAC-0003")
    assert patient.comorbidades == ("hipertensao", "dislipidemia")
    assert patient.alergias == ()  # "nenhuma" no CSV
    assert patient.ano_nascimento == 1961


def test_unknown_patient_raises_in_every_query(conn):
    for query in (get_patient, list_consultations, list_exams, get_patient_summary):
        with pytest.raises(PatientNotFoundError, match="PAC-9999"):
            query(conn, "PAC-9999")
    with pytest.raises(PatientNotFoundError):
        get_latest_exam(conn, "PAC-9999", "HBA1C")


@pytest.mark.parametrize("bad_id", ["PAC-0001' OR '1'='1", "PAC-0001; DROP TABLE patients", "pac-0001", "", None, 1])
def test_malformed_patient_id_is_rejected_before_sql(conn, bad_id):
    with pytest.raises(ValueError, match="patient_id inválido"):
        get_patient(conn, bad_id)


def test_unknown_exam_type_is_rejected(conn):
    with pytest.raises(ValueError, match="tipo de exame inválido"):
        list_exams(conn, "PAC-0001", "LDL' OR 1=1 --")


def test_list_consultations_and_exams_are_ordered(conn):
    assert [c.consultation_id for c in list_consultations(conn, "PAC-0006")] == ["CON-0006"]
    exams = list_exams(conn, "PAC-0001")
    assert [e.data_coleta for e in exams] == sorted(e.data_coleta for e in exams)
    assert [e.exam_id for e in list_exams(conn, "PAC-0001", "HBA1C")] == ["EX-0007", "EX-0001"]


def test_patient_without_exam_of_type_returns_none(conn):
    assert list_exams(conn, "PAC-0003", "CREATININA") == []
    assert get_latest_exam(conn, "PAC-0003", "CREATININA") is None


def test_latest_exam_respects_reference_date(conn):
    assert get_latest_exam(conn, "PAC-0001", "HBA1C").exam_id == "EX-0001"
    assert get_latest_exam(conn, "PAC-0001", "HBA1C", until=date(2025, 12, 31)).exam_id == "EX-0007"


def test_latest_exam_matches_schedule_selection(conn):
    """SQL e schedule.py devem escolher o mesmo exame (mesma regra de desempate)."""
    for patient_id in load_dataset(DATA_DIR).patients:
        exams = [{**e.__dict__} for e in list_exams(conn, patient_id)]
        for tipo in schedule.MONITORED_EXAMS:
            latest = get_latest_exam(conn, patient_id, tipo)
            expected = schedule.verification(tipo, exams)["ultimo_exame"]
            assert (latest.exam_id if latest else None) == expected, (patient_id, tipo)


# --- Resumo -------------------------------------------------------------------------------------


def test_summary_reproduces_approved_dataset_expectations(conn):
    """Para todo caso com verificacoes_esperadas, o resumo do banco traz o mesmo resultado."""
    dataset = load_dataset(DATA_DIR)
    checked = 0
    for records in dataset.qa.values():
        for record in records:
            if "verificacoes_esperadas" not in record:
                continue
            summary = get_patient_summary(conn, record["patient_id"])
            by_exam = {v["exame"]: v for v in summary["verificacoes"]}
            for expected in record["verificacoes_esperadas"]:
                assert by_exam[expected["exame"]] == expected, record["qa_id"]
                checked += 1
    assert checked > 30


def test_summary_patient_up_to_date(conn):
    summary = get_patient_summary(conn, "PAC-0001")
    assert summary["pendencias"] == []
    assert summary["fora_do_limite"] == []
    assert summary["ultimos_exames"]["HBA1C"]["exam_id"] == "EX-0001"


def test_summary_patient_with_overdue_and_missing(conn):
    summary = get_patient_summary(conn, "PAC-0010")
    status = {v["exame"]: v["status"] for v in summary["verificacoes"]}
    assert status == {"HBA1C": "ausente", "CREATININA": "vencido", "TFG": "vencido", "RAC": "ausente", "LDL": "em_dia"}
    assert summary["pendencias"] == ["HBA1C", "CREATININA", "TFG", "RAC"]
    assert summary["fora_do_limite"] == ["LDL"]
    assert "HBA1C" not in summary["ultimos_exames"]


def test_summary_flags_exam_after_reference_date(conn):
    summary = get_patient_summary(conn, "PAC-0003", reference_date=date(2026, 1, 1))
    assert any("EX-0014" in issue for issue in summary["inconsistencias"])
    assert {v["exame"]: v["status"] for v in summary["verificacoes"]}["HBA1C"] == "ausente"
