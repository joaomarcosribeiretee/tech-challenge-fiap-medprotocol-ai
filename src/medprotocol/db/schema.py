"""Schema SQLite dos dados estruturados.

Tabelas STRICT, CHECKs e chaves estrangeiras funcionam como segunda barreira de
consistência, independente do validador do dataset. Ao mudar o schema, incremente
SCHEMA_VERSION: a camada de leitura recusa bancos de outra versão.
"""

SCHEMA_VERSION = "1"

SCHEMA_SQL = """
CREATE TABLE patients (
    patient_id          TEXT PRIMARY KEY CHECK (patient_id GLOB 'PAC-[0-9][0-9][0-9][0-9]'),
    sexo                TEXT NOT NULL CHECK (sexo IN ('F', 'M')),
    ano_nascimento      INTEGER NOT NULL,
    ano_diagnostico_dm2 INTEGER NOT NULL CHECK (ano_diagnostico_dm2 >= ano_nascimento),
    comorbidades        TEXT NOT NULL,
    medicamentos_em_uso TEXT NOT NULL,
    alergias            TEXT NOT NULL
) STRICT;

CREATE TABLE consultations (
    consultation_id TEXT PRIMARY KEY CHECK (consultation_id GLOB 'CON-[0-9][0-9][0-9][0-9]'),
    patient_id      TEXT NOT NULL REFERENCES patients (patient_id),
    data            TEXT NOT NULL CHECK (date(data) IS data),
    tipo            TEXT NOT NULL CHECK (tipo IN ('ambulatorial', 'internacao')),
    protocol_id     TEXT NOT NULL CHECK (protocol_id GLOB 'PROT-DM2-[0-9][0-9][0-9]'),
    resumo          TEXT NOT NULL
) STRICT;

CREATE INDEX idx_consultations_patient_data ON consultations (patient_id, data);

CREATE TABLE exams (
    exam_id     TEXT PRIMARY KEY CHECK (exam_id GLOB 'EX-[0-9][0-9][0-9][0-9]'),
    patient_id  TEXT NOT NULL REFERENCES patients (patient_id),
    tipo        TEXT NOT NULL,
    valor       REAL NOT NULL,
    unidade     TEXT NOT NULL,
    data_coleta TEXT NOT NULL CHECK (date(data_coleta) IS data_coleta)
) STRICT;

CREATE INDEX idx_exams_patient_tipo_data ON exams (patient_id, tipo, data_coleta);

CREATE TABLE build_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;
"""
